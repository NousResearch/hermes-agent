"""hermes-ssh 插件核心引擎。

组成：
- SSH 配置解析（~/.ssh/config，ProxyJump 多级展开）
- 凭据装配（~/.hermes/.env 免密 + 公钥优先；缺失给设置指引，认证失败进冷却隔离）
- 会话链 SSHSession（嵌套 client 跳板、欠账式保活、监督自愈、并发闸、pin/TTL 回收）
- SessionPool（跨工具调用复用 + Reaper 周期回收 + 总量上限）

调优参数：全部经 Tunings 动态读取（注册层注入 config reader），
用户配置 plugins.entries.hermes-ssh.settings.* > schema default > 代码兜底。
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shlex
import select as _select
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import paramiko

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 100_000
DEFAULT_ENV_PATH = "~/.hermes/.env"
ENV_PATH = os.environ.get("HERMES_SSH_ENV_FILE", DEFAULT_ENV_PATH)


def _load_official_env_map() -> Tuple[Dict[str, str], List[str]]:
    """复用 hermes-agent 官方 load_hermes_dotenv 的完整加载语义。

    按官方优先级链解析 env（不触碰进程环境，纯读取）：
      1. ~/.hermes/.env                    （用户档）
      2. /etc/hermes/.env                  （managed scope，机器全局、权重最高，
                                             见 managed_scope.get_managed_dir +
                                             env_loader._apply_managed_env override=True）
      3. HERMES_MANAGED_DIR 覆盖目录/.env   （IT 部署覆盖层）

    解析器直接 import 官方模块保持单一事实源；任何一层不存在则跳过。
    返回 (合并后的 {key: value}, warnings)。"""
    data: Dict[str, str] = {}
    warnings: List[str] = []
    try:
        sys.path.insert(0, "/home/vm/.hermes/hermes-agent")
        from hermes_cli.managed_scope import get_managed_dir  # noqa: E402

        candidates: List[Path] = [Path(os.path.expanduser(ENV_PATH))]
        managed_dir = None
        try:
            managed_dir = get_managed_dir()
        except Exception as exc:  # noqa: BLE001 — 与官方 fail-open 一致
            warnings.append(f"managed scope 解析失败: {exc}")
        if managed_dir is not None:
            candidates.append(Path(managed_dir) / ".env")

        for cand in candidates:
            if not cand.exists():
                continue
            try:
                for raw in cand.read_text(encoding="utf-8",
                                          errors="replace").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, val = line.partition("=")
                    k = k.strip()
                    if k.startswith("export "):
                        k = k[len("export "):].strip()
                    val = val.strip().strip('"').strip("'")
                    if k:
                        data[k] = val   # 后读的覆盖先读的 → managed 殿后 = 权重最高
            except OSError as exc:
                warnings.append(f"{cand} 读取失败: {exc}")
    except ImportError as exc:
        warnings.append(f"官方 managed_scope 导入失败({exc})，回退单文件 {ENV_PATH}")
    return data, warnings


# ===========================================================================
# 直推当前对话 (方案B): ssh_exec 的命令+输出以独立消息卡片直发聊天,
# 不依赖 AI 总结。全部走官方通道: session_context 取路由 → gateway loop 桥。
# 飞书单条消息上限约 8000 字符, 超长自动分段。

_FEISHU_MSG_LIMIT = 7500


def _resolve_chat_route() -> tuple:
    """(platform_str, chat_id) —— 当前 agent 会话的投递路由; 无会话上下文返回 ("","")。"""
    try:
        from gateway.session_context import get_session_env  # noqa: PLC0415
        platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
        chat_id = (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
        return platform, chat_id
    except Exception:
        return "", ""


def _split_chunks(text: str, limit: int = _FEISHU_MSG_LIMIT) -> List[str]:
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    rest = text
    while rest:
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    return chunks


def echo_to_chat(command: str, exit_code: Any, stdout: str, stderr: str,
                 target_label: str, dur_ms: int, status_note: str = "") -> None:
    """把一条 ssh 执行回执直推到当前对话(仅消息型会话)。绝不抛异常。"""
    try:
        # 配置开关: plugins.entries.hermes-ssh.settings.echo.card_enabled
        # (schema 默认 true; 每次调用时动态读, 改配置无需重启即生效)
        if not bool(cfg("echo.card_enabled", True)):
            logger.info("ssh-echo skip: card disabled by config "
                        "(plugins.entries.hermes-ssh.settings.echo.card_enabled=false)")
            return

        platform, chat_id = _resolve_chat_route()
        if not chat_id or not platform:
            logger.info("ssh-echo skip: no chat route (CLI/cron/ctx missing)")
            return  # CLI / cron / 测试进程 — 无聊天可推, 静默退出

        # ⚠ 双副本陷阱: `python -m gateway.run` 把 run.py 作为 __main__ 执行,
        # 活的 weakref 写在 __main__ 的 globals 里; 此处若 `from gateway.run
        # import _gateway_runner_ref` 会触发创建第二份模块副本, 其 ref 恒为
        # 默认 lambda:None (实测 2026-09-07: 日志 "skip: runner ref is None").
        # 正确姿势: 从 sys.modules['__main__'] 直取同进程活引用.
        runner = None
        main_mod = sys.modules.get("__main__")
        ref = getattr(main_mod, "_gateway_runner_ref", None)
        if callable(ref):
            try:
                runner = ref()
            except Exception:  # noqa: BLE001 — dead weakref 等, 尽力而为
                runner = None
        if runner is None and main_mod is not None \
                and getattr(main_mod, "__name__", "") != "gateway.run":
            # 兜底2: 非 -m 场景(如 python run.py 直接跑), 副本也可能被注册过
            grun = sys.modules.get("gateway.run")
            ref2 = getattr(grun, "_gateway_runner_ref", None)
            if callable(ref2):
                try:
                    runner = ref2()
                except Exception:  # noqa: BLE001
                    runner = None
        if runner is None:
            logger.info("ssh-echo skip: runner ref is None "
                        "(main=%s, gateway.run in modules=%s)",
                        getattr(main_mod, "__name__", None),
                        "gateway.run" in sys.modules)
            return
        loop = getattr(runner, "_gateway_loop", None)
        closed = (loop.is_closed() if hasattr(loop, "is_closed") else False)
        if loop is None or closed:
            logger.info("ssh-echo skip: gateway loop dead (closed=%s)", closed)
            return

        from gateway.config import Platform  # noqa: PLC0415
        try:
            adapter = runner.adapters.get(Platform(platform))
        except (ValueError, KeyError):
            adapter = None
        if adapter is None:
            logger.info("ssh-echo skip: no adapter for platform %s "
                        "(registered: %s)", platform,
                        [getattr(k, 'value', k) for k in runner.adapters])
            return

        logger.info("ssh-echo dispatching to %s/%s via %s",
                    platform, chat_id, type(adapter).__name__)

        # ---- 组飞书交互卡片 (interactive card) ----
        # 状态三档 (布布 2026-09-07 定稿方案A):
        #   exit=0                    → ✅ 绿 (成功)
        #   exit≠0 但 stdout 非空      → 🔶 橙 "完成·有告警" (探测类命令预期内非零,
        #                                如 which/grep 无命中 — 有实际产出不算失败)
        #   TIMEOUT / 无输出且非零      → ❌ 红 (真失败)
        _timeout_hit = str(exit_code).upper() == "TIMEOUT"
        if not _timeout_hit and exit_code == 0:
            icon, template, verdict_note = "✅", "green", ""
        elif not _timeout_hit and stdout.strip():
            icon, template = "🔶", "orange"
            verdict_note = f"exit={exit_code} · 完成(有告警), 输出见下"
        else:
            icon = "⏱" if _timeout_hit else "❌"
            template = "orange" if _timeout_hit else "red"
            verdict_note = ""

        def _esc(s: str) -> str:
            # 只清 \\r (飞书 markdown 不需要手动转义引号/反斜杠 —
            # JSON 序列化由 json.dumps 统一处理; 手动加 \\ 会渲染成
            # 可见的反斜杠尸体, 实测 2026-09-07 NAME=\"TencentOS\")
            return s.replace("\r", "").rstrip("\n")

        from datetime import datetime as _dt  # noqa: PLC0415
        ts = _dt.now().strftime("%H:%M:%S")

        # ⚠ 飞书卡片 JSON 1.0: div+lark_md 的 text 组件【不支持```代码块】,
        # 必须用独立 "tag":"markdown" 元素(官方富文本组件, 支持全部子集语法).
        body_lines = [f"**command:**\n```\n{_esc(command[:2000])}\n```"]
        if stdout:
            body_lines.append(f"```\n{_esc(stdout[:7000])}\n```")
        if stderr:
            body_lines.append("**stderr:**\n```\n%s\n```" % _esc(stderr[:3000]))
        if status_note:
            body_lines.append(f"*{status_note}*")
        elif verdict_note:
            body_lines.append(f"<font color='orange'>ℹ️ {verdict_note}</font>")
        body_lines.append(f"<font color='grey'>🕐 {ts}</font>")

        elements = [{
            "tag": "markdown",
            "content": "\n".join(body_lines),
        }]
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"{icon} ssh · {target_label} · exit={exit_code} · {dur_ms}ms",
                },
                "template": template,
            },
            "elements": elements,
        }

        import json as _json  # noqa: PLC0415
        coros = []
        chunks = _split_chunks(_json.dumps(card, ensure_ascii=False), limit=28000)
        for i, chunk in enumerate(chunks):
            payload = chunk if len(chunks) == 1 else \
                _json.dumps({
                    "config": {"wide_screen_mode": True},
                    "header": {"title": {"tag": "plain_text",
                                         "content": f"{icon} ssh · {target_label} (续{i})"},
                               "template": template},
                    "elements": [{"tag": "markdown",
                                  "content": "⚠️ 卡片超长被拆分，本段为原始 JSON 片段：" + chunk}],
                }, ensure_ascii=False)
            # 官方同型先例: send_exec_approval/send_update_prompt 走
            # _feishu_send_with_retry(msg_type="interactive"); 飞书 adapter
            # 与本插件同仓同步演进, 直接复用其内部发送管线.
            coros.append(adapter._feishu_send_with_retry(
                chat_id=chat_id,
                msg_type="interactive",
                payload=payload,
                reply_to=None,
                metadata=None,
            ))
        for coro in coros:
            from agent.async_utils import safe_schedule_threadsafe  # noqa: PLC0415
            fut = safe_schedule_threadsafe(
                coro, loop,
                logger=logger,
                log_message="ssh-echo push failed",
                log_level=logging.WARNING,
            )
            if fut is not None:
                def _log_echo_done(f, label=target_label):  # noqa: E306
                    try:
                        res = f.result()
                        # 兼容两种返回形态: SendResult(.success bool) 与
                        # 飞书 SDK CreateMessageResponse(.success() 方法 +
                        # resp.data.message_id). 按实际类型解析.
                        s_fn = getattr(res, "success", None)
                        ok = s_fn() if callable(s_fn) else bool(
                            getattr(res, "success", True))
                        data = getattr(res, "data", None)
                        mid = (getattr(data, "message_id", None)
                               or getattr(res, "message_id", None))
                        err = getattr(res, "error", None) or getattr(res, "msg", None)
                        (logger.info if ok else logger.warning)(
                            "ssh-echo delivered to %s: success=%s mid=%s err=%s",
                            label, ok, mid, err)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("ssh-echo future failed: %s", e)
                fut.add_done_callback(_log_echo_done)
            else:
                logger.warning("ssh-echo schedule returned None "
                               "(loop mismatch or shutdown)")
    except Exception:  # noqa: BLE001 — 推送是尽力而为, 绝不影响执行主流程
        pass


# ===========================================================================
# 异常族
# ===========================================================================


class SSHParseError(ValueError):
    """target/配置无法解析。"""


class SSHSetupRequired(RuntimeError):
    """缺少免密凭据 —— message 含可复制执行的设置命令。"""

    def __init__(self, hops: List[Dict[str, str]]):
        self.hops = hops
        lines = [
            "无法免密登录：以下主机既没有可用公钥，也没有配置密码。",
            "请在宿主机执行对应命令，把 SSH 密码写入 Hermes .env：",
            "",
        ]
        for h in hops:
            note = f"（{h['note']}）" if isinstance(h, dict) and h.get("note") else ""
            env_var = str(h)
            host_desc = "?"
            if isinstance(h, dict):
                env_var = h.get("env_var", "?")
                host_desc = f"{h.get('user', '?')}@{h.get('host', '?')}"
            lines.append(
                f"echo 'export {env_var}=\"该主机({host_desc})的SSH密码\"' >> {ENV_PATH}"
                + note
            )
        lines += [
            "",
            f"· 密码仅存于本机 {ENV_PATH}，不入库不出网。",
            "· 写入后对新连接立即生效，无需重启网关。",
            "· 公钥路线：装好公钥后在 ~/.ssh/config 对应 Host 下设 PubkeyAuthentication yes。",
        ]
        super().__init__("\n".join(lines))


class SSHAuthCooldown(RuntimeError):
    """凭据处于认证失败冷却期；期间绝不带旧密码重试（防堡垒机锁号）。"""

    def __init__(self, node_desc: str, remaining_s: float, env_var: str):
        self.node_desc = node_desc
        self.remaining_s = remaining_s
        self.env_var = env_var
        super().__init__(
            f"{node_desc}: 上次认证失败，凭据冷却中（还剩 {remaining_s:.0f}s）。"
            f"期间不会重试以免触发账户锁定；若密码已改，请更新 {env_var} "
            f"（保存 {ENV_PATH} 即解除冷却）。"
        )


class SSHAuthPermanentFailure(RuntimeError):
    """同一凭据累计认证失败达上限，停止一切自动重试。"""

    def __init__(self, node_desc: str, env_var: str):
        self.node_desc = node_desc
        self.env_var = env_var
        super().__init__(
            f"{node_desc}: 该凭据已累计 {AUTH_FAIL_LIMIT} 次认证失败，已停用自动重连。"
            f"请确认密码并更新 {env_var} 后重试（保存 {ENV_PATH} 自动复位）。"
        )


AUTH_FAIL_LIMIT = 3


# ===========================================================================
# ~/.ssh/config 解析
# ===========================================================================

_TOKEN_SPLIT_RE = re.compile(r"\s+")

_KEYMAP = {
    "proxyjump": "proxyjump",
    "hostname": "hostname",
    "user": "username",
    "port": "port",
    "identityfile": "identityfile",
    "pubkeyauthentication": "pubkeyauthentication",
    "preferredauthentications": "preferredauthentications",
}


def parse_ssh_config(path: str = "~/.ssh/config") -> Dict[str, Dict[str, Any]]:
    cfg_path = Path(os.path.expanduser(path))
    if not cfg_path.exists():
        return {}

    entries: Dict[str, Dict[str, Any]] = {}
    current_patterns: List[str] = []

    for raw_line in cfg_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = _TOKEN_SPLIT_RE.split(line, maxsplit=1)
        key = parts[0].lower()
        value = parts[1].strip().strip('"') if len(parts) > 1 else ""

        if key == "match":
            current_patterns = []               # Match 块不支持：终止当前段
            continue
        if key == "host":
            current_patterns = [p for p in _TOKEN_SPLIT_RE.split(value) if p]
            for pat in current_patterns:
                entries.setdefault(pat, {})
            continue
        if not current_patterns:
            continue
        norm = _KEYMAP.get(key, key.replace("-", "").replace("_", ""))
        for pat in current_patterns:
            entry = entries.setdefault(pat, {})
            if norm == "identityfile":
                lst = entry.get("identityfile")
                entry["identityfile"] = (lst or []) + [value]
            else:
                entry[norm] = value
    return entries


def _glob_to_re(glob: str) -> str:
    out = ""
    for ch in glob:
        if ch == "*":
            out += ".*"
        elif ch == "?":
            out += "."
        else:
            out += re.escape(ch)
    return out


def _pattern_matches(label: str, pattern: str) -> bool:
    if pattern.startswith("!"):
        return False
    return bool(re.fullmatch(_glob_to_re(pattern), label))


def resolve_host(entries: Dict[str, Dict[str, Any]], label: str) -> Dict[str, Any]:
    """OpenSSH 语义：先命中先生效，后续命中只补未设字段。"""
    merged: Dict[str, Any] = {}
    for pattern, params in entries.items():
        if any(_pattern_matches(label, p) for p in pattern.split()):
            for k, v in params.items():
                if k == "identityfile":
                    merged["identityfile"] = (merged.get("identityfile") or []) + list(v)
                elif k not in merged:
                    merged[k] = v
    return merged


# ===========================================================================
# ~/.hermes/.env 运行时读取（mtime 缓存 → 改完即时生效）
# ===========================================================================

_env_cache: Dict[str, Any] = {"mtime": None, "data": {}, "warnings": [], "exists": None}


def get_env_mtime() -> Optional[float]:
    """多源 env 的变化指纹：任一文件 mtime 变化即视为 env 更新。

    覆盖用户档 (~/.hermes/.env) + managed 档 (/etc/hermes/.env 或
    $HERMES_MANAGED_DIR)，与 _load_official_env_map 的候选集一致。"""
    parts: List[Optional[int]] = []
    try:
        sys.path.insert(0, "/home/vm/.hermes/hermes-agent")
        from hermes_cli.managed_scope import get_managed_dir  # noqa: E402

        candidates: List[Path] = [Path(os.path.expanduser(ENV_PATH))]
        try:
            md = get_managed_dir()
        except Exception:  # noqa: BLE001
            md = None
        if md is not None:
            candidates.append(Path(md) / ".env")

        for cand in candidates:
            try:
                parts.append(cand.stat().st_mtime_ns)
            except OSError:
                parts.append(None)
    except ImportError:
        path = Path(os.path.expanduser(ENV_PATH))
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None
    # 组合成单一指纹（mtime 缺失用 -1 占位，存在性变化也会改变指纹）
    return hash(tuple(p if p is not None else -1 for p in parts))


def get_env() -> Tuple[Dict[str, str], List[str]]:
    """多源 env 读取（官方 load_hermes_dotenv 语义的纯读版）。

    候选：~/.hermes/.env + managed 档(/etc/hermes/.env 等)。
    mtime 组合指纹做缓存键，任一档变化即失效重载。"""
    try:
        fingerprint = get_env_mtime()
    except Exception:  # noqa: BLE001
        fingerprint = None

    if fingerprint is not None and _env_cache["mtime"] == fingerprint \
            and _env_cache["exists"]:
        return _env_cache["data"], _env_cache["warnings"]

    data, warnings = _load_official_env_map()

    user_exists = Path(os.path.expanduser(ENV_PATH)).exists()
    changed = _env_cache["exists"] is not False or bool(data)
    if not user_exists and not data:
        _env_cache.update(exists=False, mtime=None, data={}, warnings=[])
        return {}, ([f"{ENV_PATH} 不存在"] if changed else [])

    _env_cache.update(exists=True, mtime=fingerprint, data=data,
                      warnings=warnings)
    return data, warnings


def _env_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.upper()).strip("_")


# ---- sudo -S 注入工具 (sudo 自动喂密码链路, 见 exec_command) ----

_SUDO_STDIN_FLAGS = {"-S", "--stdin"}
_SUDO_NOPASS_FLAGS = {"-n", "--non-interactive", "-A", "--askpass"}


def command_has_sudo_stdin_flag(command: str) -> bool:
    """命令里(任一 sudo 词元后)已带 -S/-n/-A 就不重复注入。"""
    toks = shlex.split(command)
    for i, tok in enumerate(toks):
        if tok == "sudo" or tok.endswith("/sudo"):
            for t2 in toks[i + 1:]:
                if t2 in _SUDO_STDIN_FLAGS or t2 in _SUDO_NOPASS_FLAGS:
                    return True
                if not t2.startswith("-"):   # 第一个非 flag 即子命令, 停止
                    break
    return False


def inject_sudo_dash_S(command: str) -> str:
    """给首个裸 sudo 注入 -S; 只动第一个 token 位, 管道/组合跳过。"""
    try:
        toks = shlex.split(command)
    except ValueError:
        return command
    for i, tok in enumerate(toks):
        if tok == "sudo" or tok.endswith("/sudo"):
            return " ".join(toks[:i + 1] + ["-S"] + toks[i + 1:])
    return command


def _password_candidates(node: Dict[str, Any]) -> List[str]:
    label_tok = _env_token(node["_label"])
    host_tok = _env_token(node["hostname"])
    user_tok = _env_token(node["username"]) if node.get("username") else ""

    keys: List[str] = []
    if user_tok:
        keys.append(f"HERMES_SSH_PASSWORD_{label_tok}_{user_tok}")
    keys.append(f"HERMES_SSH_PASSWORD_{label_tok}")
    if host_tok and host_tok != label_tok:
        if user_tok:
            keys.append(f"HERMES_SSH_PASSWORD_{host_tok}_{user_tok}")
        keys.append(f"HERMES_SSH_PASSWORD_{host_tok}")
    hn = node["hostname"].lower()
    if "bastionhost" in hn or "bastion" in node["_label"].lower():
        keys.append("HERMES_SSH_BASTION_PASSWORD")
    deduped: List[str] = []
    for k in keys:
        if k not in deduped:
            deduped.append(k)
    return deduped


def node_identity(cfg: Dict[str, Any]) -> Tuple[str, int, str]:
    return (str(cfg["hostname"]).lower(), int(cfg["port"]), str(cfg["username"]))


# ===========================================================================
# 凭据冷却隔离（Q1：密码错了绝不能撞）
# ===========================================================================


class AuthIsolator:
    """per-node 认证失败隔离区。

    - 单次 AuthenticationException → cooldown（默认 10min）
    - 累计 AUTH_FAIL_LIMIT 次 → permanent：不再自动重试
    - .env mtime 变化 → 全部复位（用户改了密码是最明确的修正信号）
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        # None = 尚未观察过 .env；首次 reset_if_env_changed 只记录基线不清状态
        self.last_seen_mtime: Optional[float] = None
        self._baseline_set = False

    def reset_if_env_changed(self) -> bool:
        m = get_env_mtime()
        with self.lock:
            if not self._baseline_set:
                self._baseline_set = True
                self.last_seen_mtime = m
                return False
            if m != self.last_seen_mtime:
                self.last_seen_mtime = m
                if self.state:
                    logger.info("auth isolator: %s cleared by .env change", len(self.state))
                self.state.clear()
                return True
            return False

    def record_failure(self, identity: Tuple[str, int, str],
                       env_var: str, cooldown_s: float,
                       error: str) -> Tuple[int, bool, float]:
        """返回 (累计次数, 是否 permanent, 冷却剩余秒)。"""
        with self.lock:
            st = self.state.setdefault(identity, {"count": 0})
            st["count"] += 1
            st["env_var"] = env_var
            st["last_error"] = error[:300]
            permanent = st["count"] >= AUTH_FAIL_LIMIT
            until = 0.0 if permanent else time.time() + cooldown_s
            st["cooldown_until"] = until
            st["permanent"] = permanent
            return st["count"], permanent, cooldown_s

    def check(self, identity: Tuple[str, int, str],
              node_desc: str) -> None:
        """冷却/permanent 时抛出对应异常；否则放行。"""
        self.reset_if_env_changed()
        with self.lock:
            st = self.state.get(identity)
            if not st:
                return
            if st.get("permanent"):
                raise SSHAuthPermanentFailure(node_desc, st.get("env_var", "?"))
            remain = st.get("cooldown_until", 0) - time.time()
            if remain > 0:
                raise SSHAuthCooldown(node_desc, remain, st.get("env_var", "?"))

    def status(self) -> List[Dict[str, Any]]:
        self.reset_if_env_changed()
        now = time.time()
        with self.lock:
            out = []
            for ident, st in self.state.items():
                out.append({
                    "node": f"{ident[2]}@{ident[0]}:{ident[1]}",
                    "failures": st["count"],
                    "permanent": st.get("permanent", False),
                    "cooldown_remaining_s": round(max(0, st.get("cooldown_until", 0) - now), 1),
                    "env_var": st.get("env_var"),
                })
            return out


AUTH_ISOLATOR = AuthIsolator()


# ===========================================================================
# plan 构建：target 表达式 → 节点链 → 凭据装配
# ===========================================================================


def _parse_node_expr(expr: str) -> Tuple[str, Optional[str], Optional[int]]:
    expr = expr.strip()
    user: Optional[str] = None
    rest = expr
    if "@" in expr:
        user, _, rest = expr.rpartition("@")
    port: Optional[int] = None
    m = re.match(r"^\[(.+)\](?::(\d+))?$", rest)
    if m:
        rest, port_s = m.group(1), m.group(2)
        port = int(port_s) if port_s else None
    elif rest.count(":") == 1:
        host_part, maybe_port = rest.rsplit(":", 1)
        if maybe_port.isdigit():
            rest, port = host_part, int(maybe_port)
    if not rest:
        raise SSHParseError(f"空目标表达式: {expr!r}")
    return rest, (user or None), port


def _apply_overrides(d: Dict[str, Any],
                     overrides: Tuple[str, Optional[str], Optional[int]],
                     label: str) -> None:
    _, o_user, o_port = overrides
    if o_user:
        d["username"] = o_user
    if not d.get("username"):
        raise SSHParseError(f"{label}: 缺少 User（config 与表达式均未提供）")
    if o_port:
        d["port"] = o_port
    if not d.get("port"):
        d["port"] = 22
    if not d.get("hostname"):
        d["hostname"] = label


def _materialize_node(params: Dict[str, Any], label: str) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "_label": label,
        "hostname": str(params["hostname"]),
        "port": int(params["port"]),
        "username": params["username"],
    }
    pubkey_ok = str(params.get("pubkeyauthentication", "yes")).lower() not in ("no", "false")
    idfiles = params.get("identityfile") or ["~/.ssh/id_rsa"]
    if isinstance(idfiles, str):
        idfiles = [idfiles]

    picked: Optional[str] = None
    if pubkey_ok:
        for idf in idfiles:
            p = os.path.expanduser(idf)
            if os.path.isfile(p):
                picked = p
                break

    if picked:
        node["auth_kind"] = "publickey"
        node["identityfile"] = picked
    else:
        node["auth_kind"] = "password-or-kbd-interactive"
    node["env_candidates"] = _password_candidates(node)
    return node


def build_plan(target_expr: str,
               entries: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    overrides = _parse_node_expr(target_expr)
    root_label = overrides[0]

    root_resolved = dict(resolve_host(entries, root_label))
    jump_raw = root_resolved.get("proxyjump", "")
    jump_list = [j for j in jump_raw.split(";") if j and j.lower() != "none"] if jump_raw else []

    nodes: List[Dict[str, Any]] = []
    seen: set = set()

    for jexpr in jump_list:
        j_over = _parse_node_expr(jexpr)
        j_label = j_over[0]
        if j_label in seen:
            raise SSHParseError(f"跳板链出现环: {jexpr}")
        seen.add(j_label)
        j_params = dict(resolve_host(entries, j_label))
        _apply_overrides(j_params, j_over, j_label)
        nodes.append(_materialize_node(j_params, j_label))

    if root_label in seen:
        raise SSHParseError(f"目标同时是其自身跳板: {root_label}")
    _apply_overrides(root_resolved, overrides, root_label)
    nodes.append(_materialize_node(root_resolved, root_label))

    canon = "→".join(n["_label"] for n in nodes)
    return nodes, canon


def attach_auth(nodes: List[Dict[str, Any]]) -> None:
    """就地填凭据。缺失 → SSHSetupRequired；冷却 → SSHAuthCooldown。"""
    AUTH_ISOLATOR.reset_if_env_changed()
    env_data, _ = get_env()
    missing: List[Dict[str, str]] = []

    for node in nodes:
        desc = f"{node['username']}@{node['hostname']}:{node['port']}"
        # 冷却检查（无论哪种 auth 模式——冷却针对这个节点的全部凭据尝试）
        try:
            AUTH_ISOLATOR.check(node_identity(node), desc)
        except (SSHAuthCooldown, SSHAuthPermanentFailure):
            raise

        env_val: Optional[str] = None
        used_key: Optional[str] = None
        for cand in node["env_candidates"]:
            v = env_data.get(cand)
            if v:
                env_val, used_key = v, cand
                break

        if node["auth_kind"] == "publickey":
            try:
                node["pkey"] = _load_pkey(node["identityfile"])
                node["auth_used"] = "publickey"
            except Exception as exc:  # noqa: BLE001
                node.pop("pkey", None)
                logger.debug("pkey %s load failed: %s", node["_label"], exc)
                if env_val is None:
                    missing.append({"env_var": node["env_candidates"][0],
                                    "user": node["username"], "host": node["hostname"],
                                    "note": f"公钥加载失败({type(exc).__name__})需密码兜底"})
                    continue
            if env_val is not None:
                node["password"] = env_val     # 公钥失败的 second factor
        else:
            if env_val is None:
                missing.append({"env_var": node["env_candidates"][0],
                                "user": node["username"], "host": node["hostname"]})
                continue
            node["password"] = env_val
            node["auth_used"] = f"password via {used_key}"

    if missing:
        raise SSHSetupRequired(missing)


def _load_pkey(path: str) -> paramiko.PKey:
    expanded = os.path.expanduser(path)
    errs: List[str] = []
    for cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return cls.from_private_key_file(expanded)
        except paramiko.PasswordRequiredException:
            raise
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{cls.__name__}: {exc}")
    raise ValueError(f"无法加载私钥 {path}: " + "; ".join(errs)[:200])


# ===========================================================================
# 保活（欠账模型）＋ 监督者
# ===========================================================================


class _GapFiller(threading.Thread):
    """每 interval 保证至少一次出向流量；醒晚了把欠的一次性补齐。"""

    def __init__(self, transport: paramiko.Transport, interval: float,
                 on_dead: Callable[[str], None]):
        super().__init__(name=f"ssh-gapfill-{id(self):x}", daemon=True)
        self._transport = transport
        self._interval = interval
        self._wake = threading.Event()
        self._stopping = False
        self._on_dead = on_dead
        self.last_send_ts: Optional[float] = None
        self.sent_total = 0
        self.error: Optional[str] = None

    def stop(self) -> None:
        self._stopping = True
        self._wake.set()

    def run(self) -> None:
        next_due = time.monotonic() + self._interval
        while not self._stopping:
            now = time.monotonic()
            caught_up = 0
            while next_due <= now and caught_up < 240:
                try:
                    self._transport.send_ignore()
                except EOFError:
                    self.error = "EOF — TCP 半关闭"
                    self._on_dead(self.error)
                    return
                except Exception as exc:  # noqa: BLE001
                    self.error = f"{type(exc).__name__}: {exc}"
                    self._on_dead(self.error)
                    return
                self.last_send_ts = time.time()
                self.sent_total += 1
                caught_up += 1
                next_due += self._interval
            if caught_up > 1:
                logger.debug("gapfiller caught up %d windows", caught_up)
            sleep_for = max(0.05, min(self._interval, next_due - time.monotonic()))
            self._wake.wait(sleep_for)
            self._wake.clear()


# ===========================================================================
# 调优参数（settings > schema default > 代码兜底）
# ===========================================================================

_CFG_READER: Optional[Callable[[str, Any], Any]] = None


def install_cfg_reader(reader: Optional[Callable[[str, Any], Any]]) -> None:
    global _CFG_READER
    _CFG_READER = reader


def cfg(key: str, fallback: Any) -> Any:
    if _CFG_READER is not None:
        try:
            v = _CFG_READER(key, None)
            if v is not None:
                return v
        except Exception:  # noqa: BLE001
            pass
    return fallback


class Tunings:
    """动态快照 —— 每次 pool 入口刷新一次。"""

    __slots__ = ("idle_ttl", "max_sessions", "channel_limit",
                 "auth_cooldown", "keepalive", "supervisor_poll",
                 "record_output", "audit_enabled", "sudo_autofill")

    @classmethod
    def snapshot(cls) -> "Tunings":
        t = cls()
        t.idle_ttl = float(cfg("tuning.idle_ttl_seconds", 1800))
        t.max_sessions = int(cfg("tuning.max_sessions", 16))
        t.channel_limit = int(cfg("tuning.channel_limit", 8))
        t.auth_cooldown = float(cfg("tuning.auth_cooldown_seconds", 600))
        t.keepalive = float(cfg("tuning.keepalive_interval_seconds", 20))
        t.supervisor_poll = float(cfg("tuning.supervisor_poll_seconds", 5))
        t.audit_enabled = bool(cfg("audit.enabled", True))
        t.record_output = bool(cfg("audit.record_output", False))
        t.sudo_autofill = bool(cfg("tuning.sudo_autofill", True))
        return t


# ===========================================================================
# 会话链
# ===========================================================================


class SSHSessionNode:
    __slots__ = ("cfg", "client", "connected_at", "last_used")

    def __init__(self, cfg_dict: Dict[str, Any]):
        self.cfg = cfg_dict
        self.client: Optional[paramiko.SSHClient] = None
        self.connected_at: Optional[float] = None
        self.last_used: Optional[float] = None

    @property
    def transport(self) -> Optional[paramiko.Transport]:
        return self.client.get_transport() if self.client else None


class SSHSession:
    """一条跳板链。线程安全；exec 共享 transport 并受并发通道闸限制。"""

    RECONNECT_BASE_DELAY = 2.0
    RECONNECT_MAX_DELAY = 300.0
    AUTH_TIMEOUT = 45.0
    BANNER_TIMEOUT = 30.0
    CHANNEL_OPEN_TIMEOUT = 15.0
    TCP_CONNECT_TIMEOUT = 12.0

    def __init__(self, label: str, nodes_cfg: List[Dict[str, Any]],
                 pool: "SessionPool"):
        assert nodes_cfg
        self.label = label
        self.nodes_cfg = nodes_cfg
        self.pool_ref = pool
        self.lock = threading.RLock()
        self.chain: List[SSHSessionNode] = []
        self.pinned = False
        self.created_at = time.time()
        self.last_used = time.time()
        self.last_error: Optional[str] = None
        self.reconnect_attempts = 0
        self.backoff_until = 0.0
        self.built_count = 0

        self.state = threading.Event()
        self._gapfill: Optional[_GapFiller] = None
        self._reconnect_now = threading.Event()
        self._closed = False
        self._hop_locks: Dict[int, threading.RLock] = {}

        self._chan_cond = threading.Condition(self.lock)
        self._active_cmd_channels = 0

        t = threading.Thread(target=self._supervise_loop,
                             name=f"ssh-supervisor-{label}", daemon=True)
        t.start()

    # ------------------------------------------------------------ 状态 --

    def is_alive(self) -> bool:
        with self.lock:
            tr = self.chain[-1].transport if self.chain else None
            return bool(tr and tr.is_active()) and self.state.is_set() and not self._closed

    def touch(self) -> None:
        with self.lock:
            self.last_used = time.time()
            if self.chain:
                self.chain[-1].last_used = self.last_used

    def status_dict(self) -> Dict[str, Any]:
        with self.lock:
            gf = self._gapfill
            last_hk = gf.last_send_ts if gf else None
            return {
                "session": self.label,
                "state": "closed" if self._closed else (
                    "connected" if self.is_alive_unlocked() else
                    ("backoff" if time.time() < self.backoff_until else "disconnected")),
                "pinned": self.pinned,
                "idle_s": round(time.time() - self.last_used, 1),
                "chain": [
                    {
                        "label": n.cfg["_label"],
                        "target": f"{n.cfg['username']}@{n.cfg['hostname']}:{n.cfg['port']}",
                        "auth": n.cfg.get("auth_used", "?"),
                        "age_s": round(time.time() - n.connected_at, 1) if n.connected_at else None,
                    } for n in self.chain
                ],
                "keepalive_last_send": datetime.fromtimestamp(last_hk).strftime("%H:%M:%S") if last_hk else None,
                "gapfills_sent": gf.sent_total if gf else 0,
                "built_count": self.built_count,
                "reconnect_attempts": self.reconnect_attempts,
                "backoff_remaining_s": round(max(0, self.backoff_until - time.time()), 1),
                "active_cmd_channels": self._active_cmd_channels,
                "last_error": self.last_error,
            }

    def is_alive_unlocked(self) -> bool:
        tr = self.chain[-1].transport if self.chain else None
        return bool(tr and tr.is_active()) and self.state.is_set() and not self._closed

    # -------------------------------------------------------- 连接管理 --

    def ensure_connected(self, force: bool = False,
                         tunings: Optional[Tunings] = None) -> Dict[str, Any]:
        tun = tunings or Tunings.snapshot()
        with self.lock:
            if self._closed:
                raise RuntimeError("会话已关闭")
            if not force and self.is_alive_unlocked():
                self.touch()
                return {"ok": True, "reused": True, **self.status_dict()}
            if force:
                self._teardown_unlocked()
            try:
                AUTH_ISOLATOR.reset_if_env_changed()
                attach_auth(self.nodes_cfg)         # 可能抛 SetupRequired/Cooldown
                info = self._build_chain_locked(tun)
                return {"ok": True, "reused": False, "rebuilt": True, **info}
            finally:
                # 无论成败都清掉密码残留（连接期内内存持有即可）
                self._scrub_passwords()

    def reconnect_async(self) -> None:
        self._reconnect_now.set()

    def close(self) -> None:
        with self.lock:
            self._closed = True
            self._chan_cond.notify_all()
            self._teardown_unlocked()
        logger.info("session [%s] closed", self.label)

    def _teardown_unlocked(self) -> None:
        if self._gapfill is not None:
            self._gapfill.stop()
            self._gapfill = None
        for node in reversed(self.chain):
            try:
                if node.client is not None:
                    node.client.close()
            except Exception:  # noqa: BLE001
                pass
        self.chain = []
        self.state.clear()

    def _scrub_passwords(self) -> None:
        for n in self.nodes_cfg:
            n.pop("password", None)

    def _build_chain_locked(self, tun: Tunings) -> Dict[str, Any]:
        started = time.time()
        prev_transport: Optional[paramiko.Transport] = None
        hop_oplock: Optional[threading.RLock] = None
        built: List[SSHSessionNode] = []

        try:
            for idx, cfg_n in enumerate(self.nodes_cfg):
                node = SSHSessionNode(cfg_n)
                lk = threading.RLock()
                self._hop_locks[id(cfg_n)] = lk

                dst = (cfg_n["hostname"], int(cfg_n["port"]))
                if prev_transport is None:
                    sock_obj = socket.create_connection(dst, timeout=self.TCP_CONNECT_TIMEOUT)
                    sock_obj.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                else:
                    with hop_oplock:                 # type: ignore[arg-type]
                        if not prev_transport.is_active():
                            raise RuntimeError(
                                f"上一跳 {self.nodes_cfg[idx-1]['_label']} transport 已失效")
                        local_addr = (prev_transport.getpeername()[0], dst[1])
                        chan = prev_transport.open_channel(
                            "direct-tcpip", dst, local_addr,
                            timeout=self.CHANNEL_OPEN_TIMEOUT)
                    if chan is None:
                        raise RuntimeError(
                            f"{cfg_n['_label']}: direct-tcpip 被拒/超时（via {dst}）")
                    sock_obj = chan

                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                try:
                    client.connect(
                        hostname=cfg_n["hostname"],
                        port=int(cfg_n["port"]),
                        username=cfg_n["username"],
                        password=cfg_n.get("password"),
                        pkey=cfg_n.get("pkey"),
                        sock=sock_obj,
                        timeout=self.TCP_CONNECT_TIMEOUT,
                        auth_timeout=self.AUTH_TIMEOUT,
                        banner_timeout=self.BANNER_TIMEOUT,
                        allow_agent=False,
                        look_for_keys=False,
                        compress=False,
                    )
                except paramiko.AuthenticationException as exc:
                    try:
                        if client.get_transport() is not None:
                            client.close()
                    except Exception:  # noqa: BLE001
                        pass
                    count, permanent, cd = AUTH_ISOLATOR.record_failure(
                        node_identity(cfg_n),
                        (cfg_n.get("env_candidates") or ["?"])[0],
                        tun.auth_cooldown, str(exc))
                    if permanent:
                        raise SSHAuthPermanentFailure(desc_of(cfg_n),
                                                      (cfg_n.get("env_candidates") or ["?"])[0])
                    raise SSHAuthCooldown(desc_of(cfg_n), cd,
                                          (cfg_n.get("env_candidates") or ["?"])[0]) from exc
                except Exception:
                    try:
                        if client.get_transport() is not None:
                            client.close()
                    except Exception:  # noqa: BLE001
                        pass
                    raise

                transport = client.get_transport()
                if transport is None or not transport.is_active():
                    raise RuntimeError(f"{cfg_n['_label']}: 连接后 transport 无效")
                transport.set_keepalive(0)           # 用自己的 gapfiller

                node.client = client
                node.connected_at = time.time()
                node.last_used = time.time()
                node.cfg["auth_used"] = node.cfg.get("auth_used") or (
                    "publickey" if node.cfg.get("pkey") else "password/kbd-interactive")
                built.append(node)

                if idx < len(self.nodes_cfg) - 1:
                    prev_transport = transport
                    hop_oplock = lk

            self.chain = built
            self.built_count += 1
            self.last_error = None
            self.reconnect_attempts = 0
            self.backoff_until = 0.0
            self.state.set()

            base_tr = built[-1].transport
            if self._gapfill is not None:
                self._gapfill.stop()
            self._gapfill = _GapFiller(base_tr, tun.keepalive, self._mark_gap_dead)
            self._gapfill.start()

            logger.info("session [%s] chain rebuilt: %s (%.1fs)",
                        self.label,
                        " → ".join(n.cfg["_label"] for n in built),
                        time.time() - started)
            d = self.status_dict()
            d["rebuilt_in_s"] = round(time.time() - started, 2)
            return d
        except Exception:
            for node in reversed(built):
                try:
                    if node.client is not None:
                        node.client.close()
                except Exception:  # noqa: BLE001
                    pass
            self.chain = []
            self.state.clear()
            raise

    def _mark_gap_dead(self, reason: str) -> None:
        logger.warning("session [%s] keepalive dead: %s", self.label, reason)
        with self.lock:
            self.last_error = f"keepalive: {reason}"
            self.state.clear()

    # ------------------------------------------------------------ 监督 --

    def _supervise_loop(self) -> None:
        while True:
            try:
                action_needed = False
                forced = self._reconnect_now.is_set()
                with self.lock:
                    if self._closed:
                        return
                    alive = self.is_alive_unlocked()
                    in_backoff = time.time() < self.backoff_until
                if ((forced or (self.chain and not alive)) and not in_backoff):
                    if forced:
                        self._reconnect_now.clear()
                    action_needed = True
                if action_needed:
                    self._attempt_rebuild()
                time.sleep(float(cfg("tuning.supervisor_poll_seconds", 5)))
            except Exception:  # noqa: BLE001   监督者永不退出
                time.sleep(5)

    def _attempt_rebuild(self) -> None:
        attempts = self.reconnect_attempts
        delay = min(self.RECONNECT_MAX_DELAY,
                    self.RECONNECT_BASE_DELAY * (2 ** min(attempts, 6)))
        self.reconnect_attempts = attempts + 1
        logger.info("session [%s] rebuilding (attempt %d, backoff %.0fs)",
                    self.label, attempts + 1, delay)
        time.sleep(delay)
        if self._closed:
            return
        tun = Tunings.snapshot()
        try:
            with self.lock:
                if self._closed:
                    return
                self._teardown_unlocked()
                AUTH_ISOLATOR.reset_if_env_changed()
                attach_auth(self.nodes_cfg)
                self._build_chain_locked(tun)
        except (SSHSetupRequired, SSHAuthCooldown, SSHAuthPermanentFailure) as exc:
            with self.lock:
                self.last_error = f"{type(exc).__name__}: {exc}"[:400]
                self.backoff_until = time.time() + 90.0
            logger.info("session [%s] paused rebuild: %.120s", self.label, exc)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.last_error = f"reconnect failed: {type(exc).__name__}: {exc}"[:400]
                self.backoff_until = time.time() + delay * 2
        finally:
            self._scrub_passwords()

    # ------------------------------------------------------------- exec --

    def exec_command(self, command: str, timeout: float,
                     tun: Tunings) -> Tuple[str, str, int, bool]:
        start = time.time()
        deadline = start + timeout if timeout > 0 else None

        # ---- 通道闸 ----
        with self._chan_cond:
            while self._active_cmd_channels >= tun.channel_limit and not self._closed:
                if not self._chan_cond.wait(timeout=min(10.0, timeout or 10.0)):
                    pass
                if deadline and time.time() > deadline:
                    raise TimeoutError(f"等待空闲通道超过 {timeout}s")
            if self._closed:
                raise RuntimeError("会话已关闭")
            self._active_cmd_channels += 1

        chan: Optional[paramiko.Channel] = None
        try:
            with self.lock:
                if self._closed:
                    raise RuntimeError("会话已关闭")
                if not self.is_alive_unlocked():
                    try:
                        self.ensure_connected(tunings=tun)
                    except (SSHSetupRequired, SSHAuthCooldown, SSHAuthPermanentFailure):
                        raise
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(f"链路不可用: {self.last_error or exc}") from exc
                node = self.chain[-1]
                transport = node.transport
                oplock = self._hop_locks.get(id(node.cfg))
                if oplock:
                    oplock.acquire()
                try:
                    chan = transport.open_session(timeout=self.CHANNEL_OPEN_TIMEOUT) \
                        if transport else None
                finally:
                    if oplock:
                        oplock.release()
            if chan is None:
                raise RuntimeError("无法打开执行通道")

            chan.settimeout(2.0)
            # ---- sudo 密码自动喂入 (tuning.sudo_autofill, 默认开) ----
            # 在 exec 前完成: 取密码 + 裸 sudo 注入 -S（stdin 读密码，免 tty）。
            # 凭据只在内存流转，不落日志/审计；单通道只喂一次。
            _sudo_pw = ""
            if getattr(tun, "sudo_autofill", True):
                try:
                    label_tok = re.sub(r"[^A-Z0-9]+", "_",
                                       self.label.upper()).strip("_")
                    env_map, _warns = get_env()
                    for cand in (
                            f"HERMES_SSH_SUDO_PASSWORD_{label_tok}",
                            "HERMES_SSH_SUDO_PASSWORD"):
                        val = env_map.get(cand)
                        if val:
                            _sudo_pw = val
                            break
                    if not _sudo_pw:  # 回落: SSH 登录同把密码 — 用【末跳】(真正执行命令的主机)
                        for node_cfg in reversed(self.nodes_cfg):
                            for ev in node_cfg.get("env_candidates", []) or []:
                                val = env_map.get(ev)
                                if val:
                                    _sudo_pw = val
                                    break
                            if _sudo_pw:
                                break
                except Exception:  # noqa: BLE001 — 凭据读取故障不阻断普通命令
                    _sudo_pw = ""
                if _sudo_pw and not command_has_sudo_stdin_flag(command):
                    command = inject_sudo_dash_S(command)
            _sudo_fed = False

            chan.exec_command(command)

            stdout_buf = bytearray()
            stderr_buf = bytearray()
            truncated = False
            exit_code: Optional[int] = None

            while True:
                if deadline and time.time() > deadline:
                    raise TimeoutError(
                        f"远程命令超过 {timeout}s 未完成（会话保持，不影响链路）")
                r, _, _ = _select.select([chan], [], [], 0.5)
                if r:
                    if chan.recv_ready():
                        data = chan.recv(65536)
                        if data:
                            room = MAX_OUTPUT_CHARS - len(stdout_buf)
                            if room > 0:
                                stdout_buf += data[:room]
                            if len(data) > room:
                                truncated = True
                    if chan.recv_stderr_ready():
                        data = chan.recv_stderr(65536)
                        if data:
                            room = MAX_OUTPUT_CHARS - len(stderr_buf)
                            if room > 0:
                                stderr_buf += data[:room]
                            if len(data) > room:
                                truncated = True
                    # ---- sudo 提示出现 → 喂密码 (一次) ----
                    if (_sudo_pw and not _sudo_fed
                            and b"password for" in bytes(stderr_buf).lower()
                            + bytes(stdout_buf).lower()):
                        try:
                            chan.sendall(_sudo_pw.encode() + b"\n")
                        except Exception:  # noqa: BLE001
                            pass
                        _sudo_fed = True
                    if (chan.exit_status_ready()
                            and not chan.recv_ready()
                            and not chan.recv_stderr_ready()):
                        exit_code = chan.recv_exit_status()
                        break
                elif chan.closed or (chan.exit_status_ready()):
                    exit_code = chan.recv_exit_status() if chan.exit_status_ready() else -1
                    break

            self.touch()
            return (stdout_buf.decode("utf-8", errors="replace"),
                    stderr_buf.decode("utf-8", errors="replace"),
                    exit_code if exit_code is not None else -1,
                    truncated)
        except (TimeoutError, RuntimeError):
            if chan is not None and not chan.closed:
                try:
                    chan.close()
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            with self._chan_cond:
                self._active_cmd_channels = max(0, self._active_cmd_channels - 1)
                self._chan_cond.notify()


def desc_of(cfg_n: Dict[str, Any]) -> str:
    return f"{cfg_n['username']}@{cfg_n['hostname']}:{cfg_n['port']}"


# ===========================================================================
# 会话池 ＋ Reaper
# ===========================================================================


class SessionPool:
    IDLE_TTL_DEFAULT = 1800.0
    MAX_SESSIONS_DEFAULT = 16

    def __init__(self) -> None:
        self.sessions: Dict[str, SSHSession] = {}
        self.mutex = threading.Lock()
        self._stop = threading.Event()
        self._reaper = threading.Thread(target=self._reap_loop,
                                        name="ssh-session-reaper", daemon=True)
        self._reaper.start()
        atexit.register(self.shutdown)

    # -- 入口 --

    def get(self, target_expr: str) -> SSHSession:
        tun = Tunings.snapshot()
        entries = parse_ssh_config()
        plan, canon = build_plan(target_expr, entries)

        with self.mutex:
            sess = self.sessions.get(canon)
            if sess is not None and not sess._closed:
                sess.touch()
                return sess

            # 总量闸：满了就驱逐最久未用的未钉会话
            live = {k: v for k, v in self.sessions.items() if not v._closed}
            if len(live) >= tun.max_sessions:
                evictable = [(v.last_used, k) for k, v in live.items() if not v.pinned]
                if evictable:
                    evictable.sort()
                    victim_k = evictable[0][1]
                    logger.info("pool full (%d): evicting idlest unpinned %s",
                                len(live), victim_k)
                    victim = self.sessions.pop(victim_k)
                    victim.close()
                else:
                    raise SSHParseError(
                        f"会话数已达上限 {tun.max_sessions} 且全部被 pin；"
                        f"请先用 ssh_disconnect 释放不需要的会话")

            sess = SSHSession(canon, plan, pool=self)
            self.sessions[canon] = sess
            return sess

    def lookup_canon(self, target_expr: str) -> Optional[str]:
        try:
            _, canon = build_plan(target_expr, parse_ssh_config())
            return canon
        except Exception:  # noqa: BLE001
            return None

    def drop(self, canon: str) -> bool:
        with self.mutex:
            s = self.sessions.pop(canon, None)
        if s is None:
            return False
        s.close()
        return True

    def drop_all(self) -> int:
        with self.mutex:
            items = list(self.sessions.keys())
        n = 0
        for k in items:
            if self.drop(k):
                n += 1
        return n

    def snapshot(self) -> List[Dict[str, Any]]:
        with self.mutex:
            sessions = [s for s in self.sessions.values()]
        tun = Tunings.snapshot()
        out = [s.status_dict() for s in sessions if not s._closed]
        for item in out:
            item["ttl_remaining_s"] = round(
                max(0, tun.idle_ttl - item.get("idle_s", 0)), 1)
        return out

    # -- Reaper --

    def _reap_loop(self) -> None:
        while not self._stop.wait(60.0):
            try:
                self._reap_once()
            except Exception:  # noqa: BLE001
                logger.exception("reaper error")

    def _reap_once(self) -> None:
        tun = Tunings.snapshot()
        now = time.time()
        victims: List[str] = []
        with self.mutex:
            for key, s in list(self.sessions.items()):
                if s._closed:
                    self.sessions.pop(key, None)
                    continue
                if not s.pinned and (now - s.last_used) > tun.idle_ttl:
                    victims.append(key)
        for key in victims:
            logger.info("reaper: dropping idle session %s (>%.0fs)",
                        key, tun.idle_ttl)
            self.drop(key)

    def shutdown(self) -> None:
        self._stop.set()
        with self.mutex:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for s in sessions:
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass


POOL = SessionPool()
