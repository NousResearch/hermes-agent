"""hermes-ssh — Hermes 原生 SSH 客户端插件。

注册 4 个工具（toolset: ssh）：
- ssh_exec       远程执行命令（核心入口；自动建链/免密检查/审计）
- ssh_status     会话池状态（跳板链/保活心跳/重连退避/认证隔离区）
- ssh_connect    显式预热或强制重建（keep_alive=true 钉住常驻，54 用法）
- ssh_disconnect 关闭会话（支持 "*" 全撤）

调优与审计参数：plugins.entries.hermes-ssh.settings.{tuning,audit}.*
缺失时回落 plugin.yaml config_schema 的 default。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List

from tools.registry import tool_error, tool_result

from . import audit as _audit
from . import core as CORE  # noqa: F401  (echo_to_chat 经由 CORE 引用)
from .core import (
    ENV_PATH,
    AUTH_ISOLATOR,
    POOL,
    SSHParseError,
    SSHAuthCooldown,
    SSHAuthPermanentFailure,
    SSHSetupRequired,
    Tunings,
    echo_to_chat,
    install_cfg_reader,
)

logger = logging.getLogger(__name__)


def _cfg_factory(ctx):
    """把 PluginContext.get_config 包装成 core 可用的 reader。"""
    def reader(key: str, default=None):
        try:
            return ctx.get_config(key, default)
        except Exception:  # noqa: BLE001
            return default
    return reader


# ---------------------------------------------------------------- schemas --

_STRING = {"type": "string"}

SSH_EXEC_SCHEMA = {
    "name": "ssh_exec",
    "description": (
        "在远程 SSH 主机执行 shell 命令并返回 {exit_code, stdout, stderr}。"
        "target 支持 ~/.ssh/config 别名（byd-54、gz，自动 ProxyJump 跳板）或 "
        "user@host[:port] 直连。连接跨调用复用、断线自动重建；首次使用若缺密码"
        "配置将返回需写入 " + ENV_PATH + " 的确切变量名。每次调用计入审计日志。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {**_STRING, "description": "Host 别名或 user@host[:port]"},
            "command": {**_STRING, "description": "shell 命令"},
            "timeout": {"type": "number", "description": "秒，默认60；0=不限。超时不中断会话"},
        },
        "required": ["target", "command"],
    },
}

SSH_STATUS_SCHEMA = {
    "name": "ssh_status",
    "description": (
        "hermes-ssh 运行状态：所有会话的跳板链、认证方式、保活心跳、重连退避、"
        "pin/TTL 回收倒计时，以及认证失败隔离区当前状况。"
    ),
    "parameters": {"type": "object", "properties": {}},
}

SSH_CONNECT_SCHEMA = {
    "name": "ssh_connect",
    "description": (
        "建立（或重建）到目标主机的连接并将其标记为【常驻】：此后 supervisor "
        "持续保活+自动重连，不被空闲回收。适合 '用户点名的主机必须实时在线' "
        "的场景（如 byd-54）。force=true 强制拆除现有链路完全重建。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {**_STRING, "description": "Host 别名或 user@host[:port]"},
            "keep_alive": {
                "type": "boolean",
                "description": "true=常驻（默认）；false=仅本次临时连接，交给TTL回收",
            },
            "force": {"type": "boolean", "description": "丢弃现有连接彻底重建（默认false）"},
        },
        "required": ["target"],
    },
}

SSH_DISCONNECT_SCHEMA = {
    "name": "ssh_disconnect",
    "description": "关闭会话并从池中移除。target=\"*\" 关闭全部。",
    "parameters": {
        "type": "object",
        "properties": {"target": {**_STRING, "description": "Host 别名/表达式，或 *"}},
        "required": ["target"],
    },
}


# ---------------------------------------------------------------- helpers --


def _setup_payload(exc: SSHSetupRequired) -> str:
    hops = []
    howto = []
    for h in exc.hops:
        note = f"（{h['note']}）" if isinstance(h, dict) and h.get("note") else ""
        env_var = h["env_var"] if isinstance(h, dict) else str(h)
        host = h.get("host", "?") if isinstance(h, dict) else "?"
        user = h.get("user", "?") if isinstance(h, dict) else "?"
        hops.append({"env_var": env_var, "host": f"{user}@{host}", "note": note.strip("（）")})
        howto.append(f"echo 'export {env_var}=\"该主机({user}@{host})的SSH密码\"' >> {ENV_PATH}")
    return json.dumps({
        "status": "setup_required",
        "message": "缺少免密凭据配置",
        "todo": hops,
        "howto": howto,
        "env_file": ENV_PATH,
    }, ensure_ascii=False, indent=2)


def _cooldown_payload(exc: Exception) -> str:
    return json.dumps({
        "status": "auth_cooldown" if isinstance(exc, SSHAuthCooldown) else "auth_permanent_failure",
        "message": str(exc),
        "isolator": AUTH_ISOLATOR.status(),
    }, ensure_ascii=False, indent=2)


def _now_ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def _warmup_worker(targets: List[str]) -> None:
    """启动预热线程：把 settings.tuning.warmup 清单里的主机逐个建链并 pin。

    网关每次重启后执行一次到成功为止；缺凭据(SetupRequired)/冷却时安静跳过、
    60s 后重试（用户补完 .env 即自动恢复）。绝不阻塞主启动流程。
    """
    logger.info("ssh warmup: %d target(s): %s", len(targets), targets)
    pending = list(targets)
    for _round in range(30):                 # 最多 ~30 分钟内反复尝试
        still_pending: List[str] = []
        for tgt in pending:
            try:
                sess = POOL.get(tgt)
                sess.pinned = True           # 常驻语义：清单成员永不回收
                info = sess.ensure_connected()
                logger.info("ssh warmup [%s]: %s (%s)",
                            sess.label,
                            "connected" if info.get("ok") else "?",
                            ",".join(c["label"] for c in info.get("chain", [])))
            except (SSHSetupRequired, SSHAuthCooldown, SSHAuthPermanentFailure) as exc:
                logger.info("ssh warmup [%s]: waiting for credentials (%.80s)", tgt, exc)
                still_pending.append(tgt)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ssh warmup [%s]: %s: %s", tgt, type(exc).__name__, exc)
                still_pending.append(tgt)
        pending = still_pending
        if not pending:
            return
        time.sleep(60)
    logger.warning("ssh warmup: giving up on %s after max rounds", pending)


def register(ctx) -> None:
    """loader 入口：装配置读取器 → 配审计 → 注册工具 → 可选预热。"""
    install_cfg_reader(_cfg_factory(ctx))

    tun = Tunings.snapshot()
    path = _audit.configure(max_mb=float(
        (ctx.get_config("audit.max_file_mb", 20) or 20)), backups=3)
    logger.info("hermes-ssh ready (audit=%s at %s)",
                tun.audit_enabled, path or "(unavailable)")

    warmup_raw = ctx.get_config("tuning.warmup", None)
    if warmup_raw:
        warmup = ([str(warmup_raw)] if isinstance(warmup_raw, str)
                  else [str(x) for x in warmup_raw])
        warmup = [w for w in warmup if w.strip()]
        if warmup:
            threading.Thread(target=_warmup_worker, args=(warmup,),
                             name="hermes-ssh-warmup", daemon=True).start()

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="ssh",
            schema=schema,
            handler=handler,
            check_fn=None,
            is_async=False,
            description=schema["description"],
            emoji=emoji,
        )


# ---------------------------------------------------------------- handlers --


def _handle_ssh_exec(args: Dict[str, Any], **kw) -> str:
    t0 = time.time()
    tun = Tunings.snapshot()
    target = str(args.get("target") or "").strip()
    command = str(args.get("command") or "").strip()
    try:
        timeout = float(args.get("timeout") or 60)
    except (TypeError, ValueError):
        timeout = 60.0
    caller_session = str(kw.get("session_id") or "")
    task_id = str(kw.get("task_id") or "")

    def _extra(**kw2: Any) -> Dict[str, Any]:
        # 审计溯源统一透传 task_id; 其余字段由调用点给出
        base: Dict[str, Any] = dict(kw2)
        if task_id:
            base["task_id"] = task_id
        return base

    if not target or not command:
        return tool_error("target 和 command 都不能为空")

    # ---- 安全守卫: 复用 Hermes 官方审批管线 (与 terminal 工具同一套) ----
    # hardline/sudo/deny规则/危险模式/tirith 内容扫描 + [o]nce/[s]ession/
    # [a]lways/[d]eny 人工审批, gateway 场景下自动变成飞书审批卡片。
    # 被拒时返回 BLOCKED 消息并审计 exec_rejected。
    try:
        from tools.approval import check_all_command_guards  # noqa: PLC0415
        verdict = check_all_command_guards(command, "local")
        if not verdict.get("approved"):
            _audit.emit("exec_rejected", target=target,
                        caller_session=caller_session, command=command,
                        status="blocked",
                        extra=_extra(note="approval guard denied"))
            echo_to_chat(command, "BLOCKED", "", str(verdict.get('message') or ""),
                         target, 0, status_note="approval guard 拦截")
            return tool_error(
                f"BLOCKED by approval guard: {verdict.get('message') or '未获批准'}\n"
                "(远程命令同样经过 Hermes 危险命令检测; 如误拦可在对话中 "
                "/approve 或调整 approvals 配置)")
    except ImportError:
        pass  # 守卫模块缺失时不阻塞基本功能 (理论不发生)
    # 守卫内部抛异常 → 放给外层 except 统一处理

    if not tun.audit_enabled:
        _audit._configured = False   # 动态关审计（极少用）

    try:
        session = POOL.get(target)
    except (SSHParseError, SSHSetupRequired) as exc:
        payload = (_setup_payload(exc) if isinstance(exc, SSHSetupRequired)
                   else None)
        _audit.emit("exec_rejected", target=target, caller_session=caller_session,
                    command=command,
                    status="error", error_type=type(exc).__name__,
                    extra=_extra(note="plan/setup failed"))
        if payload:
            return tool_result(payload)
        return tool_error(str(exc))
    except (SSHAuthCooldown, SSHAuthPermanentFailure) as exc:
        return tool_result(_cooldown_payload(exc))

    try:
        out, err, code, truncated = session.exec_command(command, timeout, tun)
        dur = _now_ms(t0)
        _audit.emit("exec", target=session.label, caller_session=caller_session,
                    command=command, exit_code=code, duration_ms=dur,
                    extra=_extra(stdout_len=len(out), stderr_len=len(err)))
        # 方案B: 命令+输出直推当前对话 (独立消息, 不经 AI 总结; 尽力而为)
        echo_to_chat(command, code, out, err, session.label, dur,
                     status_note="输出截断" if truncated else "")
        payload = {"session": session.label, "exit_code": code,
                   "stdout": out, "stderr": err}
        if truncated:
            payload["truncated"] = True
        return tool_result(json.dumps(payload, ensure_ascii=False))
    except (SSHSetupRequired,) as exc:
        return tool_result(_setup_payload(exc))
    except (SSHAuthCooldown, SSHAuthPermanentFailure) as exc:
        return tool_result(_cooldown_payload(exc))
    except TimeoutError as exc:
        _audit.emit("exec", target=session.label, caller_session=caller_session,
                    command=command, duration_ms=_now_ms(t0), status="timeout",
                    error_type="TimeoutError", extra=_extra())
        echo_to_chat(command, "TIMEOUT", "", str(exc), session.label,
                     _now_ms(t0), status_note=f"超过 {timeout}s 被终止")
        return tool_error(str(exc), status_code=408)
    except Exception as exc:  # noqa: BLE001
        _audit.emit("exec", target=str(getattr(session, "label", target)),
                    caller_session=caller_session, command=command,
                    duration_ms=_now_ms(t0), status="error",
                    error_type=type(exc).__name__, extra=_extra())
        echo_to_chat(command, "ERR", "", str(exc),
                     str(getattr(session, "label", target)), _now_ms(t0))
        return tool_error(
            f"ssh_exec 失败 ({type(exc).__name__}): {exc}\n"
            "提示：链路会在后台自愈；可 ssh_status 查看，或 ssh_connect force=true 手动重建。")


def _handle_ssh_status(args: Dict[str, Any], **kw) -> str:
    from .core import get_env
    tun = Tunings.snapshot()
    _, warnings = get_env()
    data = {
        "sessions": POOL.snapshot(),
        "auth_isolator": AUTH_ISOLATOR.status(),
        "audit_log": str(_audit._log_dir() / "audit.log"),
        "tunings": {
            "idle_ttl_s": tun.idle_ttl,
            "max_sessions": tun.max_sessions,
            "channel_limit": tun.channel_limit,
            "auth_cooldown_s": tun.auth_cooldown,
            "keepalive_s": tun.keepalive,
        },
        "env_file": ENV_PATH,
        "warnings": warnings,
        "config_hint": (
            "覆盖参数请写 profiles/<本profile>/config.yaml → "
            "plugins.entries.hermes-ssh.settings.tuning.* / audit.*"),
    }
    return tool_result(json.dumps(data, ensure_ascii=False, indent=2))


def _handle_ssh_connect(args: Dict[str, Any], **kw) -> str:
    t0 = time.time()
    caller_session = str(kw.get("session_id") or "")
    task_id = str(kw.get("task_id") or "")

    def _extra(**kw2: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = dict(kw2)
        if task_id:
            base["task_id"] = task_id
        return base

    target = str(args.get("target") or "").strip()
    keep_alive = args.get("keep_alive")
    # 语义: ssh_connect 的本意就是"用户点名要这条链常驻"
    # → 未显式传 keep_alive 时默认 True
    if keep_alive is None:
        pinned_wanted = True
    else:
        pinned_wanted = bool(keep_alive)
    force = bool(args.get("force"))
    if not target:
        return tool_error("target 不能为空")

    try:
        session = POOL.get(target)
    except SSHSetupRequired as exc:
        return tool_result(_setup_payload(exc))
    except (SSHAuthCooldown, SSHAuthPermanentFailure) as exc:
        return tool_result(_cooldown_payload(exc))
    except SSHParseError as exc:
        return tool_error(str(exc))

    if pinned_wanted:
        session.pinned = True
    try:
        info = session.ensure_connected(force=force)
        _audit.emit("connect", target=session.label, caller_session=caller_session,
                    duration_ms=_now_ms(t0),
                    extra=_extra(note=f"pinned={session.pinned} force={force}"))
        info["pinned"] = session.pinned
        return tool_result(json.dumps(info, ensure_ascii=False, indent=2))
    except SSHSetupRequired as exc:
        return tool_result(_setup_payload(exc))
    except (SSHAuthCooldown, SSHAuthPermanentFailure) as exc:
        return tool_result(_cooldown_payload(exc))
    except Exception as exc:  # noqa: BLE001
        st = session.status_dict()
        _audit.emit("connect", target=session.label, caller_session=caller_session,
                    duration_ms=_now_ms(t0), status="error",
                    error_type=type(exc).__name__, extra=_extra())
        return tool_error(
            f"连接失败: {type(exc).__name__}: {exc}\n最近错误: {st.get('last_error')}\n"
            "后台会继续按退避自动重连。")


def _handle_ssh_disconnect(args: Dict[str, Any], **kw) -> str:
    caller_session = str(kw.get("session_id") or "")
    task_id = str(kw.get("task_id") or "")

    def _extra(**kw2: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = dict(kw2)
        if task_id:
            base["task_id"] = task_id
        return base

    target = str(args.get("target") or "").strip()
    if not target:
        return tool_error("target 不能为空")

    if target == "*":
        n = POOL.drop_all()
        _audit.emit("disconnect_all", caller_session=caller_session,
                    extra=_extra(note=f"closed={n}"))
        return tool_result(f"已关闭 {n} 个会话")

    canon = POOL.lookup_canon(target)
    if canon is None:
        return tool_error(f"无法解析目标 {target!r}")
    removed = POOL.drop(canon)
    _audit.emit("disconnect", target=canon, caller_session=caller_session,
                status="ok" if removed else "noop", extra=_extra())
    msg = (f"会话 {canon} 已关闭并移除" if removed
           else f"没有找到活跃会话 {canon}")
    return tool_result(msg)


_TOOLS = (
    ("ssh_exec", SSH_EXEC_SCHEMA, _handle_ssh_exec, "🔌"),
    ("ssh_status", SSH_STATUS_SCHEMA, _handle_ssh_status, "📊"),
    ("ssh_connect", SSH_CONNECT_SCHEMA, _handle_ssh_connect, "🔗"),
    ("ssh_disconnect", SSH_DISCONNECT_SCHEMA, _handle_ssh_disconnect, "✂️"),
)
