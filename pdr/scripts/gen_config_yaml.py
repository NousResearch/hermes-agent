"""把 pdr-agent 的 DEFAULT_CONFIG 连同中文元数据写入 pdr/config.yaml 的 hermes 段。

用法（仓库根目录）：python pdr/scripts/gen_config_yaml.py（pdr-agent 源码位于 pdr-agent/）
- 只替换 config.yaml 中 `# >>> hermes:begin` 与 `# <<< hermes:end` 之间的内容，PDR 段保持手写。
- 已有的 hermes 取值会保留（用户在页面或文件里改过的值不被默认值覆盖）。
- 每个配置项上方写一行元数据注释，格式与 PDR 段相同，配置页面直接解析这些注释：
      #: 中文名｜说明｜范围:<取值范围>｜生效:<热加载|下一会话|重启>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PDR = ROOT / "pdr"
HERMES = ROOT / "pdr-agent"
CONFIG = PDR / "config.yaml"
ZH = PDR / "server" / "pdr_server" / "agent" / "config_zh.json"
BEGIN, END = "# >>> hermes:begin", "# <<< hermes:end"

GROUP_ZH = {
    "general": "通用", "providers": "模型提供商", "fallback": "备用模型", "credential_pool_strategies": "凭据池策略",
    "database": "会话数据库", "runtime": "运行时", "session": "会话", "attachments": "附件", "agent": "智能体核心",
    "terminal": "终端执行", "web": "网页工具", "browser": "浏览器", "worktrees": "工作树", "checkpoints": "检查点",
    "mcp": "MCP 服务", "tool_output": "工具输出", "tool_loop_guardrails": "工具循环防护", "compression": "上下文压缩",
    "prompt_caching": "提示缓存", "openrouter": "OpenRouter", "bedrock": "AWS Bedrock", "auxiliary": "辅助模型",
    "display": "显示", "webapp": "Web 应用", "dashboard": "仪表盘", "privacy": "隐私", "tts": "语音合成",
    "stt": "语音识别", "voice": "语音对话", "vision": "视觉", "wake_word": "唤醒词", "human_delay": "拟人延迟",
    "context": "上下文", "memory": "记忆", "delegation": "子任务委派", "goals": "目标", "loops": "循环任务",
    "moa": "多模型协作", "skills": "技能", "curator": "技能整理", "slack": "Slack", "discord": "Discord",
    "whatsapp": "WhatsApp", "telegram": "Telegram", "mattermost": "Mattermost", "matrix": "Matrix",
    "approvals": "执行审批", "quick_commands": "快捷命令", "platform_hints": "平台提示", "plugins": "插件",
    "hooks": "钩子", "personalities": "人格", "auth": "认证", "security": "安全", "cron": "定时任务",
    "kanban": "看板", "bot_mode": "Bot 模式", "code_execution": "代码执行", "tools": "工具", "logging": "日志",
    "model_catalog": "模型目录", "model_overrides": "模型覆盖", "models_dev": "models.dev 元数据",
    "network": "网络", "monitoring": "监控", "hosted_rooms": "托管房间", "gateway": "消息网关",
    "streaming": "流式输出", "sessions": "会话管理", "onboarding": "引导", "telemetry": "遥测",
    "doctor": "诊断", "updates": "更新", "lsp": "语言服务", "x_search": "X 搜索", "vault": "保险库",
    "secrets": "密钥管理", "bot_desktop": "桌面 Bot", "computer_use": "电脑操作", "proxy": "代理",
    "desktop": "桌面端", "nous": "Nous Portal", "vertex": "Google Vertex", "local_runtime": "本地模型运行时",
}
# 进程级设置：改了需要重启 hermes 网关；其余默认「下一会话」生效（不破坏进行中会话的提示缓存）
RESTART_GROUPS = {"gateway", "dashboard", "webapp", "logging", "database", "runtime", "network", "proxy",
                  "monitoring", "telemetry", "slack", "discord", "telegram", "whatsapp", "mattermost", "matrix"}
ENUM_RE = re.compile(r"(?:^|[：:，,；;、\s(（])([A-Za-z0-9_.]+)-(?=[^\d\s])")
SKIP_KEYS = {"_config_version"}


def _flat(d: dict, prefix: str = ""):
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            yield from _flat(value, path + ".")
        else:
            yield path, value


def value_range(value, desc: str) -> str:
    if isinstance(value, bool):
        return "true/false"
    enums = [e for e in dict.fromkeys(ENUM_RE.findall(desc or "")) if not e.isdigit()]
    if len(enums) >= 2:
        return "|".join(enums)
    if value is None:
        return "可空"
    if isinstance(value, int):
        return "整数"
    if isinstance(value, float):
        return "小数"
    if isinstance(value, list):
        return "列表"
    if isinstance(value, dict):
        return "键值对"
    return "字符串"


def meta_line(key: str, value, zh: dict, indent: str) -> str:
    item = zh.get(key, {})
    label = item.get("label") or key.rsplit(".", 1)[-1]
    desc = (item.get("desc") or "（待补充中文说明）").replace("\n", " ").replace("｜", "/")
    group = key.split(".", 1)[0] if "." in key else "general"
    reload_ = "重启" if group in RESTART_GROUPS else "下一会话"
    return f"{indent}#: {label}｜{desc}｜范围:{value_range(value, desc)}｜生效:{reload_}"


def _scalar(value) -> str:
    return yaml.safe_dump(value, allow_unicode=True, default_flow_style=True, width=10**6).strip().removesuffix("\n...").removesuffix("...").strip()


def emit(d: dict, zh: dict, prefix: str, depth: int, out: list[str]) -> None:
    indent = "  " * depth
    for key, value in d.items():
        if key in SKIP_KEYS:
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            if depth == 1:
                out.append("")
                out.append(f"{indent}# ---- {GROUP_ZH.get(key, key)} ----")
            out.append(f"{indent}{_key(key)}:")
            emit(value, zh, path + ".", depth + 1, out)
        else:
            out.append(meta_line(path, value, zh, indent))
            out.append(f"{indent}{_key(key)}: {_scalar(value)}")


def _key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*", key) else json.dumps(key, ensure_ascii=False)


def _deep_merge(base: dict, over: dict) -> dict:
    merged = dict(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and merged[key]:
            merged[key] = _deep_merge(merged[key], value)
        elif key in merged:
            merged[key] = value
    return merged


def main() -> int:
    sys.path.insert(0, str(HERMES))
    from hermes_cli.config_defaults import DEFAULT_CONFIG  # noqa: PLC0415 - 需要先把 pdr-agent/ 加入 sys.path

    zh = json.loads(ZH.read_text(encoding="utf-8"))
    text = CONFIG.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"{CONFIG} 缺少 {BEGIN} / {END} 标记", file=sys.stderr)
        return 1
    current = (yaml.safe_load(text) or {}).get("hermes") or {}
    values = _deep_merge(DEFAULT_CONFIG, current)
    general = {k: v for k, v in values.items() if not (isinstance(v, dict) and v)}
    sections = {k: v for k, v in values.items() if isinstance(v, dict) and v}

    out = [BEGIN, "# 以下由 pdr/scripts/gen_config_yaml.py 依据 pdr-agent 的 DEFAULT_CONFIG 生成；可直接修改取值，",
           "# 重新生成时保留已修改的值。租户级覆盖存于 src.src_opr_config_value_f，不写入本文件。", "hermes:",
           "", "  # ---- 通用 ----"]
    emit(general, zh, "", 1, out)
    emit(sections, zh, "", 1, out)
    out.append(END)
    head, rest = text.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    CONFIG.write_text(head + "\n".join(out) + tail, encoding="utf-8")

    leaves = [k for k, _ in _flat(DEFAULT_CONFIG) if k not in SKIP_KEYS]
    missing = [k for k in leaves if k not in zh]
    print(f"hermes 配置项 {len(leaves)} 个，已中文化 {len(leaves) - len(missing)} 个")
    if missing:
        print("缺少中文元数据：" + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
