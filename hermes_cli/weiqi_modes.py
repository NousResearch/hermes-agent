"""Wei Qi / 7v profile manual mode presets.

Shared data for CLI/gateway surfaces. These presets are intentionally
session-scoped when applied by the gateway; they must not mutate profile config.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class WeiqiModePreset:
    key: str
    label: str
    provider: str
    model: str
    reasoning: str
    description: str
    aliases: tuple[str, ...] = ()
    note: str = ""
    base_url: str | None = None
    acp_command: str | None = None
    acp_args: tuple[str, ...] | None = None


OPUS_STATUS = "OPUS_ENABLED_CLAUDE_CODE_MAX_CHAT_ONLY"
OPUS_ACP_COMMAND = "/Users/neo/.hermes/bin/claude-code-print-acp"

_PRESETS: dict[str, WeiqiModePreset] = {
    "default": WeiqiModePreset(
        key="default",
        label="默认助手",
        provider="openai-codex",
        model="gpt-5.5",
        reasoning="medium",
        description="日常问题、一般助手、简洁但有判断的中文回复。",
        aliases=("默认", "default"),
    ),
    "research": WeiqiModePreset(
        key="research",
        label="论文/研究",
        provider="openai-codex",
        model="gpt-5.5",
        reasoning="high",
        description="论文、学术、行业研究；区分事实/假设/观点/待验证，保留来源意识。",
        aliases=("论文", "研究", "research", "thesis"),
    ),
    "business": WeiqiModePreset(
        key="business",
        label="MBA/商业分析",
        provider="openai-codex",
        model="gpt-5.5",
        reasoning="high",
        description="MBA、案例、商业模式、战略、AI robotics 商业化；偏结构化分析和严谨推理。",
        aliases=("mba", "business"),
    ),
    "advisor": WeiqiModePreset(
        key="advisor",
        label="顾问/第二意见",
        provider="copilot-acp",
        model="opus",
        reasoning="high",
        description="第二意见、盲点挑战、战略顾问、教授/投资人视角；适合需要品味和判断的深度反馈。",
        aliases=("顾问", "advisor"),
        note=f"Opus 已通过 Claude Code Max 订阅路线启用；此模式仅作聊天大脑，不让 Opus 自己调用工具：{OPUS_STATUS}。",
        base_url="acp://copilot",
        acp_command=OPUS_ACP_COMMAND,
        acp_args=(),
    ),
    "creative": WeiqiModePreset(
        key="creative",
        label="播客/创意/品牌",
        provider="copilot-acp",
        model="opus",
        reasoning="high",
        description="播客、内容方向、品牌创意、sevenchic、slogan/命名/叙事；优先使用 Opus 创意路线。",
        aliases=("播客", "创意", "creative", "podcast"),
        note=f"Opus 已通过 Claude Code Max 订阅路线启用；此模式仅作聊天大脑，不让 Opus 自己调用工具：{OPUS_STATUS}。",
        base_url="acp://copilot",
        acp_command=OPUS_ACP_COMMAND,
        acp_args=(),
    ),
    "notes": WeiqiModePreset(
        key="notes",
        label="整理/知识沉淀",
        provider="openai-codex",
        model="gpt-5.5",
        reasoning="high",
        description="整理、总结、归纳、会议纪要、开放问题、记忆候选和项目标签。",
        aliases=("整理", "notes", "organize", "summary"),
    ),
    "polish": WeiqiModePreset(
        key="polish",
        label="中文润色",
        provider="openrouter",
        model="moonshotai/kimi-k2.6",
        reasoning="medium",
        description="中文表达润色、小红书/口播/标题/更自然高级的中文版本；优先使用 Kimi 中文路线。",
        aliases=("中文润色", "润色", "polish", "chinese"),
    ),
    "cheap": WeiqiModePreset(
        key="cheap",
        label="省钱/低风险草稿",
        provider="openrouter",
        model="qwen/qwen3.6-plus",
        reasoning="low",
        description="低风险草稿、中文初稿、便宜路线；高风险任务应切回高级路线。",
        aliases=("省钱", "cheap", "local"),
        note="本机 litellm-qwen/ollama smoke 未通过，临时使用已通过的 OpenRouter Qwen；不用于默认流量。",
    ),
}

_ALIAS_TO_KEY: dict[str, str] = {}
for _key, _preset in _PRESETS.items():
    _ALIAS_TO_KEY[_key.lower()] = _key
    for _alias in _preset.aliases:
        _ALIAS_TO_KEY[_alias.lower()] = _key

STATUS_ALIASES = {"状态", "status", "mode", "模式"}
AUTO_ALIASES = {"自动", "智能", "auto", "smart"}


def all_weiqi_command_aliases() -> tuple[str, ...]:
    """Aliases exposed as slash commands for Wei Qi manual modes."""
    aliases: list[str] = []
    for preset in _PRESETS.values():
        aliases.extend(preset.aliases)
    aliases.extend(STATUS_ALIASES)
    aliases.extend(AUTO_ALIASES)
    # resolve_command lowercases command names, so include lowercase ASCII aliases.
    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        normalized = alias.lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized if alias.isascii() else alias)
    return tuple(result)


def resolve_weiqi_mode(command: str = "", args: str = "") -> WeiqiModePreset | None:
    """Resolve a mode from the raw slash command and/or /mode argument."""
    candidates: Iterable[str] = (args.strip().split()[0] if args.strip() else "", command.strip().lstrip("/"))
    for candidate in candidates:
        key = _ALIAS_TO_KEY.get(candidate.lower())
        if key:
            return _PRESETS[key]
    return None


def is_status_request(command: str = "", args: str = "") -> bool:
    token = (args.strip().split()[0] if args.strip() else command.strip().lstrip("/"))
    return token.lower() in {a.lower() for a in STATUS_ALIASES}


def is_auto_request(command: str = "", args: str = "") -> bool:
    token = (args.strip().split()[0] if args.strip() else command.strip().lstrip("/"))
    return token.lower() in {a.lower() for a in AUTO_ALIASES}


def get_weiqi_mode(key: str) -> WeiqiModePreset | None:
    return _PRESETS.get(key)


def list_weiqi_modes() -> tuple[WeiqiModePreset, ...]:
    return tuple(_PRESETS.values())


_AUTO_ROUTE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cheap",
        (
            r"省钱", r"便宜", r"低成本", r"快速草稿", r"随便起草", r"先给我个草稿",
            r"cheap", r"low[- ]?cost",
        ),
    ),
    (
        "polish",
        (
            r"润色", r"改写", r"优化表达", r"更自然", r"更高级", r"更像中文",
            r"标题", r"小红书", r"朋友圈", r"口播", r"文案", r"措辞", r"语气",
            r"polish", r"rewrite", r"copywriting",
        ),
    ),
    (
        "creative",
        (
            r"播客", r"脚本", r"选题", r"创意", r"品牌", r"slogan", r"命名",
            r"叙事", r"故事线", r"人设", r"内容方向", r"sevenchic", r"视觉风格",
            r"podcast", r"creative", r"brand", r"tagline",
        ),
    ),
    (
        "advisor",
        (
            r"顾问", r"第二意见", r"盲点", r"挑战我", r"你怎么看", r"帮我判断",
            r"战略建议", r"投资人视角", r"教授视角", r"高层视角", r"长期规划",
            r"advisor", r"second opinion", r"pushback", r"blind spot",
        ),
    ),
    (
        "business",
        (
            r"\bmba\b", r"商业模式", r"商业化", r"战略", r"市场", r"竞品",
            r"融资", r"投资", r"增长", r"定位", r"ai robotics", r"机器人", r"案例分析",
            r"business model", r"go[- ]?to[- ]?market", r"market sizing",
        ),
    ),
    (
        "research",
        (
            r"论文", r"文献", r"研究", r"课题", r"开题", r"摘要", r"引用",
            r"方法论", r"数据分析", r"假设", r"变量", r"访谈", r"问卷", r"学术",
            r"thesis", r"literature", r"citation", r"methodology", r"hypothesis",
        ),
    ),
    (
        "notes",
        (
            r"整理", r"总结", r"归纳", r"复盘", r"会议纪要", r"提纲", r"要点",
            r"行动项", r"知识沉淀", r"summary", r"organize", r"meeting notes",
        ),
    ),
)


def classify_weiqi_auto_mode(text: str, *, has_media: bool = False) -> WeiqiModePreset:
    """Deterministically choose a Wei Qi mode for a normal WeChat turn.

    Conservative default: route uncertain work to GPT-5.5 default rather than
    paid/specialized lanes. Manual slash modes still override this classifier.
    """
    haystack = (text or "").strip().lower()
    if has_media and not haystack:
        return _PRESETS["default"]
    for key, patterns in _AUTO_ROUTE_RULES:
        for pattern in patterns:
            if re.search(pattern, haystack, flags=re.IGNORECASE):
                return _PRESETS[key]
    return _PRESETS["default"]
