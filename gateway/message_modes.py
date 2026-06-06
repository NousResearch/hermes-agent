"""Gateway message mode routing helpers.

A single messaging adapter can serve multiple prompt/tool profiles by routing
well-known text prefixes before session selection and agent construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_BASE_REQUIRED_SKILLS = ("using-superpowers",)
_DEV_REQUIRED_SKILLS = (*_BASE_REQUIRED_SKILLS, "project-dev-workflow")


@dataclass(frozen=True)
class GatewayMessageMode:
    """Resolved gateway mode for one inbound user message."""

    name: str
    message: str
    prefix: str = ""
    session_scope: Optional[str] = None
    enabled_toolsets: Optional[tuple[str, ...]] = None
    skip_context_files: bool = False
    load_soul_identity: bool = False
    skip_memory: bool = False
    system_prompt: str = ""
    required_skills: tuple[str, ...] = _BASE_REQUIRED_SKILLS
    sticky_mode: Optional[str] = None
    control_response: str = ""


_LITE_TOOLSETS = ("web", "vision", "image_gen", "session_search", "clarify", "skills")
_OPS_TOOLSETS = (
    "terminal",
    "file",
    "skills",
    "memory",
    "session_search",
    "todo",
    "clarify",
)
_LITE_BASE_SKILLS = (
    "yume-skill/using-superpowers",
    "yume-skill/verification-before-completion",
)
_LITE_INTENT_SKILL_RULES = (
    (
        (
            "hermes",
            "agent",
            "路由",
            "gateway",
            "技能",
            "profile",
            "cron",
            "memory",
        ),
        ("autonomous-ai-agents/hermes-agent",),
    ),
    (
        ("qq", "qqbot", "控制台", "私聊", "群聊", "/new", "/reset"),
        ("qq-control-console",),
    ),
    (
        ("debug", "修复", "报错", "为什么", "根因", "失败"),
        ("yume-skill/systematic-debugging",),
    ),
)
_LITE_DEV_DIRECT_KEYWORDS = (
    "建库",
    "改代码",
    "修改代码",
    "写代码",
    "跑测试",
    "部署",
    "commit",
    "diff",
    "branch",
    "push",
    "提交代码",
)
_LITE_DEV_ACTION_WORDS = (
    "建",
    "创建",
    "新建",
    "改",
    "修改",
    "修",
    "修复",
    "跑",
    "部署",
    "提交",
    "push",
    "commit",
)
_LITE_DEV_OBJECT_WORDS = (
    "github",
    "repo",
    "仓库",
    "代码",
    "测试",
    "路由",
    "gateway",
    "lite",
)

_LITE_PROMPT = """# Gateway mode: lightweight daily chat

This gateway turn was routed by the user's `日常聊天` prefix or sticky daily-chat route.
Operate as a lightweight Q&A/chat assistant:
- Prefer concise Chinese answers.
- Treat the task as read-only unless the user explicitly asks for creation or changes.
- For video/link summaries, use lightweight public metadata or provided text first; do not download media, install dependencies, repair browser runtimes, or do full audio transcription unless explicitly requested.
- Keep the route lightweight, but use preloaded core skills and lazy intent skills when the stripped user request asks for Hermes, QQ, or troubleshooting context.
- If the stripped user request is an execution-oriented development task (code/repo/GitHub/tests/deploy), escalate to the default development route instead of staying in lightweight chat.
- Do not assume the current repository/project context is relevant.
- For references like `刚才`, `上次`, `那个`, or `你推荐的那个`, infer from the nearest visible topic first.
- If visible context is insufficient, ask one concise clarifying question or use token-frugal history lookup: prefer current-scope handoff/previous context first; otherwise use narrow concrete keywords from the user's wording with `limit=1`.
- Avoid broad `OR` history searches, generic keywords, global fallback, or pulling large unrelated transcripts unless the user explicitly asks for full history search.
"""

_OPS_PROMPT = """# Gateway mode: Hermes operations

This gateway turn was routed by the user's `运维` prefix or sticky ops route.
Operate as Hermes runtime/config operations mode:
- Focus on Hermes Agent, gateway, WebUI, providers, tools, skills, logs, and runtime health.
- Keep diagnostics staged and evidence-based.
- Read config/log/state first; show exact diffs and get confirmation before broad config edits, restarts, deletes, restores, or secret-affecting changes.
- Do not assume the current development repository context is relevant unless the user explicitly asks for code changes.
"""


_PREFIXES = (
    (
        "日常聊天",
        GatewayMessageMode(
            name="lite",
            message="",
            prefix="日常聊天",
            session_scope="lite",
            enabled_toolsets=_LITE_TOOLSETS,
            skip_context_files=True,
            load_soul_identity=True,
            skip_memory=True,
            system_prompt=_LITE_PROMPT,
            required_skills=_LITE_BASE_SKILLS,
        ),
    ),
    (
        "运维",
        GatewayMessageMode(
            name="ops",
            message="",
            prefix="运维",
            session_scope="ops",
            enabled_toolsets=_OPS_TOOLSETS,
            skip_context_files=True,
            load_soul_identity=True,
            skip_memory=False,
            system_prompt=_OPS_PROMPT,
            required_skills=("using-superpowers",),
        ),
    ),
)

_STRIP_CHARS = " \t\r\n:：,，。.!！?？-—_、|/\\+＋"
_MODE_BY_NAME = {template.name: template for _prefix, template in _PREFIXES}
_DEV_PREFIXES = ("开发",)
_STICKY_MODE_SWITCHES = (
    (
        "lite",
        (
            "切到日常聊天路由",
            "切换到日常聊天路由",
            "进入日常聊天路由",
            "切到日常聊天模式",
            "切换到日常聊天模式",
            "进入日常聊天模式",
            "日常聊天模式",
        ),
    ),
    (
        "ops",
        (
            "切到运维路由",
            "切换到运维路由",
            "进入运维路由",
            "切到运维模式",
            "切换到运维模式",
            "进入运维模式",
            "运维模式",
        ),
    ),
    (
        "dev",
        (
            "切回开发路由",
            "切到开发路由",
            "切回默认路由",
            "切到默认路由",
            "切回开发模式",
            "切到开发模式",
            "切回默认模式",
            "切到默认模式",
            "开发模式",
            "默认模式",
        ),
    ),
)


def _strip_mode_prefix(text: str, prefix: str) -> str:
    stripped = text.lstrip()
    remainder = stripped[len(prefix):]
    return remainder.lstrip(_STRIP_CHARS)


def _copy_mode(
    template: GatewayMessageMode,
    *,
    message: str,
    sticky_mode: Optional[str] = None,
    control_response: str = "",
    required_skills: Optional[tuple[str, ...]] = None,
) -> GatewayMessageMode:
    return GatewayMessageMode(
        name=template.name,
        message=message,
        prefix=template.prefix,
        session_scope=template.session_scope,
        enabled_toolsets=template.enabled_toolsets,
        skip_context_files=template.skip_context_files,
        load_soul_identity=template.load_soul_identity,
        skip_memory=template.skip_memory,
        system_prompt=template.system_prompt,
        required_skills=required_skills if required_skills is not None else template.required_skills,
        sticky_mode=sticky_mode,
        control_response=control_response,
    )


def _dev_mode(
    message: str,
    *,
    sticky_mode: Optional[str] = None,
    control_response: str = "",
) -> GatewayMessageMode:
    return GatewayMessageMode(
        name="dev",
        message=message,
        required_skills=_DEV_REQUIRED_SKILLS,
        sticky_mode=sticky_mode,
        control_response=control_response,
    )


def _normalize_switch_text(text: str) -> str:
    return "".join(str(text or "").split()).strip(_STRIP_CHARS)


def _fold_text(text: str | None) -> str:
    return str(text or "").casefold()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(str(keyword).casefold() in text for keyword in keywords)


def _dedupe_skills(*groups: tuple[str, ...]) -> tuple[str, ...]:
    skills: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for skill in group:
            if skill and skill not in seen:
                seen.add(skill)
                skills.append(skill)
    return tuple(skills)


def _lite_required_skills(message: str | None) -> tuple[str, ...]:
    folded = _fold_text(message)
    matched: list[tuple[str, ...]] = []
    for keywords, skills in _LITE_INTENT_SKILL_RULES:
        if _contains_any(folded, keywords):
            matched.append(skills)
    return _dedupe_skills(_LITE_BASE_SKILLS, *matched)


def _lite_should_escalate_to_dev(message: str | None) -> bool:
    folded = _fold_text(message)
    if not folded.strip():
        return False
    if _contains_any(folded, _LITE_DEV_DIRECT_KEYWORDS):
        return True
    return _contains_any(folded, _LITE_DEV_ACTION_WORDS) and _contains_any(
        folded,
        _LITE_DEV_OBJECT_WORDS,
    )


def _copy_lite_mode(
    template: GatewayMessageMode,
    *,
    message: str,
    sticky_mode: Optional[str] = None,
    control_response: str = "",
) -> GatewayMessageMode:
    if _lite_should_escalate_to_dev(message):
        return _dev_mode(
            message=message,
            sticky_mode="dev",
        )
    return _copy_mode(
        template,
        message=message,
        sticky_mode=sticky_mode,
        control_response=control_response,
        required_skills=_lite_required_skills(message),
    )


def resolve_gateway_message_mode(text: str | None, active_mode: str | None = None) -> GatewayMessageMode:
    """Resolve a lightweight/ops/dev mode from a user-facing text prefix.

    Prefixes are intentionally plain Chinese phrases for QQ remote control:
    `日常聊天 ...` selects lightweight Q&A mode, `运维 ...` selects Hermes
    operations mode.  ``active_mode`` is a sticky per-chat default set by
    explicit phrases like `切到日常聊天路由`; it lets follow-up messages keep the
    lightweight lane without repeating the prefix.  Slash commands are never
    mode-routed so `/new`, `/approve`, etc. keep their established semantics.
    """
    raw = str(text or "")
    stripped = raw.lstrip()
    if not stripped:
        return _dev_mode(raw)
    if stripped.startswith("/"):
        return GatewayMessageMode(name="dev", message=raw)

    normalized = _normalize_switch_text(stripped)
    for mode_name, phrases in _STICKY_MODE_SWITCHES:
        if normalized in phrases:
            if mode_name == "dev":
                return _dev_mode(
                    message=raw,
                    sticky_mode="dev",
                    control_response="已切回开发路由。",
                )
            template = _MODE_BY_NAME[mode_name]
            response_label = "日常聊天" if mode_name == "lite" else "运维"
            copy_fn = _copy_lite_mode if template.name == "lite" else _copy_mode
            return copy_fn(
                template,
                message=raw,
                sticky_mode=mode_name,
                control_response=f"已切到{response_label}路由。",
            )

    for prefix, template in _PREFIXES:
        if stripped.startswith(prefix):
            message = _strip_mode_prefix(raw, prefix)
            if not message:
                label = "日常聊天" if template.name == "lite" else "运维"
                copy_fn = _copy_lite_mode if template.name == "lite" else _copy_mode
                return copy_fn(
                    template,
                    message=raw,
                    sticky_mode=template.name,
                    control_response=f"已切到{label}路由。",
                )
            if template.name == "lite":
                return _copy_lite_mode(template, message=message, sticky_mode=template.name)
            return _copy_mode(template, message=message, sticky_mode=template.name)

    for prefix in _DEV_PREFIXES:
        if stripped.startswith(prefix):
            message = _strip_mode_prefix(raw, prefix)
            if not message:
                return _dev_mode(
                    message=raw,
                    sticky_mode="dev",
                    control_response="已切回开发路由。",
                )
            return _dev_mode(
                message=message,
                sticky_mode="dev",
            )

    active_template = _MODE_BY_NAME.get(str(active_mode or "").strip())
    if active_template is not None:
        if active_template.name == "lite":
            return _copy_lite_mode(active_template, message=raw)
        return _copy_mode(active_template, message=raw)

    return _dev_mode(raw)


__all__ = ["GatewayMessageMode", "resolve_gateway_message_mode"]
