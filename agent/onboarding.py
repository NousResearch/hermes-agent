"""
Contextual first-touch onboarding hints.

Instead of blocking first-run questionnaires, show a one-time hint the *first*
time a user hits a behavior fork — message-while-running, first long-running
tool, etc.  Each hint is shown once per install (tracked in ``config.yaml`` under
``onboarding.seen.<flag>``) and then never again.

Keep this module tiny and dependency-free so both the CLI and gateway can import
it without pulling in heavy modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Flag names (stable — used as config.yaml keys under onboarding.seen)
# -------------------------------------------------------------------------

BUSY_INPUT_FLAG = "busy_input_prompt"
TOOL_PROGRESS_FLAG = "tool_progress_prompt"
OPENCLAW_RESIDUE_FLAG = "openclaw_residue_cleanup"


# -------------------------------------------------------------------------
# Hint content
# -------------------------------------------------------------------------

def busy_input_hint_gateway(mode: str) -> str:
    """Hint shown the first time a user messages while the agent is busy.

    ``mode`` is the effective busy_input_mode that was just applied, so the
    message matches reality ("I just interrupted…" vs "I just queued…").
    """
    if mode == "queue":
        return (
            "💡 首次提示：我已把你的消息排队，没有中断当前任务。"
            "发送 `/busy interrupt` 可改为新消息立即中断当前任务，"
            "或发送 `/busy status` 查看状态。此提示不会再出现。"
        )
    if mode == "steer":
        return (
            "💡 首次提示：我已把你的消息插入当前任务；"
            "它会在下一个工具调用后进入，不会中断任务。"
            "发送 `/busy interrupt` 或 `/busy queue` 可切换模式，"
            "或发送 `/busy status` 查看状态。此提示不会再出现。"
        )
    return (
        "💡 首次提示：我刚刚中断当前任务来回复你。"
        "发送 `/busy queue` 可让后续消息排队到当前任务之后，"
        "发送 `/busy steer` 可在不中断的情况下插入当前任务，"
        "或发送 `/busy status` 查看状态。此提示不会再出现。"
    )


def busy_input_hint_cli(mode: str) -> str:
    """CLI version of the busy-input hint (plain text, no markdown)."""
    if mode == "queue":
        return (
            "(tip) Your message was queued for the next turn. "
            "Use /busy interrupt to make Enter stop the current run instead, "
            "or /busy steer to inject mid-run. This tip only shows once."
        )
    if mode == "steer":
        return (
            "(tip) Your message was steered into the current run; it arrives "
            "after the next tool call. Use /busy interrupt or /busy queue to "
            "change this. This tip only shows once."
        )
    return (
        "(tip) Your message interrupted the current run. "
        "Use /busy queue to queue messages for the next turn instead, "
        "or /busy steer to inject mid-run. This tip only shows once."
    )


def tool_progress_hint_gateway() -> str:
    return (
        "💡 首次提示：这个工具运行较久，我正在流式显示每一步。"
        "如果觉得进度消息太多，发送 `/verbose` 可切换显示模式"
        "（all → new → off）。此提示不会再出现。"
    )


def tool_progress_hint_cli() -> str:
    return (
        "(tip) That tool ran for a while. Use /verbose to cycle tool-progress "
        "display modes (all -> new -> off -> verbose). This tip only shows once."
    )


def openclaw_residue_hint_cli() -> str:
    """Banner shown the first time Hermes starts and finds ``~/.openclaw/``.

    Points users at ``hermes claw migrate`` (non-destructive port of config,
    memory, and skills) first. ``hermes claw cleanup`` is mentioned as the
    follow-up step for users who have already migrated and want to archive
    the old directory — with a warning that archiving breaks OpenClaw.
    """
    return (
        "A legacy OpenClaw directory was detected at ~/.openclaw/.\n"
        "To port your config, memory, and skills over to Hermes, run "
        "`hermes claw migrate`.\n"
        "If you've already migrated and want to archive the old directory, "
        "run `hermes claw cleanup` (renames it to ~/.openclaw.pre-migration — "
        "OpenClaw will stop working after this).\n"
        "This tip only shows once."
    )


def detect_openclaw_residue(home: Optional[Path] = None) -> bool:
    """Return True if an OpenClaw workspace directory is present in ``$HOME``.

    Pure filesystem check — no side effects. ``home`` override exists for tests.
    """
    base = home or Path.home()
    try:
        return (base / ".openclaw").is_dir()
    except OSError:
        return False


# -------------------------------------------------------------------------
# State read / write
# -------------------------------------------------------------------------

def _get_seen_dict(config: Mapping[str, Any]) -> Mapping[str, Any]:
    onboarding = config.get("onboarding") if isinstance(config, Mapping) else None
    if not isinstance(onboarding, Mapping):
        return {}
    seen = onboarding.get("seen")
    return seen if isinstance(seen, Mapping) else {}


def is_seen(config: Mapping[str, Any], flag: str) -> bool:
    """Return True if the user has already been shown this first-touch hint."""
    return bool(_get_seen_dict(config).get(flag))


def mark_seen(config_path: Path, flag: str) -> bool:
    """Persist ``onboarding.seen.<flag> = True`` to ``config_path``.

    Uses the atomic YAML writer so a concurrent process can't observe a
    partially-written file.  Returns True on success, False on any error
    (including the config file being absent — onboarding is best-effort).
    """
    try:
        import yaml
        from utils import atomic_yaml_write
    except Exception as e:  # pragma: no cover — dependency issue
        logger.debug("onboarding: failed to import yaml/utils: %s", e)
        return False

    try:
        cfg: dict = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg.get("onboarding"), dict):
            cfg["onboarding"] = {}
        seen = cfg["onboarding"].get("seen")
        if not isinstance(seen, dict):
            seen = {}
            cfg["onboarding"]["seen"] = seen
        if seen.get(flag) is True:
            return True  # already marked — nothing to do
        seen[flag] = True
        atomic_yaml_write(config_path, cfg)
        return True
    except Exception as e:
        logger.debug("onboarding: failed to mark flag %s: %s", flag, e)
        return False


__all__ = [
    "BUSY_INPUT_FLAG",
    "TOOL_PROGRESS_FLAG",
    "OPENCLAW_RESIDUE_FLAG",
    "busy_input_hint_gateway",
    "busy_input_hint_cli",
    "tool_progress_hint_gateway",
    "tool_progress_hint_cli",
    "openclaw_residue_hint_cli",
    "detect_openclaw_residue",
    "is_seen",
    "mark_seen",
]
