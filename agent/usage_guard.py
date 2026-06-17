"""Usage-limit guardrails for provider-backed Hermes sessions.

The guard is intentionally conservative: it never guesses hidden account state
and it never silently changes model/provider routing.  When enabled for a
provider with account-usage visibility, it emits threshold notices and injects a
small API-time context block that tells the model to checkpoint or wind down.

Important prompt-cache invariant: this module does *not* change the tool schema
sent to the model mid-conversation.  In wind-down mode it only narrows runtime
validation/dispatch to configured safe-mode toolsets, so the cached system
prompt/tool prefix remains stable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Optional

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow, fetch_account_usage

logger = logging.getLogger(__name__)


class UsageGuardLevel(Enum):
    NORMAL = "normal"
    WARN = "warn"
    BLOCK_NEW_LONG_TASKS = "block_new_long_tasks"
    WIND_DOWN = "wind_down"


@dataclass(frozen=True)
class UsageGuardConfig:
    enabled: bool = False
    provider: str = "openai-codex"
    warn_at_percent: float = 75.0
    wind_down_at_percent: float = 90.0
    block_new_long_tasks_at_percent: float = 85.0
    fallback_requires_user_confirmation: bool = True
    safe_mode_toolsets: tuple[str, ...] = ("file", "terminal", "messaging")


@dataclass(frozen=True)
class UsageGuardDecision:
    config: UsageGuardConfig
    level: UsageGuardLevel = UsageGuardLevel.NORMAL
    used_percent: Optional[float] = None
    window_label: str = ""
    reset_at: Any = None
    status: str = ""
    context: str = ""

    @property
    def active(self) -> bool:
        return self.level is not UsageGuardLevel.NORMAL

    @property
    def wind_down(self) -> bool:
        return self.level is UsageGuardLevel.WIND_DOWN


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "on", "1"}:
            return True
        if text in {"false", "no", "off", "0"}:
            return False
    return default


def _as_percent(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(0.0, min(100.0, number))


def _as_str_tuple(value: Any, default: Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        parts = [str(part).strip() for part in value]
    else:
        parts = [str(part).strip() for part in default]
    return tuple(part for part in parts if part)


def load_usage_guard_config(config: Optional[dict[str, Any]] = None) -> UsageGuardConfig:
    """Parse the root-level ``usage_guard`` block from ``config.yaml``.

    ``config`` may be the whole Hermes config mapping or the usage_guard block
    itself.  Invalid values are fail-soft and fall back to safe defaults.
    """
    cfg = _as_mapping(config)
    raw = _as_mapping(cfg.get("usage_guard")) if "usage_guard" in cfg else cfg
    defaults = UsageGuardConfig()
    return UsageGuardConfig(
        enabled=_as_bool(raw.get("enabled"), defaults.enabled),
        provider=str(raw.get("provider") or defaults.provider).strip().lower(),
        warn_at_percent=_as_percent(raw.get("warn_at_percent"), defaults.warn_at_percent),
        wind_down_at_percent=_as_percent(raw.get("wind_down_at_percent"), defaults.wind_down_at_percent),
        block_new_long_tasks_at_percent=_as_percent(
            raw.get("block_new_long_tasks_at_percent"),
            defaults.block_new_long_tasks_at_percent,
        ),
        fallback_requires_user_confirmation=_as_bool(
            raw.get("fallback_requires_user_confirmation"),
            defaults.fallback_requires_user_confirmation,
        ),
        safe_mode_toolsets=_as_str_tuple(
            raw.get("safe_mode_toolsets"),
            defaults.safe_mode_toolsets,
        ),
    )


def _finite_used_percent(window: AccountUsageWindow) -> Optional[float]:
    value = window.used_percent
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, min(100.0, number))


def _highest_usage_window(snapshot: Optional[AccountUsageSnapshot]) -> tuple[Optional[AccountUsageWindow], Optional[float]]:
    if snapshot is None:
        return None, None
    best_window: Optional[AccountUsageWindow] = None
    best_percent: Optional[float] = None
    for window in snapshot.windows:
        percent = _finite_used_percent(window)
        if percent is None:
            continue
        if best_percent is None or percent > best_percent:
            best_window = window
            best_percent = percent
    return best_window, best_percent


def _level_for_percent(config: UsageGuardConfig, used_percent: Optional[float]) -> UsageGuardLevel:
    if used_percent is None:
        return UsageGuardLevel.NORMAL
    if used_percent >= config.wind_down_at_percent:
        return UsageGuardLevel.WIND_DOWN
    if used_percent >= config.block_new_long_tasks_at_percent:
        return UsageGuardLevel.BLOCK_NEW_LONG_TASKS
    if used_percent >= config.warn_at_percent:
        return UsageGuardLevel.WARN
    return UsageGuardLevel.NORMAL


def _format_reset_hint(window: Optional[AccountUsageWindow]) -> str:
    if not window or not window.reset_at:
        return ""
    try:
        local = window.reset_at.astimezone()
        return f" Resets at {local.strftime('%Y-%m-%d %H:%M %Z')}."
    except Exception:
        return ""


def _safe_toolsets_csv(config: UsageGuardConfig) -> str:
    return ", ".join(config.safe_mode_toolsets) if config.safe_mode_toolsets else "none"


def _status_for_decision(
    config: UsageGuardConfig,
    level: UsageGuardLevel,
    window: Optional[AccountUsageWindow],
    used: Optional[float],
) -> str:
    if level is UsageGuardLevel.NORMAL or used is None:
        return ""
    label = window.label if window else "account"
    used_display = f"{used:.0f}%"
    reset = _format_reset_hint(window)
    if level is UsageGuardLevel.WIND_DOWN:
        return (
            f"🛑 Usage guard: {config.provider} {label} usage is {used_display}, "
            f"at/above the {config.wind_down_at_percent:.0f}% wind-down threshold. "
            f"Safe wind-down mode is active; runtime tool validation is limited to: "
            f"{_safe_toolsets_csv(config)}.{reset}"
        )
    if level is UsageGuardLevel.BLOCK_NEW_LONG_TASKS:
        return (
            f"⚠️ Usage guard: {config.provider} {label} usage is {used_display}, "
            f"at/above the {config.block_new_long_tasks_at_percent:.0f}% new-long-task block threshold. "
            f"Avoid starting broad new work; checkpoint or ask before continuing.{reset}"
        )
    return (
        f"⚠️ Usage guard: {config.provider} {label} usage is {used_display}, "
        f"at/above the {config.warn_at_percent:.0f}% warning threshold.{reset}"
    )


def _context_for_decision(
    config: UsageGuardConfig,
    level: UsageGuardLevel,
    window: Optional[AccountUsageWindow],
    used: Optional[float],
) -> str:
    if level is UsageGuardLevel.NORMAL or used is None:
        return ""
    label = window.label if window else "account"
    used_display = f"{used:.0f}%"
    reset = _format_reset_hint(window)
    common = (
        "<usage-guard>\n"
        f"Provider {config.provider} {label} usage is {used_display}."
        f"{reset}\n"
    )
    if level is UsageGuardLevel.WIND_DOWN:
        return (
            common
            + "Safe wind-down mode is active. Finish only a small, already-started, safe atomic step if needed. "
            + "Do not start broad new work, new subagents, deployments, store submissions, DNS/payment/credential flows, "
            + "or other high-autonomy side-effecting work. Checkpoint current state, summarize what is done and what remains, "
            + "and ask the user before continuing after the reset or under a different provider. "
            + f"Runtime tool validation is limited to these safe-mode toolsets: {_safe_toolsets_csv(config)}.\n"
            + "</usage-guard>"
        )
    if level is UsageGuardLevel.BLOCK_NEW_LONG_TASKS:
        return (
            common
            + "New-long-task block is active. Do not begin large multi-step work, new subagents, or long-running jobs. "
            + "If the user's request is short and safe, proceed carefully; otherwise give a concise checkpoint and ask whether to wait for reset.\n"
            + "</usage-guard>"
        )
    return (
        common
        + "Warning threshold is active. Continue normally only for bounded work; prefer concise progress, avoid avoidable exploration, "
        + "and be ready to checkpoint if usage rises further.\n"
        + "</usage-guard>"
    )


def evaluate_usage_guard(
    config: UsageGuardConfig,
    snapshot: Optional[AccountUsageSnapshot],
) -> UsageGuardDecision:
    if not config.enabled:
        return UsageGuardDecision(config=config)
    if snapshot is not None and snapshot.provider:
        snapshot_provider = str(snapshot.provider).strip().lower()
        if snapshot_provider and snapshot_provider != config.provider:
            return UsageGuardDecision(config=config)
    window, used = _highest_usage_window(snapshot)
    level = _level_for_percent(config, used)
    return UsageGuardDecision(
        config=config,
        level=level,
        used_percent=used,
        window_label=window.label if window else "",
        reset_at=window.reset_at if window else None,
        status=_status_for_decision(config, level, window, used),
        context=_context_for_decision(config, level, window, used),
    )


def safe_tool_names_for_toolsets(toolsets: Iterable[str]) -> set[str]:
    names: set[str] = set()
    try:
        from toolsets import resolve_toolset
    except Exception:
        return names
    for toolset in toolsets:
        try:
            names.update(resolve_toolset(str(toolset)))
        except Exception:
            logger.debug("usage_guard: unknown safe_mode_toolset %r", toolset, exc_info=True)
    return names


def active_valid_tool_names(agent: Any) -> set[str]:
    """Return valid tool names after runtime usage-guard restrictions.

    This intentionally restricts validation/dispatch only; it does not change
    the API tool schema and therefore preserves the prompt-cache prefix.
    """
    valid = set(getattr(agent, "valid_tool_names", None) or set())
    allowed = getattr(agent, "_usage_guard_safe_tool_names", None)
    if allowed is None:
        return valid
    return valid & set(allowed)


def _agent_usage_guard_config(agent: Any) -> UsageGuardConfig:
    cfg = getattr(agent, "_usage_guard_config", None)
    if isinstance(cfg, UsageGuardConfig):
        return cfg
    try:
        from hermes_cli.config import load_config

        cfg = load_usage_guard_config(load_config())
    except Exception:
        cfg = UsageGuardConfig()
    try:
        agent._usage_guard_config = cfg
    except Exception:
        pass
    return cfg


def _provider_matches_agent(agent: Any, config: UsageGuardConfig) -> bool:
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    primary = str((getattr(agent, "_primary_runtime", None) or {}).get("provider") or "").strip().lower()
    return config.provider in {provider, primary}


def evaluate_usage_guard_for_turn(
    agent: Any,
    *,
    fetcher: Callable[[str], Optional[AccountUsageSnapshot]] = fetch_account_usage,
) -> UsageGuardDecision:
    """Fetch account usage and arm per-turn guardrails on ``agent``.

    This is fail-open by design. If the provider usage endpoint is unavailable,
    normal work continues rather than inventing a quota state.
    """
    config = _agent_usage_guard_config(agent)
    try:
        agent._usage_guard_safe_tool_names = None
    except Exception:
        pass
    if not config.enabled or not _provider_matches_agent(agent, config):
        decision = UsageGuardDecision(config=config)
        try:
            agent._usage_guard_last_decision = decision
        except Exception:
            pass
        return decision

    snapshot: Optional[AccountUsageSnapshot]
    try:
        snapshot = fetcher(config.provider)
    except Exception:
        logger.debug("usage_guard: fetch failed for provider %s", config.provider, exc_info=True)
        snapshot = None

    decision = evaluate_usage_guard(config, snapshot)
    try:
        agent._usage_guard_last_decision = decision
    except Exception:
        pass

    if decision.status:
        notice_key = (decision.level.value, decision.window_label, round(decision.used_percent or 0))
        if getattr(agent, "_usage_guard_last_notice_key", None) != notice_key:
            try:
                agent._emit_status(decision.status)
            except Exception:
                pass
            try:
                agent._usage_guard_last_notice_key = notice_key
            except Exception:
                pass

    if decision.wind_down:
        safe_names = safe_tool_names_for_toolsets(config.safe_mode_toolsets)
        try:
            agent._usage_guard_safe_tool_names = safe_names or set()
        except Exception:
            pass
        budget = getattr(agent, "iteration_budget", None)
        if budget is not None and hasattr(budget, "max_total"):
            try:
                budget.max_total = min(int(budget.max_total), 3)
            except Exception:
                pass

    return decision


def should_require_fallback_confirmation(agent: Any) -> tuple[bool, str]:
    config = _agent_usage_guard_config(agent)
    if not (config.enabled and config.fallback_requires_user_confirmation):
        return False, ""
    if getattr(agent, "_usage_guard_fallback_confirmed", False):
        return False, ""
    if not getattr(agent, "_fallback_chain", None):
        return False, ""
    if not _provider_matches_agent(agent, config):
        return False, ""
    message = (
        f"Usage guard requires explicit user confirmation before switching from {config.provider} "
        "to a fallback provider. Automatic fallback is blocked so the agent's judgment engine does not change silently. "
        "Ask the user to confirm the provider/model switch, or wait for the primary provider reset."
    )
    return True, message


__all__ = [
    "UsageGuardConfig",
    "UsageGuardDecision",
    "UsageGuardLevel",
    "active_valid_tool_names",
    "evaluate_usage_guard",
    "evaluate_usage_guard_for_turn",
    "load_usage_guard_config",
    "safe_tool_names_for_toolsets",
    "should_require_fallback_confirmation",
]
