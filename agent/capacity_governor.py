"""Pre-flight capacity checks for provider-backed LLM calls.

The governor is deliberately fail-open and disabled by default. When enabled,
it prevents non-interactive growth/background work from spending calls on a
provider that Hermes already knows is quota-exhausted, while preserving the
interactive main loop and critical compression path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_BLOCKED_TASK_CLASSES = {
    "background_review",
    "title",
    "session_search",
    "skills_hub",
    "profile_describer",
    "triage_specifier",
    "curator",
    "cron_judgment",
    "delegation",
}

PROTECTED_TASK_CLASSES = {
    "interactive_main",
    "compression_critical",
}


@dataclass(frozen=True)
class CapacityDecision:
    allowed: bool
    provider: str
    model: Optional[str]
    task_class: str
    reason: str = ""
    reset_at: Optional[float] = None
    action: str = "allow"


def normalize_provider(provider: Optional[str]) -> str:
    raw = str(provider or "").strip().lower()
    if raw in {"codex", "openai_codex"}:
        return "openai-codex"
    return raw


def is_codex_provider(provider: Optional[str]) -> bool:
    return normalize_provider(provider) == "openai-codex"


def task_class_for_auxiliary_task(
    task: Optional[str],
    *,
    raw_codex: bool = False,
) -> str:
    if raw_codex and not task:
        return "interactive_main"
    normalized = str(task or "").strip().lower()
    if normalized in {"compression", "context_compression", "summary"}:
        return "compression_critical"
    if normalized in {"title", "title_generation", "generate_title"}:
        return "title"
    if normalized.startswith("curator"):
        return "curator"
    if normalized in {
        "session_search",
        "skills_hub",
        "profile_describer",
        "triage_specifier",
        "vision",
        "web_extract",
        "mcp",
    }:
        return normalized
    return normalized or "auxiliary"


def _load_governor_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
        except Exception:
            cfg = {}
    raw = (cfg or {}).get("capacity_governor") or {}
    return raw if isinstance(raw, dict) else {}


def _blocked_task_classes(raw_cfg: Dict[str, Any]) -> set[str]:
    raw = raw_cfg.get("block_task_classes")
    if raw is None:
        return set(DEFAULT_BLOCKED_TASK_CLASSES)
    if not isinstance(raw, list):
        return set(DEFAULT_BLOCKED_TASK_CLASSES)
    return {str(item).strip().lower() for item in raw if str(item).strip()}


def _codex_status() -> Dict[str, Any]:
    from hermes_cli.auth import get_codex_auth_status

    return get_codex_auth_status() or {}


def _is_rate_limited(status: Dict[str, Any]) -> bool:
    return bool(
        status.get("rate_limited")
        or status.get("error_code") == "codex_rate_limited"
    )


def check_capacity(
    *,
    provider: Optional[str],
    model: Optional[str] = None,
    task_class: str,
    estimated_tokens: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> CapacityDecision:
    """Return whether a call should proceed.

    ``estimated_tokens`` is accepted for the public API but not enforced yet;
    the first implementation only gates known exhausted providers.
    """
    del estimated_tokens
    normalized_provider = normalize_provider(provider)
    normalized_task = str(task_class or "auxiliary").strip().lower()
    cfg = _load_governor_config(config)

    if not cfg.get("enabled", False):
        return CapacityDecision(True, normalized_provider, model, normalized_task)
    if not is_codex_provider(normalized_provider):
        return CapacityDecision(True, normalized_provider, model, normalized_task)
    if normalized_task in PROTECTED_TASK_CLASSES and cfg.get("protect_interactive_main", True):
        return CapacityDecision(True, normalized_provider, model, normalized_task)

    try:
        status = _codex_status()
    except Exception:
        return CapacityDecision(True, normalized_provider, model, normalized_task)

    if not _is_rate_limited(status):
        return CapacityDecision(True, normalized_provider, model, normalized_task)

    if normalized_task not in _blocked_task_classes(cfg):
        return CapacityDecision(True, normalized_provider, model, normalized_task)

    reset_at = status.get("reset_at")
    if not isinstance(reset_at, (int, float)):
        reset_at = None
    reason = (
        f"Codex quota is exhausted; deferred {normalized_task}"
        + (f" until reset_at={reset_at}" if reset_at else "")
    )
    return CapacityDecision(
        False,
        normalized_provider,
        model,
        normalized_task,
        reason=reason,
        reset_at=float(reset_at) if reset_at else None,
        action="defer",
    )


def reserve(*args: Any, **kwargs: Any) -> CapacityDecision:
    """Compatibility entry point for future token reservation accounting."""
    return check_capacity(*args, **kwargs)


def release(*args: Any, **kwargs: Any) -> None:
    """No-op placeholder for the future reservation API."""
    del args, kwargs
    return None


def format_capacity_block_message(decision: CapacityDecision) -> str:
    if decision.reason:
        return decision.reason
    return f"Capacity governor blocked {decision.task_class} on {decision.provider}"
