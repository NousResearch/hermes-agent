"""Process-level Brain Network orchestrator accessors.

Keeps one orchestrator per HERMES_HOME so ECN/DMN/Limbic/Dream state survives
across turns without recreating fresh instances (the old herens turn_hooks bug).

Thread-safe via double-checked locking. Callers may also use the per-agent
``agent._brain_orchestrator`` when present — prefer that when available.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

_lock = threading.RLock()
_orchestrator = None  # type: ignore[var-annotated]
_initialized_for: Optional[str] = None


def _hermes_home_key() -> str:
    try:
        from hermes_constants import get_hermes_home

        return str(get_hermes_home())
    except Exception:
        return ""


def is_brain_networks_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when brain_networks.enabled is set in config."""
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        bn = (config or {}).get("brain_networks") or {}
        return bool(bn.get("enabled", False))
    except Exception:
        return False


def normalize_brain_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map flat brain_networks config keys into nested dmn/ecn/dreaming/limbic."""
    raw = dict(cfg or {})
    dmn = dict(raw.get("dmn") or {})
    if "dmn_reflection_chance" in raw:
        dmn.setdefault("reflection_chance", raw["dmn_reflection_chance"])
    if "use_llm_for_reflection" in raw:
        dmn.setdefault("use_llm", raw["use_llm_for_reflection"])

    ecn = dict(raw.get("ecn") or {})
    if "ecn_max_task_stack" in raw:
        ecn.setdefault("max_task_stack", raw["ecn_max_task_stack"])

    dreaming = dict(raw.get("dreaming") or {})
    if "dream_idle_threshold_seconds" in raw:
        dreaming.setdefault(
            "idle_threshold_seconds", raw["dream_idle_threshold_seconds"]
        )
    # Back-compat alias used by /dream status messaging
    if "idle_threshold_seconds" in raw and "idle_threshold_seconds" not in dreaming:
        dreaming.setdefault("idle_threshold_seconds", raw["idle_threshold_seconds"])
    if "use_llm_for_reflection" in raw:
        dreaming.setdefault("use_llm", raw["use_llm_for_reflection"])

    limbic = dict(raw.get("limbic") or {})

    out = dict(raw)
    out["dmn"] = dmn
    out["ecn"] = ecn
    out["dreaming"] = dreaming
    out["limbic"] = limbic
    return out


def get_orchestrator(
    *,
    force_reinit: bool = False,
    config: Optional[Dict[str, Any]] = None,
):
    """Return the process-wide BrainNetworkOrchestrator (initialized).

    Returns None when brain_networks is disabled (unless config forces enable
    via an explicit enabled=True dict passed in).
    """
    global _orchestrator, _initialized_for

    if config is None:
        try:
            from hermes_cli.config import load_config

            full = load_config()
            bn_cfg = (full.get("brain_networks") or {}) if isinstance(full, dict) else {}
        except Exception:
            bn_cfg = {}
    else:
        # Caller passed either full config or brain_networks section
        if "brain_networks" in config and isinstance(config.get("brain_networks"), dict):
            bn_cfg = config["brain_networks"] or {}
        else:
            bn_cfg = config

    if not bool(bn_cfg.get("enabled", False)):
        return None

    home_key = _hermes_home_key()
    with _lock:
        if (
            _orchestrator is not None
            and not force_reinit
            and _initialized_for == home_key
        ):
            return _orchestrator

        from agent.brain_networks import BrainNetworkOrchestrator

        orch = BrainNetworkOrchestrator()
        orch.initialize(normalize_brain_config(bn_cfg))
        _orchestrator = orch
        _initialized_for = home_key
        return _orchestrator


def reset_orchestrator_for_tests() -> None:
    """Clear the process singleton (tests only)."""
    global _orchestrator, _initialized_for
    with _lock:
        _orchestrator = None
        _initialized_for = None


def build_brain_turn_context(
    user_message: str,
    *,
    session_id: str = "",
    agent: Any = None,
) -> str:
    """Run on_turn_start and format a cache-safe volatile context block.

    Prefers ``agent._brain_orchestrator`` when present; falls back to the
    process singleton. Returns "" when disabled or inactive.
    """
    orch = None
    if agent is not None:
        orch = getattr(agent, "_brain_orchestrator", None)
    if orch is None:
        orch = get_orchestrator()
    if orch is None:
        return ""

    sid = session_id or getattr(agent, "session_id", "") or ""
    try:
        if hasattr(orch, "bind_session"):
            orch.bind_session(sid)
        results = orch.on_turn_start(
            {
                "user_message": user_message or "",
                "session_id": sid,
            }
        )
        if hasattr(orch, "format_turn_block"):
            return orch.format_turn_block(results) or ""
    except Exception:
        return ""
    return ""
