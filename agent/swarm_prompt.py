"""Prompt contract for the Jeeves swarm operator.

This module is intentionally small and side-effect free. The operator prompt is
only emitted when explicitly enabled in config; disabled-by-default rollout must
not bloat normal Hermes sessions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                value = getter(key)
                return default if value is None else value
            except Exception:
                return default
    return getattr(config, key, default)


def swarm_operator_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return whether the prompt contract should be injected."""
    if config is None:
        try:
            from hermes_cli.config import cfg_get, load_config

            config = cfg_get(load_config(), "swarm_operator", default={}) or {}
        except Exception:
            config = {}
    return bool(_get_config_value(config, "enabled", False))


def build_swarm_operator_prompt(config: Optional[Dict[str, Any]] = None) -> str:
    """Build the stable Jeeves operator prompt section when enabled.

    The returned text is deterministic and compact. It does not enable live
    gateway interception or dispatch anything; it only tells the parent agent how
    to use existing Hermes primitives when the operator feature is enabled.
    """
    if config is None:
        try:
            from hermes_cli.config import cfg_get, load_config

            config = cfg_get(load_config(), "swarm_operator", default={}) or {}
        except Exception:
            config = {}
    if not swarm_operator_enabled(config):
        return ""

    max_children = _get_config_value(config, "max_children", 3)
    dry_run = _get_config_value(config, "dry_run", True)
    persist_to_honcho = _get_config_value(config, "persist_to_honcho", False)

    return (
        "# Jeeves Swarm Operator Contract\n"
        "When the swarm operator is enabled, you are Jeeves: the single parent "
        "operator responsible for triage, routing, verification, and one final "
        "user-facing synthesis. Child agents must never send messages to the user.\n"
        "\n"
        "## Routing rules\n"
        "- Handle simple explanatory or single-step work directly.\n"
        "- Use delegate_task only for independent subtasks that benefit from parallel "
        f"work, bounded by max_children={max_children}.\n"
        "- Use scripts for >3 procedural steps/source-of-truth/validation/eval; "
        "prefer deterministic local checks over ad-hoc conversational repetition.\n"
        "- Use cron or kanban for durable work that must outlive the current turn; "
        "delegate_task is only for synchronous fan-out inside this run.\n"
        "\n"
        "## Permission model\n"
        "- External, destructive, deploy, payment, publish, send/message/email, or "
        "cross-system side effects are default-denied unless already allowed by "
        "Hermes policy or explicitly approved by the user.\n"
        "- Permission-required subtasks stay blocked; do not dispatch children for "
        "actions that need approval.\n"
        "\n"
        "## Subsystem policy\n"
        "- n8n/docker are dumb pipes: use them only as execution substrates with an "
        "explicit payload contract, dry-run/verification evidence, and permission "
        "where required. They are not reasoning engines.\n"
        "- Honcho/memory persistence is parent-owned: persist compact final summaries "
        f"only when configured (persist_to_honcho={bool(persist_to_honcho)}), never "
        "raw child scratchpads or secrets.\n"
        "\n"
        "## Status and rollout\n"
        f"- Feature dry_run={bool(dry_run)}; do not assume live gateway interception.\n"
        "- Keep job status inspectable: note active tasks, blockers, permission "
        "requests, verification evidence, and final outcome."
    )


__all__ = ["build_swarm_operator_prompt", "swarm_operator_enabled"]
