"""Shared logic for the /llm-pipeline slash command.

Toggles the `agent.llm_pipeline` block under config.yaml:
- `agent.llm_pipeline.enabled`
- `agent.llm_pipeline.providers`

Both CLI (`cli.py`) and gateway (`gateway/run.py`) use this module so behavior
matches across surfaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LlmPipelineStatus:
    """Result of a /llm-pipeline invocation.

    Callers render this however suits their surface (CLI uses prefix lines,
    gateway sends plain text).
    """

    success: bool
    new_enabled: Optional[bool] = None
    old_enabled: Optional[bool] = None
    new_providers: Optional[list[str]] = None
    old_providers: Optional[list[str]] = None
    message: str = ""
    requires_new_session: bool = False
    native_available: bool = True


def _normalize_providers(raw_providers: list[str] | tuple[str, ...] | str | None) -> list[str]:
    """Normalize providers input into a stable lower-case list."""
    if raw_providers is None:
        return []
    values: list[str]
    if isinstance(raw_providers, str):
        if not raw_providers.strip():
            return []
        parts = [p.strip() for p in raw_providers.split(",")]
    else:
        parts = [str(p).strip() for p in raw_providers]
    values = []
    for token in parts:
        if not token:
            continue
        for inner in token.split(","):
            inner = inner.strip().lower()
            if inner:
                values.append(inner)
    return values


def parse_args(arg_string: str) -> tuple[dict[str, object] | None, list[str]]:
    """Parse the slash-command argument string.

    Supported:
    - no args / `status` → read-only status
    - `on` / `off` / `enable` / `disable` → toggle enablement
    - `providers` [provider ...] → replace provider whitelist
    """
    raw = (arg_string or "").strip().lower()
    if not raw:
        return None, []

    parts = raw.split()
    op = parts[0]

    if op in {"status", "show"}:
        return None, []

    if op in {"on", "enable", "true"}:
        return {"enabled": True}, []
    if op in {"off", "disable", "false"}:
        return {"enabled": False}, []

    if op == "providers":
        return {"providers": _normalize_providers(parts[1:])}, []

    return None, [
        f"Unknown argument {op!r}. Use one of: status, on, off, providers ..."
    ]


def get_current_state(config: dict) -> tuple[bool, list[str]]:
    """Read the current llm-pipeline state from config."""
    if not isinstance(config, dict):
        return True, []
    agent_cfg = config.get("agent")
    if not isinstance(agent_cfg, dict):
        return True, []
    pipeline_cfg = agent_cfg.get("llm_pipeline")
    if not isinstance(pipeline_cfg, dict):
        return True, []

    enabled = bool(pipeline_cfg.get("enabled", True))
    providers = _normalize_providers(pipeline_cfg.get("providers", []))
    return enabled, providers


def set_state(config: dict, *, enabled: Optional[bool] = None, providers: Optional[list[str]] = None) -> tuple[bool, list[str]]:
    """Mutate config in place and return previous state (enabled, providers)."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    old_enabled, old_providers = get_current_state(config)

    if not isinstance(config.get("agent"), dict):
        config["agent"] = {}
    if not isinstance(config["agent"].get("llm_pipeline"), dict):
        config["agent"]["llm_pipeline"] = {}

    if enabled is not None:
        config["agent"]["llm_pipeline"]["enabled"] = bool(enabled)
    if providers is not None:
        config["agent"]["llm_pipeline"]["providers"] = list(providers)

    return old_enabled, old_providers


def _native_available() -> bool:
    """Best-effort check for the native llm-pipeline extension."""
    try:
        from agent.transports.ri_llm import _NATIVE_AVAILABLE

        return bool(_NATIVE_AVAILABLE)
    except Exception:  # pragma: no cover
        return False


def _providers_to_text(values: list[str]) -> str:
    return ", ".join(values) if values else "all"


def apply(
    config: dict,
    parsed: dict[str, object] | None,
    *,
    persist_callback=None,
) -> LlmPipelineStatus:
    """Top-level entry point used by CLI and gateway handlers."""
    current_enabled, current_providers = get_current_state(config)
    native_available = _native_available()

    if parsed is None:
        return LlmPipelineStatus(
            success=True,
            new_enabled=current_enabled,
            old_enabled=current_enabled,
            new_providers=list(current_providers),
            old_providers=list(current_providers),
            message=(
                f"llm-pipeline: {'on' if current_enabled else 'off'}\n"
                f"providers: {_providers_to_text(current_providers)}\n"
                f"native extension: {'available' if native_available else 'not available'}"
            ),
            native_available=native_available,
        )

    enabled: Optional[bool] = parsed.get("enabled") if isinstance(parsed.get("enabled"), bool) else None
    providers_raw = parsed.get("providers")
    providers: Optional[list[str]] = None
    if providers_raw is not None:
        providers = _normalize_providers(providers_raw)

    if enabled is None and providers is None:
        return LlmPipelineStatus(
            success=False,
            new_enabled=current_enabled,
            old_enabled=current_enabled,
            new_providers=list(current_providers),
            old_providers=list(current_providers),
            message="Could not apply /llm-pipeline command; no effective changes requested.",
            native_available=native_available,
        )

    enabled_changed = enabled is not None and enabled != current_enabled
    providers_changed = providers is not None and providers != current_providers
    if not enabled_changed and not providers_changed:
        return LlmPipelineStatus(
            success=True,
            new_enabled=current_enabled,
            old_enabled=current_enabled,
            new_providers=list(current_providers),
            old_providers=list(current_providers),
            message=(
                f"llm-pipeline already {'on' if current_enabled else 'off'} with "
                f"providers: {_providers_to_text(current_providers)}"
            ),
            native_available=native_available,
        )

    old_enabled = current_enabled
    old_providers = current_providers

    set_state(
        config,
        enabled=enabled if enabled is not None else current_enabled,
        providers=providers if providers is not None else current_providers,
    )

    new_enabled = current_enabled if enabled is None else enabled
    new_providers = current_providers if providers is None else providers

    if persist_callback is not None:
        try:
            persist_callback(config)
        except Exception as exc:
            logger.exception("failed to persist llm_pipeline change")
            return LlmPipelineStatus(
                success=False,
                new_enabled=new_enabled,
                old_enabled=old_enabled,
                new_providers=list(new_providers),
                old_providers=list(old_providers),
                message=f"updated config in memory but persist failed: {exc}",
                requires_new_session=True,
                native_available=native_available,
            )

    lines = [
        f"enabled: {'on' if old_enabled else 'off'} -> {'on' if new_enabled else 'off'}",
        f"providers: {_providers_to_text(old_providers)} -> {_providers_to_text(new_providers)}",
    ]
    if native_available:
        lines.append("native extension present")
    else:
        lines.append("native extension unavailable: transport stays on legacy path")

    return LlmPipelineStatus(
        success=True,
        new_enabled=new_enabled,
        old_enabled=old_enabled,
        new_providers=list(new_providers),
        old_providers=list(old_providers),
        message="\n".join(lines),
        requires_new_session=True,
        native_available=native_available,
    )
