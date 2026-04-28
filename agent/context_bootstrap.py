"""Ephemeral context bootstrap providers.

Context bootstrap is intentionally separate from context compression and memory:
providers add fresh, API-call-time context to the next user message without
persisting it or replacing the configured context engine.
"""

from __future__ import annotations

import importlib
import logging
import shutil
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ContextBootstrapProvider(Protocol):
    """Provider interface for first-turn and delegated context packets."""

    name: str

    def is_available(self) -> bool:
        """Return whether the provider can run in this environment."""

    def context_for_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        is_first_turn: bool,
        workspace_root: Path,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return ephemeral context for the current parent turn."""

    def context_for_delegation(
        self,
        *,
        goal: str,
        context: str,
        workspace_root: Path,
    ) -> str:
        """Return ephemeral context for a delegated child task."""


class ContextBootstrapManager:
    """Small fan-out wrapper around configured context bootstrap providers."""

    def __init__(self, providers: list[ContextBootstrapProvider] | None = None):
        self.providers = providers or []

    def context_for_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        is_first_turn: bool,
        workspace_root: Path,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        parts: list[str] = []
        for provider in self.providers:
            try:
                context = provider.context_for_turn(
                    session_id=session_id,
                    user_message=user_message,
                    is_first_turn=is_first_turn,
                    workspace_root=workspace_root,
                    conversation_history=conversation_history,
                )
            except Exception as exc:
                logger.warning("Context bootstrap provider %s failed: %s", provider.name, exc)
                continue
            if context and context.strip():
                parts.append(context.strip())
        return "\n\n".join(parts)

    def context_for_delegation(
        self,
        *,
        goal: str,
        context: str = "",
        workspace_root: Path,
    ) -> str:
        parts: list[str] = []
        for provider in self.providers:
            context_for_delegation = getattr(provider, "context_for_delegation", None)
            if not callable(context_for_delegation):
                continue
            try:
                delegated_context = context_for_delegation(
                    goal=goal,
                    context=context,
                    workspace_root=workspace_root,
                )
            except Exception as exc:
                logger.warning(
                    "Context bootstrap provider %s failed for delegation: %s",
                    provider.name,
                    exc,
                )
                continue
            if delegated_context and delegated_context.strip():
                parts.append(delegated_context.strip())
        return "\n\n".join(parts)

def build_context_bootstrap_manager(
    cfg: dict[str, Any] | None,
    *,
    workspace_root: Path | None = None,
) -> ContextBootstrapManager | None:
    """Build the config-presence-driven bootstrap manager.

    Providers are not general plugins and are not activated through
    ``plugins.enabled`` or ``mcp_servers``. A provider opts in when its native
    config exists, e.g. ``lean_ctx`` for the lean-ctx provider.
    """

    providers: list[ContextBootstrapProvider] = []
    if _lean_ctx_config_present(cfg):
        provider = _load_provider(
            "plugins.context_bootstrap.lean_ctx",
            cfg=cfg or {},
            workspace_root=workspace_root,
        )
        if provider is not None:
            providers.append(provider)
    if not providers:
        return None
    return ContextBootstrapManager(providers)


def _lean_ctx_config_present(cfg: dict[str, Any] | None) -> bool:
    if not isinstance(cfg, dict):
        return False
    lean_cfg = cfg.get("lean_ctx")
    if isinstance(lean_cfg, dict):
        return _enabled_with_available_binary(lean_cfg)
    bootstrap_cfg = cfg.get("context_bootstrap")
    if isinstance(bootstrap_cfg, dict):
        nested = bootstrap_cfg.get("lean_ctx")
        if isinstance(nested, dict):
            return _enabled_with_available_binary(nested)
    return False


def _enabled_with_available_binary(raw: dict[str, Any]) -> bool:
    enabled = raw.get("enabled", "auto")
    if enabled is False:
        return False
    if isinstance(enabled, str) and enabled.strip().lower() == "auto":
        command = str(raw.get("command") or "lean-ctx")
        return shutil.which(command) is not None
    return True


def _load_provider(
    module_name: str,
    *,
    cfg: dict[str, Any],
    workspace_root: Path | None,
) -> ContextBootstrapProvider | None:
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, "create_provider", None)
        provider = factory(cfg=cfg, workspace_root=workspace_root) if factory else None
        if provider is not None and provider.is_available():
            return provider
        if provider is not None:
            logger.debug("Context bootstrap provider %s is unavailable", provider.name)
    except Exception as exc:
        logger.warning("Failed to load context bootstrap provider %s: %s", module_name, exc)
    return None
