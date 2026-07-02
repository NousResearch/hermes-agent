"""DefaultAgentFactory — constructs real AIAgent instances for the RuntimeExecutor.

Uses the same provider/credential resolution as the gateway (``_resolve_runtime_agent_kwargs``)
so that executor-owned runs behave identically to platform-owned sessions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DefaultAgentFactory:
    """Production ``AgentFactory`` that constructs real ``AIAgent`` instances.

    Credentials are resolved through the same ``_resolve_runtime_agent_kwargs``
    path used by the gateway's platform adapters.  Constructor kwargs can be
    injected explicitly for tests.

    Parameters
    ----------
    agent_kwargs:
        Pre-resolved keyword arguments for ``AIAgent.__init__``.  When
        ``None`` the factory calls ``_resolve_runtime_agent_kwargs()``
        on every ``create_agent`` call (live mode).  When provided,
        those kwargs are used directly (test mode).
    """

    def __init__(
        self,
        agent_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._agent_kwargs = agent_kwargs
        self._created_agents: list[Any] = []

    @property
    def created_agents(self) -> list[Any]:
        return list(self._created_agents)

    async def create_agent(
        self,
        run_id: str,
        session_id: str,
        message: str,
        session_key: str,
        **kwargs: Any,
    ) -> Any:
        """Construct and return an ``AIAgent`` instance.

        Resolution order:
        1. Explicit ``agent_kwargs`` injected at construction (test mode).
        2. ``_resolve_runtime_agent_kwargs()`` for live credential resolution.
        3. Run-specific overrides from *kwargs* (e.g. ``model``).
        """
        run_model = kwargs.get("model")
        resolved = self._resolve_kwargs()

        if not resolved.get("api_key"):
            raise RuntimeError(
                "No API key configured. Set the appropriate provider API key "
                "in your .env file (e.g. DEEPSEEK_API_KEY, OPENAI_API_KEY) "
                "or configure model/provider in config.yaml."
            )

        if not resolved.get("provider"):
            raise RuntimeError(
                "No provider configured. Set 'model.provider' in config.yaml "
                "(e.g. deepseek, openai, anthropic, openrouter) or ensure a "
                "default provider is available."
            )

        if not resolved.get("model") and not run_model:
            raise RuntimeError(
                "No model configured. Set 'model.default' or 'model.model' "
                "in config.yaml, or pass 'model' in the run request."
            )

        if run_model:
            resolved["model"] = run_model
        elif not resolved.get("model"):
            resolved["model"] = ""

        session_id_str = str(session_id or "")
        resolved.setdefault("session_id", session_id_str)
        resolved.setdefault("platform", "runtime_executor")
        resolved.setdefault("quiet_mode", True)
        resolved.setdefault("skip_memory", True)
        resolved.setdefault("skip_context_files", True)

        agent = self._build_agent(resolved)
        self._created_agents.append(agent)
        return agent

    def _resolve_kwargs(self) -> Dict[str, Any]:
        """Return the base keyword dict for AIAgent construction.

        Resolution sources (test vs. live):
        - When ``agent_kwargs`` was injected at construction, those are
          returned as-is (test mode).
        - Otherwise, ``_resolve_runtime_agent_kwargs()`` provides provider
          credentials and ``_resolve_gateway_model()`` provides the default
          model — matching the same resolution chain used by the gateway's
          platform adapters.
        """
        if self._agent_kwargs is not None:
            return dict(self._agent_kwargs)
        try:
            from gateway.run import (
                _resolve_runtime_agent_kwargs,
                _resolve_gateway_model,
            )
            resolved = _resolve_runtime_agent_kwargs()
            model = _resolve_gateway_model()
            if model:
                resolved["model"] = model
            return resolved
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resolve runtime agent configuration: {_redact_error(str(exc))}"
            ) from exc

    def _build_agent(self, resolved: Dict[str, Any]) -> Any:
        """Construct an AIAgent from resolved keyword arguments.

        Overridable in subclasses or mocks for testing without importing
        the real AIAgent.
        """
        try:
            from run_agent import AIAgent
        except ImportError:
            raise RuntimeError("AIAgent module (run_agent) is not available")

        try:
            return AIAgent(**resolved)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create AIAgent: {_redact_error(str(exc))}"
            ) from exc


def _redact_error(error: str) -> str:
    """Redact secrets from error messages."""
    try:
        from agent.redact import redact_sensitive_text
    except ImportError:
        return str(error)
    return redact_sensitive_text(str(error))


def create_default_agent_factory() -> DefaultAgentFactory:
    """Create a ``DefaultAgentFactory`` that resolves credentials live.

    Convenience function for wiring the factory into ``RuntimeExecutor``
    without importing ``_resolve_runtime_agent_kwargs`` at the call site.
    """
    return DefaultAgentFactory()


def create_runtime_executor_with_default_factory(
    run_manager: Any,
    *,
    control_bridge: Optional[Any] = None,
) -> Any:
    """Create a ``RuntimeExecutor`` with a ``DefaultAgentFactory``.

    Shorthand for ``RuntimeExecutor(run_manager, control_bridge=cb,
    agent_factory=DefaultAgentFactory())``.  Returns ``None`` if the
    ``DefaultAgentFactory`` cannot be created (e.g. missing dependencies).
    """
    try:
        factory = create_default_agent_factory()
    except Exception as exc:
        logger.debug("Cannot create DefaultAgentFactory: %s", _redact_error(str(exc)))
        return None
    from gateway.runtime.executor import RuntimeExecutor
    return RuntimeExecutor(
        run_manager,
        control_bridge=control_bridge,
        agent_factory=factory,
    )
