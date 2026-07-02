"""Bridge between run_id-based runtime controls and session_key-based live agent primitives.

The ``RuntimeControlBridge`` provides an optional layer that maps runtime
``run_id`` operations to the live GatewayRunner / AIAgent execution context.
When no bridge is present, routes fall back to standalone ``RunManager``
behaviour (Phase 11B).

Session-key mapping is deliberately NOT built into ``RunManager`` — the
runtime layer is protocol-level and session_key is a gateway-internal
identifier.  Callers that want live control must supply a mapping callable
(``get_session_key_for_run``) or the resolve primitives in
``tools.approval`` / ``tools.clarify_gateway`` that operate on
``clarify_id`` directly (which is universally unique).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from gateway.runtime.run_manager import RunManager

logger = logging.getLogger(__name__)


class RuntimeControlBridge:
    """Optional bridge between runtime run controls and live gateway execution.

    Parameters
    ----------
    run_manager:
        The ``RunManager`` instance backing the runtime routes.
    get_session_key_for_run:
        Optional callable ``(run_id: str) -> Optional[str]`` that returns
        a gateway ``session_key`` when the run has a live execution
        context, or ``None`` otherwise.
    gateway_runner_ref:
        Optional callable ``() -> Optional[GatewayRunner]`` (typically
        a ``weakref.ref``) that returns the live ``GatewayRunner``
        instance when available.

    When both ``get_session_key_for_run`` and ``gateway_runner_ref`` are
    ``None``, the bridge behaves as a transparent pass-through to
    ``RunManager`` (Phase 11B compatibility).
    """

    def __init__(
        self,
        run_manager: RunManager,
        *,
        get_session_key_for_run: Optional[Callable[[str], Optional[str]]] = None,
        gateway_runner_ref: Optional[Callable[[], Optional[Any]]] = None,
    ):
        self._run_manager = run_manager
        self._get_session_key = get_session_key_for_run
        self._gateway_ref = gateway_runner_ref
        self._bindings: Dict[str, str] = {}
        self._live_agents: Dict[str, Any] = {}

    def bind_run(self, run_id: str, session_key: str, agent: Any = None) -> None:
        self._bindings[run_id] = session_key
        if agent is not None:
            self._live_agents[run_id] = agent

    def unbind_run(self, run_id: str) -> None:
        self._bindings.pop(run_id, None)
        self._live_agents.pop(run_id, None)

    # -- resolve approval ---------------------------------------------------

    def resolve_approval(
        self,
        run_id: str,
        approval_id: str,
        choice: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve an approval, updating RunManager and optionally the live gateway.

        Returns a dict compatible with the route handler's expected shape:
        ``{"error": "not_found|conflict"}`` on failure, or a success dict
        with ``status: resolved``.
        """
        rm_result = self._run_manager.resolve_approval(
            run_id, approval_id, choice, payload=payload,
        )

        if rm_result.get("error"):
            return rm_result

        session_key = self._resolve_session_key(run_id)
        if session_key:
            try:
                import tools.approval as _approval

                _approval.resolve_gateway_approval(session_key, choice)
            except Exception:
                logger.debug(
                    "Live approval resolution failed for run=%s session=%s",
                    run_id, session_key, exc_info=True,
                )

        return rm_result

    # -- resolve clarify ----------------------------------------------------

    def resolve_clarify(
        self,
        run_id: str,
        clarify_id: str,
        answer: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve a clarify request, updating RunManager and the live gateway.

        ``clarify_id`` is universally unique, so the live resolution
        primitive operates directly on it without needing a session_key.
        """
        rm_result = self._run_manager.resolve_clarify(
            run_id, clarify_id, answer, payload=payload,
        )

        if rm_result.get("error"):
            return rm_result

        try:
            import tools.clarify_gateway as _cg

            _cg.resolve_gateway_clarify(clarify_id, answer)
        except Exception:
            logger.debug(
                "Live clarify resolution failed for run=%s clarify=%s",
                run_id, clarify_id, exc_info=True,
            )

        return rm_result

    # -- stop ---------------------------------------------------------------

    def stop_run(self, run_id: str) -> Dict[str, Any]:
        """Stop a run in RunManager and attempt live agent interrupt.

        When a live agent reference is reachable the bridge signals
        ``AIAgent.interrupt()``.  If no live context exists the
        RunManager-only cancellation proceeds (Phase 11B semantics).

        Resolution order:
        1. Direct agent reference from ``bind_run(run_id, ..., agent=...)``.
        2. Via ``session_key`` → ``GatewayRunner._running_agents``.
        3. RunManager-only (Phase 11B stand-alone fallback).

        The check for ``result_already_terminal`` uses the fact that
        ``RunManager.stop_run`` returns a ``message`` key only when
        the run was *already* in a terminal state — not when it
        was transitioned to ``cancelled`` inside this call.
        """
        rm_result = self._run_manager.stop_run(run_id)

        if rm_result.get("error") == "not_found":
            return rm_result

        already_terminal = bool(rm_result.get("message"))

        if not already_terminal:
            agent = self._live_agents.get(run_id)
            if agent is not None:
                try:
                    agent.interrupt("run_stop")
                except Exception:
                    logger.debug(
                        "Live agent interrupt failed for run=%s (direct ref)",
                        run_id, exc_info=True,
                    )
            else:
                session_key = self._resolve_session_key(run_id)
                if session_key and self._gateway_ref is not None:
                    try:
                        runner = self._gateway_ref()
                    except Exception:
                        runner = None
                    if runner is not None:
                        try:
                            agent = getattr(runner, "_running_agents", {}).get(session_key)
                            if agent is not None:
                                agent.interrupt("run_stop")
                        except Exception:
                            logger.debug(
                                "Live agent interrupt failed for run=%s session=%s",
                                run_id, session_key, exc_info=True,
                            )

        return rm_result

    # -- request approval (record + optional live notify) -------------------

    def request_approval(
        self,
        run_id: str,
        approval_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record an approval request in RunManager.

        The live gateway notify path (``register_gateway_notify``) is
        session-scoped and set up by the gateway *before* the agent runs;
        the bridge does not replicate this.  This method exists for
        callers that want a single entry point.
        """
        return self._run_manager.request_approval(
            run_id, approval_id, payload=payload,
        )

    # -- request clarify (record) -------------------------------------------

    def request_clarify(
        self,
        run_id: str,
        clarify_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record a clarify request in RunManager."""
        return self._run_manager.request_clarify(
            run_id, clarify_id, payload=payload,
        )

    # -- helpers ------------------------------------------------------------

    def _resolve_session_key(self, run_id: str) -> Optional[str]:
        """Return the gateway session_key for *run_id*, or None.

        Checks the explicit binding dict first, then falls back to the
        *get_session_key_for_run* callable.
        """
        if run_id in self._bindings:
            return self._bindings[run_id]
        if self._get_session_key is None:
            return None
        try:
            return self._get_session_key(run_id)
        except Exception:
            logger.debug(
                "session-key resolution failed for run=%s", run_id, exc_info=True,
            )
            return None

    @property
    def run_manager(self) -> RunManager:
        """The underlying RunManager (for direct access in tests)."""
        return self._run_manager
