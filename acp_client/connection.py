"""Outbound ACP connection — the public surface of ``acp_client``.

The only public import surface for the Phase-1 skeleton::

    from acp_client.connection import OutboundConnection

``OutboundConnection`` wraps an ``acp`` *client-side* connection (the object
returned by ``acp.spawn_agent_process`` / ``acp.connect_to_agent``) and threads
together the three policy/translation pieces:

* :class:`~acp_client.permission_relay.PermissionRelay` — deny-default answers
  to the external agent's ``request_permission`` calls.
* :class:`~acp_client.event_translator.EventTranslator` — inbound
  ``session_update`` notifications → Hermes-native events + mirror history.
* :class:`~acp_client.outbound_session.OutboundSessionManager` — session
  lifecycle + SessionDB persistence (``source="acp_client"``).

The inbound method handler (:class:`HermesACPClient`) implements the ``acp``
``Client`` protocol; Hermes is the policy boundary between the editor (server
side) and the external agent (this client side) — permission requests do not
chain by default (design §2.4, R9).

Phase 1 launches **no** real external CLI in tests.  :meth:`OutboundConnection.spawn`
is the real-launch path (used by Phase 2's kanban runner); the tests drive
``OutboundConnection`` with an injected fake connection and drive
``HermesACPClient`` directly.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Mapping, Optional, Sequence

from acp_client.event_translator import EventTranslator
from acp_client.outbound_session import OutboundSessionManager, OutboundSessionState
from acp_client.permission_relay import PermissionRelay
from acp_client.transport_registry import DEFAULT_REGISTRY, TransportRegistry

logger = logging.getLogger(__name__)


# ACP stop-reason → coarse Hermes category.  Fixture-tested (design R8) so the
# kanban runner (Phase 2) can branch on a stable vocabulary instead of raw
# backend strings.
_STOP_REASON_MAP = {
    "end_turn": "done",
    "max_tokens": "limit",
    "max_turn_requests": "limit",
    "max_turns": "limit",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "refusal": "refusal",
}


def normalize_stop_reason(stop_reason: Optional[str]) -> str:
    """Map an ACP ``stop_reason`` to ``done``/``limit``/``cancelled``/``refusal``/``error``."""
    if not stop_reason:
        return "error"
    return _STOP_REASON_MAP.get(str(stop_reason), "error")


class HermesACPClient:
    """Inbound ACP method handler — Hermes acting as an ACP client.

    Implements the ``acp`` ``Client`` protocol methods the external agent calls
    back into: ``session_update`` (notifications) and ``request_permission``.
    File-system and terminal callbacks are **denied** in Phase 1 — the external
    agent runs in its own workspace and must not reach back through Hermes.
    """

    def __init__(
        self,
        *,
        permission_relay: PermissionRelay,
        translators: Optional[dict[str, EventTranslator]] = None,
        on_event: Any = None,
    ):
        self.permission_relay = permission_relay
        self._translators: dict[str, EventTranslator] = translators if translators is not None else {}
        self._on_event = on_event

    def translator_for(self, session_id: str) -> EventTranslator:
        tr = self._translators.get(session_id)
        if tr is None:
            tr = EventTranslator(on_event=self._on_event)
            self._translators[session_id] = tr
        return tr

    # ---- acp.Client protocol ----------------------------------------------

    async def session_update(self, session_id: str, update: Any, **_: Any) -> None:
        """Route an inbound ``session/update`` to the per-session translator."""
        self.translator_for(session_id).translate(update)

    async def request_permission(
        self,
        options: Sequence[Any],
        session_id: str,
        tool_call: Any,
        **_: Any,
    ):
        """Answer an inbound ``session/request_permission`` with deny-default policy."""
        import acp
        from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

        decision = self.permission_relay.evaluate(
            kind=getattr(tool_call, "kind", None),
            locations=_extract_locations(tool_call),
            raw_input=getattr(tool_call, "raw_input", None),
            title=getattr(tool_call, "title", "") or "",
        )
        if decision.outcome == "allow":
            option_id = self.permission_relay.select_option(decision, options)
            if option_id is not None:
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(option_id=option_id, outcome="selected")
                )
        # Deny (default, or allow with no acceptable option).
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    # File-system + terminal callbacks: denied in Phase 1.
    async def read_text_file(self, path: str, session_id: str, **_: Any):  # noqa: D401
        _deny_callback("fs/read_text_file", path)

    async def write_text_file(self, content: str, path: str, session_id: str, **_: Any):
        _deny_callback("fs/write_text_file", path)

    async def create_terminal(self, command: str, session_id: str, **_: Any):
        _deny_callback("terminal/create", command)

    def on_connect(self, conn: Any) -> None:  # pragma: no cover - SDK hook
        self._conn = conn


def _extract_locations(tool_call: Any) -> list[str]:
    locations = getattr(tool_call, "locations", None) or []
    out: list[str] = []
    for loc in locations:
        path = getattr(loc, "path", None)
        out.append(str(path if path is not None else loc))
    return out


def _deny_callback(method: str, target: str) -> None:
    from acp.exceptions import RequestError

    logger.warning("ACP client denied callback %s for %r (Phase 1 boundary)", method, target)
    raise RequestError(
        code=-32603,
        message=f"Hermes ACP client does not service {method} (Phase 1 boundary)",
    )


class OutboundConnection:
    """High-level driver around an ``acp`` client-side connection.

    Construct directly with an existing connection for tests::

        oc = OutboundConnection(fake_conn, client=client, backend="claude")

    or via :meth:`spawn` for the real launch path (Phase 2).
    """

    def __init__(
        self,
        conn: Any,
        *,
        client: Optional[HermesACPClient] = None,
        sessions: Optional[OutboundSessionManager] = None,
        backend: str = "",
        process: Any = None,
    ):
        self._conn = conn
        self._process = process
        self.backend = backend
        self.sessions = sessions if sessions is not None else OutboundSessionManager()
        self.client = client

    # ---- lifecycle ---------------------------------------------------------

    async def initialize(self, protocol_version: Optional[int] = None):
        """Negotiate the ACP protocol version with the external agent."""
        import acp

        version = protocol_version if protocol_version is not None else acp.PROTOCOL_VERSION
        return await self._conn.initialize(protocol_version=version)

    async def create_session(
        self, cwd: str, mcp_servers: Optional[list] = None
    ) -> OutboundSessionState:
        """Open a new external session and register it locally.

        Per design §2.8, ``mcp_servers`` defaults to ``[]`` — Hermes does not
        advertise its own MCP servers outward in Phase 1.
        """
        resp = await self._conn.new_session(cwd=cwd, mcp_servers=mcp_servers or [])
        session_id = getattr(resp, "session_id", None) or str(resp)
        return self.sessions.register(session_id, cwd=cwd, backend=self.backend)

    async def load_session(
        self, session_id: str, cwd: str, mcp_servers: Optional[list] = None
    ) -> OutboundSessionState:
        """Reconnect to an existing external session (design §2.6).

        Callers that catch a failure here fall back to a fresh
        :meth:`create_session` + history-as-prefix (Phase 2 ``reconnect:
        degraded``).
        """
        await self._conn.load_session(
            cwd=cwd, session_id=session_id, mcp_servers=mcp_servers or []
        )
        state = self.sessions.get(session_id)
        if state is None:
            state = self.sessions.register(session_id, cwd=cwd, backend=self.backend)
        return state

    async def prompt(self, session_id: str, text: str):
        """Send a user prompt and return the ``PromptResponse``.

        Records the outbound prompt and the resulting stop reason on the
        session.  The inbound assistant text arrives via ``session_update``
        notifications handled by :class:`HermesACPClient`.
        """
        import acp

        self.sessions.record_history(session_id, "user", text)
        self.sessions.mark_running(session_id, True)
        try:
            resp = await self._conn.prompt(
                prompt=[acp.text_block(text)], session_id=session_id
            )
        finally:
            self.sessions.mark_running(session_id, False)

        stop_reason = getattr(resp, "stop_reason", None)
        self.sessions.set_stop_reason(session_id, stop_reason)
        return resp

    async def cancel(self, session_id: str) -> None:
        """Send ``session/cancel`` and flag local cancel state (design §2.5).

        The external subprocess is **not** killed here; it is expected to honour
        the protocol and return ``stop_reason="cancelled"``.  The Phase-2 runner
        owns the ungraceful-cancel grace+kill fallback.
        """
        with contextlib.suppress(Exception):
            await self._conn.cancel(session_id=session_id)
        self.sessions.cancel(session_id)

    # ---- real-launch path (not exercised by Phase-1 unit tests) ------------

    @classmethod
    @contextlib.asynccontextmanager
    async def spawn(
        cls,
        transport_name: str,
        *,
        cwd: str,
        workspace_path: Optional[str] = None,
        registry: Optional[TransportRegistry] = None,
        base_env: Optional[Mapping[str, str]] = None,
        sessions: Optional[OutboundSessionManager] = None,
        on_event: Any = None,
    ):
        """Spawn a known external ACP agent and yield a connected driver.

        Resolves *transport_name* against the (opt-in) transport registry,
        forwards only allowlisted env keys, and binds the deny-default
        permission relay to *workspace_path* (defaults to *cwd*).
        """
        import acp

        reg = registry or DEFAULT_REGISTRY
        spec = reg.resolve(transport_name)
        env = spec.resolve_env(base_env)
        relay = PermissionRelay(workspace_path=workspace_path or cwd, audit_log=None)
        client = HermesACPClient(permission_relay=relay, on_event=on_event)

        async with acp.spawn_agent_process(
            client, spec.command, *spec.args, env=env, cwd=cwd
        ) as (conn, proc):
            oc = cls(
                conn,
                client=client,
                sessions=sessions,
                backend=transport_name,
                process=proc,
            )
            await oc.initialize()
            yield oc
