"""Tests for acp_client.connection — OutboundConnection + HermesACPClient.

No real external CLI is launched. ``OutboundConnection`` is driven against a
fake async connection; ``HermesACPClient`` is exercised directly for inbound
permission + session_update handling.
"""

import pathlib
from types import SimpleNamespace

import pytest

import acp
from acp.schema import AllowedOutcome, DeniedOutcome

from hermes_state import SessionDB
from acp_client.connection import (
    HermesACPClient,
    OutboundConnection,
    normalize_stop_reason,
)
from acp_client.event_translator import EventTranslator
from acp_client.outbound_session import OutboundSessionManager
from acp_client.permission_relay import PermissionRelay


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeConn:
    """Minimal stand-in for an acp ClientSideConnection (no subprocess)."""

    def __init__(self, *, session_id="ext-sess", stop_reason="end_turn"):
        self._session_id = session_id
        self._stop_reason = stop_reason
        self.calls = []

    async def initialize(self, *, protocol_version):
        self.calls.append(("initialize", protocol_version))
        return SimpleNamespace(protocol_version=protocol_version)

    async def new_session(self, *, cwd, mcp_servers=None):
        self.calls.append(("new_session", cwd, mcp_servers))
        return SimpleNamespace(session_id=self._session_id)

    async def load_session(self, *, cwd, session_id, mcp_servers=None):
        self.calls.append(("load_session", cwd, session_id, mcp_servers))
        return SimpleNamespace(session_id=session_id)

    async def prompt(self, *, prompt, session_id, message_id=None):
        self.calls.append(("prompt", session_id, prompt))
        return SimpleNamespace(stop_reason=self._stop_reason)

    async def cancel(self, *, session_id):
        self.calls.append(("cancel", session_id))


@pytest.fixture()
def manager(tmp_path):
    return OutboundSessionManager(db=SessionDB(pathlib.Path(tmp_path) / "state.db"))


# ---------------------------------------------------------------------------
# stop_reason normalisation (design R8)
# ---------------------------------------------------------------------------


class TestStopReason:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("end_turn", "done"),
            ("cancelled", "cancelled"),
            ("canceled", "cancelled"),
            ("refusal", "refusal"),
            ("max_tokens", "limit"),
            ("max_turn_requests", "limit"),
            ("something_new", "error"),
            (None, "error"),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_stop_reason(raw) == expected


# ---------------------------------------------------------------------------
# OutboundConnection lifecycle (create / prompt / cancel / reconnect)
# ---------------------------------------------------------------------------


class TestOutboundConnection:
    @pytest.mark.asyncio
    async def test_create_session_registers_and_forwards_cwd(self, manager):
        conn = FakeConn(session_id="ext-1")
        oc = OutboundConnection(conn, sessions=manager, backend="claude")
        state = await oc.create_session(cwd="/work")
        assert state.session_id == "ext-1"
        assert state.cwd == "/work"
        assert state.backend == "claude"
        # cwd + empty mcp_servers forwarded to new_session (design §2.7/§2.8).
        assert conn.calls[-1] == ("new_session", "/work", [])

    @pytest.mark.asyncio
    async def test_prompt_records_history_and_stop_reason(self, manager):
        conn = FakeConn(session_id="ext-2", stop_reason="end_turn")
        oc = OutboundConnection(conn, sessions=manager, backend="claude")
        await oc.create_session(cwd="/work")
        resp = await oc.prompt("ext-2", "do the task")
        assert resp.stop_reason == "end_turn"
        state = manager.get("ext-2")
        assert state.history[0] == {"role": "user", "content": "do the task"}
        assert state.last_stop_reason == "end_turn"
        assert state.is_running is False

    @pytest.mark.asyncio
    async def test_cancel_sends_rpc_and_sets_local_state(self, manager):
        conn = FakeConn(session_id="ext-3")
        oc = OutboundConnection(conn, sessions=manager, backend="claude")
        await oc.create_session(cwd="/work")
        await oc.cancel("ext-3")
        assert ("cancel", "ext-3") in conn.calls
        assert manager.get("ext-3").cancel_event.is_set() is True

    @pytest.mark.asyncio
    async def test_load_session_reconnects(self, manager):
        # Pre-existing persisted session (as if a prior worker created it).
        manager.register("ext-4", cwd="/work", backend="claude")
        conn = FakeConn(session_id="ext-4")
        oc = OutboundConnection(conn, sessions=manager, backend="claude")
        state = await oc.load_session("ext-4", cwd="/work")
        assert state.session_id == "ext-4"
        assert ("load_session", "/work", "ext-4", []) in conn.calls

    @pytest.mark.asyncio
    async def test_initialize_uses_protocol_version(self, manager):
        conn = FakeConn()
        oc = OutboundConnection(conn, sessions=manager)
        await oc.initialize()
        assert conn.calls[0] == ("initialize", acp.PROTOCOL_VERSION)


# ---------------------------------------------------------------------------
# HermesACPClient inbound handling
# ---------------------------------------------------------------------------


class TestHermesACPClient:
    def _client(self, tmp_path):
        relay = PermissionRelay(workspace_path=str(tmp_path))
        return HermesACPClient(permission_relay=relay), relay

    @pytest.mark.asyncio
    async def test_session_update_routes_to_translator(self, tmp_path):
        client, _ = self._client(tmp_path)
        await client.session_update("s1", acp.update_agent_message(acp.text_block("hi")))
        assert client.translator_for("s1").message_text == "hi"

    @pytest.mark.asyncio
    async def test_request_permission_allows_inside_workspace(self, tmp_path):
        client, _ = self._client(tmp_path)
        target = str(pathlib.Path(tmp_path) / "a.py")
        tool_call = acp.update_tool_call(
            "tc", title="read a.py", kind="read", status="pending"
        )
        tool_call.locations = [SimpleNamespace(path=target)]
        options = [
            acp.schema.PermissionOption(option_id="allow_once", kind="allow_once", name="Allow once"),
            acp.schema.PermissionOption(option_id="reject_once", kind="reject_once", name="Deny"),
        ]
        resp = await client.request_permission(
            options=options, session_id="s1", tool_call=tool_call
        )
        assert isinstance(resp.outcome, AllowedOutcome)
        assert resp.outcome.option_id == "allow_once"

    @pytest.mark.asyncio
    async def test_request_permission_denies_execute(self, tmp_path):
        client, _ = self._client(tmp_path)
        tool_call = acp.update_tool_call(
            "tc", title="run", kind="execute", status="pending"
        )
        options = [
            acp.schema.PermissionOption(option_id="allow_once", kind="allow_once", name="Allow once"),
        ]
        resp = await client.request_permission(
            options=options, session_id="s1", tool_call=tool_call
        )
        assert isinstance(resp.outcome, DeniedOutcome)

    @pytest.mark.asyncio
    async def test_fs_callbacks_are_denied(self, tmp_path):
        from acp.exceptions import RequestError

        client, _ = self._client(tmp_path)
        with pytest.raises(RequestError):
            await client.read_text_file(path="/etc/passwd", session_id="s1")
        with pytest.raises(RequestError):
            await client.write_text_file(content="x", path="/tmp/y", session_id="s1")


class TestAuthRelay:
    def test_auth_forwarding_is_refused(self):
        from acp_client.auth_relay import AuthForwardingRefused, assert_no_credential_forwarding

        with pytest.raises(AuthForwardingRefused):
            assert_no_credential_forwarding("anthropic-api-key")
