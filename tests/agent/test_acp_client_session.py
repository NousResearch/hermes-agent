"""Tests for agent.transports.acp_client_session -- ACP session adapter.

Tests cover session lifecycle (ensure_started, close), turn execution,
streaming delta projection, should_retire policy on crash/timeout, and
server-request handling (permission allow, fs/terminal decline).

Fix-specific tests:
  Fix 1 -- model pin: set_config_option sent after session/new when model is set.
  Fix 2 -- thought-chunk leak: agent_thought_chunk NOT in extracted text; agent_message_chunk IS.
  Fix 3 -- permission response shape: uses {outcome:{outcome:...}} ACP spec form.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional
from unittest.mock import MagicMock, call

import pytest

from agent.transports.acp_client import ACPClientError
from agent.transports.acp_client_session import (
    ACPClientSession,
    TurnResult,
    _coerce_user_input,
    _extract_text_from_update,
    _is_tool_iteration,
    _pick_allow_option,
    _translate_mcp_servers,
)


# ---------------------------------------------------------------------------
# Helpers -- mock ACPClient
# ---------------------------------------------------------------------------


def _make_session(
    *,
    command: str = "fake-acp",
    args=None,
    model: Optional[str] = None,
    on_delta=None,
    client_mock: Optional[MagicMock] = None,
) -> tuple[ACPClientSession, MagicMock]:
    """Return an ACPClientSession with a mock ACPClient injected."""
    if client_mock is None:
        client_mock = MagicMock()
        client_mock.is_alive.return_value = True
        client_mock.initialize.return_value = {"protocolVersion": 1}
        client_mock.request.return_value = {}
        client_mock.take_notification.return_value = None
        client_mock.take_server_request.return_value = None
        client_mock.stderr_tail.return_value = []

    session = ACPClientSession(
        command=command,
        args=args,
        model=model,
        on_delta=on_delta,
        client_factory=lambda **kw: client_mock,
    )
    return session, client_mock


# ---------------------------------------------------------------------------
# Tests: ensure_started / session lifecycle
# ---------------------------------------------------------------------------


class TestEnsureStarted:
    def test_ensure_started_initializes_and_creates_session(self):
        """ensure_started() calls initialize then session/new, stores session_id."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-abc-123"},  # session/new
        ]

        sid = session.ensure_started(cwd="/tmp")
        assert sid == "sess-abc-123"
        assert session._session_id == "sess-abc-123"

        # initialize was called once
        mock_client.initialize.assert_called_once()
        # session/new was called with correct cwd
        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "session/new"
        assert call_args[0][1]["cwd"] == "/tmp"

    def test_ensure_started_idempotent(self):
        """ensure_started() called twice returns same session_id."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "sess-001"}

        sid1 = session.ensure_started(cwd="/tmp")
        sid2 = session.ensure_started(cwd="/other")
        assert sid1 == sid2 == "sess-001"
        # initialize and session/new called only once
        assert mock_client.initialize.call_count == 1
        assert mock_client.request.call_count == 1

    def test_ensure_started_raises_on_missing_session_id(self):
        """ensure_started() raises ACPClientError if no sessionId in response."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {}  # no sessionId

        with pytest.raises(ACPClientError) as exc_info:
            session.ensure_started()
        assert "sessionId" in str(exc_info.value)
        assert session._session_id is None

    def test_ensure_started_error_sets_should_retire(self):
        """run_turn() -> ensure_started() failure sets should_retire=True."""
        session, mock_client = _make_session()
        mock_client.initialize.side_effect = ACPClientError(
            code=-32603, message="initialize failed"
        )

        result = session.run_turn("hello")
        assert result.should_retire is True
        assert result.error is not None
        assert "startup" in result.error.lower()


# ---------------------------------------------------------------------------
# Tests: Fix 1 -- model pin via session/set_config_option (verify behaviour)
# ---------------------------------------------------------------------------

def _make_config_response(current_value: str) -> dict:
    """Build a realistic set_config_option response with the given model currentValue."""
    return {
        "configOptions": [
            {
                "id": "model",
                "name": "Model",
                "type": "select",
                "category": "model",
                "currentValue": current_value,
                "options": [
                    {"value": "default", "name": "Default (Opus 4.8)"},
                    {"value": "sonnet",  "name": "Sonnet 4.6"},
                    {"value": "haiku",   "name": "Haiku 4.5"},
                ],
            },
        ]
    }


class TestModelPin:
    def test_set_config_option_sent_after_session_new_when_model_set(self):
        """Fix 1: when model is configured, session/set_config_option is sent
        after session/new with configId='model' and the resolved model string."""
        session, mock_client = _make_session(model="haiku")
        mock_client.request.side_effect = [
            {"sessionId": "sess-model"},    # session/new
            _make_config_response("haiku"), # set_config_option -> match
        ]

        session.ensure_started(cwd="/tmp")

        calls = mock_client.request.call_args_list
        assert len(calls) == 2

        new_call = calls[0]
        assert new_call[0][0] == "session/new"

        cfg_call = calls[1]
        assert cfg_call[0][0] == "session/set_config_option"
        params = cfg_call[0][1]
        assert params["sessionId"] == "sess-model"
        assert params["configId"] == "model"
        assert params["value"] == "haiku"

    def test_set_config_option_not_sent_when_model_not_set(self):
        """Fix 1: when no model is configured, set_config_option is NOT sent
        (existing tests must remain unaffected -- only session/new is called)."""
        session, mock_client = _make_session()  # no model
        mock_client.request.side_effect = [
            {"sessionId": "sess-nomodel"},  # session/new only
        ]

        session.ensure_started(cwd="/tmp")

        # Only one request call: session/new, no set_config_option
        assert mock_client.request.call_count == 1
        assert mock_client.request.call_args[0][0] == "session/new"

    # -- verify: currentValue matches -> silent OK --

    def test_model_pin_verified_match_is_silent(self):
        """Task B: currentValue == requested -> log info, no exception, session OK."""
        session, mock_client = _make_session(model="haiku")
        mock_client.request.side_effect = [
            {"sessionId": "sess-match"},
            _make_config_response("haiku"),  # currentValue matches
        ]

        sid = session.ensure_started(cwd="/tmp")
        assert sid == "sess-match"
        assert session._session_id == "sess-match"

    # -- verify: currentValue mismatch (server supported) -> raises, not swallowed --

    def test_model_pin_mismatch_raises_acp_error(self):
        """Task B: server supported set_config_option but currentValue != requested
        -> ACPClientError raised (NOT swallowed by the tolerance except)."""
        session, mock_client = _make_session(model="haiku")
        mock_client.request.side_effect = [
            {"sessionId": "sess-mismatch"},
            _make_config_response("default"),  # currentValue stayed on Opus default
        ]

        with pytest.raises(ACPClientError) as exc_info:
            session.ensure_started(cwd="/tmp")
        err = exc_info.value
        assert err.code == 1  # positive = config rejection, not transport crash
        assert "haiku" in str(err)
        assert "default" in str(err)
        # Session cleared so retry does not short-circuit idempotency guard
        assert session._session_id is None

    def test_model_pin_mismatch_does_not_retire_session(self):
        """Task B: mismatch is a config error, not a session crash.
        should_retire must be False so we don't loop (respawn -> same mismatch -> loop)."""
        session, mock_client = _make_session(model="haiku")

        def req_side(method, params=None, timeout=30):
            if method == "session/new":
                return {"sessionId": "sess-noretire"}
            if method == "session/set_config_option":
                return _make_config_response("default")  # mismatch
            return {}

        mock_client.request.side_effect = req_side

        result = session.run_turn("hello")
        assert result.error is not None
        assert "haiku" in result.error
        assert result.should_retire is False  # MUST NOT retire -> would loop

    def test_model_pin_mismatch_error_message_names_accepted_aliases(self):
        """Task B: error message includes the accepted alias list so operator can fix."""
        session, mock_client = _make_session(model="claude-haiku-4-5-20251001")
        mock_client.request.side_effect = [
            {"sessionId": "sess-alias"},
            _make_config_response("default"),
        ]

        with pytest.raises(ACPClientError) as exc_info:
            session.ensure_started(cwd="/tmp")
        msg = str(exc_info.value)
        # Should list the accepted values from the options array
        assert "haiku" in msg
        assert "sonnet" in msg
        assert "default" in msg

    # -- verify: request() raises (server lacks method) -> tolerated --

    def test_set_config_option_request_raises_is_tolerated(self):
        """Task B: if request() raises (server doesn't support set_config_option),
        session is NOT retired -- ensure_started returns the session_id."""
        session, mock_client = _make_session(model="haiku")
        mock_client.request.side_effect = [
            {"sessionId": "sess-cfg-fail"},
            ACPClientError(code=-32601, message="Method not found"),  # set_config_option
        ]

        sid = session.ensure_started(cwd="/tmp")
        assert sid == "sess-cfg-fail"  # session not aborted
        assert session._session_id == "sess-cfg-fail"

    def test_set_config_option_timeout_is_tolerated(self):
        """Task B: TimeoutError from set_config_option is tolerated (not a mismatch)."""
        session, mock_client = _make_session(model="haiku")
        mock_client.request.side_effect = [
            {"sessionId": "sess-timeout"},
            TimeoutError("set_config_option timed out"),
        ]

        sid = session.ensure_started(cwd="/tmp")
        assert sid == "sess-timeout"
        assert session._session_id == "sess-timeout"

    # -- verify: no model configOption in response -> warn + proceed --

    def test_no_model_config_option_in_response_is_tolerated(self):
        """Task B: server responds but carries no 'model' configOption
        (generic ACP server) -- cannot verify, warn + proceed."""
        session, mock_client = _make_session(model="haiku")
        mock_client.request.side_effect = [
            {"sessionId": "sess-generic"},
            {"configOptions": []},  # no model option
        ]

        sid = session.ensure_started(cwd="/tmp")
        assert sid == "sess-generic"
        assert session._session_id == "sess-generic"


# ---------------------------------------------------------------------------
# Tests: Fix 2 -- thought-chunk leak guard
# ---------------------------------------------------------------------------


class TestThoughtChunkLeak:
    def test_agent_message_chunk_extracted(self):
        """Fix 2: agent_message_chunk produces user-facing text (positive case)."""
        params = {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "hello"},
            },
        }
        assert _extract_text_from_update(params) == "hello"

    def test_agent_thought_chunk_not_extracted(self):
        """Fix 2: agent_thought_chunk (internal reasoning) returns empty string --
        must never leak into the user-facing reply."""
        params = {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "I am thinking..."},
            },
        }
        assert _extract_text_from_update(params) == ""

    def test_agent_thought_chunk_not_in_run_turn_output(self):
        """Fix 2: thought chunks from session/update are NOT included in final_text."""
        session, mock_client = _make_session()

        def req_side_effect(method, params=None, timeout=30):
            if method == "session/new":
                return {"sessionId": "sess-think"}
            if method == "session/prompt":
                time.sleep(0.05)
                return {"stopReason": "end_turn"}
            return {}

        mock_client.request.side_effect = req_side_effect

        notes = iter([
            # internal reasoning -- must be excluded
            {
                "method": "session/update",
                "params": {
                    "sessionId": "sess-think",
                    "update": {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": "reasoning step"},
                    },
                },
            },
            # user-facing reply -- must be included
            {
                "method": "session/update",
                "params": {
                    "sessionId": "sess-think",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "user reply"},
                    },
                },
            },
            None,
        ])
        mock_client.take_notification.side_effect = lambda timeout=0.0: next(notes, None)

        result = session.run_turn("test")
        assert "user reply" in result.final_text
        assert "reasoning step" not in result.final_text

    def test_snake_case_thought_chunk_not_extracted(self):
        """Fix 2: snake_case 'session_update' discriminator variant also blocked."""
        params = {
            "sessionId": "s",
            "update": {
                "session_update": "agent_thought_chunk",
                "content": {"type": "text", "text": "hidden thought"},
            },
        }
        assert _extract_text_from_update(params) == ""

    def test_snake_case_message_chunk_extracted(self):
        """Fix 2: snake_case 'session_update' agent_message_chunk still works."""
        params = {
            "sessionId": "s",
            "update": {
                "session_update": "agent_message_chunk",
                "content": {"type": "text", "text": "visible"},
            },
        }
        assert _extract_text_from_update(params) == "visible"


# ---------------------------------------------------------------------------
# Tests: Fix 3 -- permission response shape
# ---------------------------------------------------------------------------


class TestPermissionResponseShape:
    def test_permission_request_uses_acp_outcome_shape(self):
        """Fix 3: permission response uses {outcome:{outcome:'selected',optionId:'...'}}
        NOT the old {granted: false} which is not a valid ACP shape."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "sess-perm"}
        session.ensure_started()

        req = {
            "id": 42,
            "method": "session/request_permission",
            "params": {
                "sessionId": "sess-perm",
                "toolCall": {"toolName": "bash"},
                "options": [
                    {"optionId": "opt-allow", "name": "Allow once", "kind": "allow_once"},
                    {"optionId": "opt-deny", "name": "Deny", "kind": "reject_once"},
                ],
            },
        }
        session._handle_server_request(req)

        mock_client.respond.assert_called_once()
        args = mock_client.respond.call_args
        assert args[0][0] == 42
        payload = args[0][1]
        # Must use the ACP spec shape, not {granted: ...}
        assert "outcome" in payload
        assert "granted" not in payload
        inner = payload["outcome"]
        assert inner["outcome"] == "selected"
        assert inner["optionId"] == "opt-allow"  # prefers allow_once kind

    def test_permission_prefers_allow_once_option(self):
        """Fix 3: the allow_once-kinded option is preferred over others."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "s"}
        session.ensure_started()

        req = {
            "id": 1,
            "method": "session/request_permission",
            "params": {
                "sessionId": "s",
                "toolCall": {"toolName": "bash"},
                "options": [
                    {"optionId": "deny-id", "name": "Deny", "kind": "reject_once"},
                    {"optionId": "allow-id", "name": "Allow", "kind": "allow_once"},
                ],
            },
        }
        session._handle_server_request(req)

        payload = mock_client.respond.call_args[0][1]
        assert payload["outcome"]["optionId"] == "allow-id"

    def test_permission_falls_back_to_first_option_when_no_allow_once(self):
        """Fix 3: if no allow_once option, echoes first available optionId."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "s"}
        session.ensure_started()

        req = {
            "id": 2,
            "method": "session/request_permission",
            "params": {
                "sessionId": "s",
                "toolCall": {"toolName": "bash"},
                "options": [
                    {"optionId": "custom-1", "name": "Custom 1", "kind": "allow_always"},
                ],
            },
        }
        session._handle_server_request(req)

        payload = mock_client.respond.call_args[0][1]
        assert payload["outcome"]["outcome"] == "selected"
        assert payload["outcome"]["optionId"] == "custom-1"

    def test_permission_cancelled_when_no_options(self):
        """Fix 3: malformed request with no options -> cancelled outcome (not wedged)."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "s"}
        session.ensure_started()

        req = {
            "id": 3,
            "method": "session/request_permission",
            "params": {"sessionId": "s", "toolCall": {"toolName": "bash"}, "options": []},
        }
        session._handle_server_request(req)

        payload = mock_client.respond.call_args[0][1]
        assert payload["outcome"] == {"outcome": "cancelled"}

    def test_pick_allow_option_helper(self):
        """_pick_allow_option() unit tests."""
        opts = [
            {"optionId": "r", "kind": "reject_once"},
            {"optionId": "a", "kind": "allow_once"},
        ]
        assert _pick_allow_option(opts) == "a"

        # no allow_once -> first
        opts2 = [{"optionId": "x", "kind": "reject_once"}]
        assert _pick_allow_option(opts2) == "x"

        # empty
        assert _pick_allow_option([]) is None


# ---------------------------------------------------------------------------
# Tests: close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_sends_session_close_and_closes_client(self):
        """close() calls session/close then client.close()."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-xyz"},  # session/new
            {},  # session/close
        ]
        session.ensure_started()
        session.close()

        # session/close request was made
        close_call = mock_client.request.call_args_list[-1]
        assert close_call[0][0] == "session/close"
        assert close_call[0][1]["sessionId"] == "sess-xyz"
        # client.close() was called
        mock_client.close.assert_called_once()

    def test_close_idempotent(self):
        """close() called twice does not raise."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "sess-x"}
        session.ensure_started()
        session.close()
        session.close()  # must not raise

    def test_context_manager_calls_close(self):
        """ACPClientSession used as context manager calls close() on exit."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [{"sessionId": "s1"}, {}]
        with session:
            session.ensure_started()
        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: run_turn -- happy path
# ---------------------------------------------------------------------------


class TestRunTurn:
    def _setup_happy_session(self):
        """Return session + mock configured for a successful prompt turn."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-happy"},  # session/new
            {"stopReason": "end_turn"},   # session/prompt
        ]
        return session, mock_client

    def test_run_turn_sends_session_prompt(self):
        """run_turn() sends session/prompt with the user text."""
        session, mock_client = self._setup_happy_session()
        # No streaming notifications
        mock_client.take_notification.side_effect = [
            None,  # polled once, returns None
            None,  # second poll -> triggers req_thread to finish
        ]

        result = session.run_turn("hello world", cwd="/tmp")
        # Check session/prompt was called
        prompt_call = None
        for c in mock_client.request.call_args_list:
            if c[0][0] == "session/prompt":
                prompt_call = c
                break
        assert prompt_call is not None
        assert prompt_call[0][1]["sessionId"] == "sess-happy"
        assert prompt_call[0][1]["prompt"][0]["text"] == "hello world"

    def test_run_turn_collects_text_from_streaming_chunks(self):
        """Text chunks from session/update notifications are assembled."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-stream"},  # session/new
        ]

        deltas_received = []

        def on_delta(text):
            deltas_received.append(text)

        session2 = ACPClientSession(
            command="fake",
            on_delta=on_delta,
            client_factory=lambda **kw: mock_client,
        )

        # Notifications: two text chunks, then None to stop
        # The session/prompt result arrives through request()
        notes_iter = iter([
            {
                "method": "session/update",
                "params": {
                    "sessionId": "sess-stream",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "Hello "},
                    },
                },
            },
            {
                "method": "session/update",
                "params": {
                    "sessionId": "sess-stream",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "world!"},
                    },
                },
            },
            None,
        ])

        def take_notif(timeout=0.0):
            try:
                return next(notes_iter)
            except StopIteration:
                return None

        mock_client.take_notification.side_effect = take_notif
        mock_client.request.return_value = {"sessionId": "sess-stream"}

        # Override request to return promptResponse after chunks
        call_count = [0]
        def req_side_effect(method, params=None, timeout=30):
            call_count[0] += 1
            if method == "session/new":
                return {"sessionId": "sess-stream"}
            if method == "session/prompt":
                # Small sleep to let notification drain happen first
                time.sleep(0.05)
                return {"stopReason": "end_turn"}
            return {}

        mock_client.request.side_effect = req_side_effect

        result = session2.run_turn("test", cwd="/tmp")
        assert "Hello " in result.final_text
        assert "world!" in result.final_text
        assert "Hello " in deltas_received
        assert "world!" in deltas_received

    def test_run_turn_projects_message_into_messages(self):
        """A final text turn is projected into projected_messages."""
        session, mock_client = _make_session()

        def req_side_effect(method, params=None, timeout=30):
            if method == "session/new":
                return {"sessionId": "sess-proj"}
            if method == "session/prompt":
                time.sleep(0.02)
                return {"stopReason": "end_turn"}
            return {}

        mock_client.request.side_effect = req_side_effect

        # Push one text chunk via notification
        notes = [
            {
                "method": "session/update",
                "params": {
                    "sessionId": "sess-proj",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "Answer here."},
                    },
                },
            },
            None,
        ]
        notes_iter = iter(notes)
        mock_client.take_notification.side_effect = lambda timeout=0.0: next(notes_iter, None)

        result = session.run_turn("question")
        assert len(result.projected_messages) == 1
        assert result.projected_messages[0]["role"] == "assistant"
        assert result.projected_messages[0]["content"] == "Answer here."


# ---------------------------------------------------------------------------
# Tests: should_retire policy
# ---------------------------------------------------------------------------


class TestShouldRetire:
    def test_subprocess_crash_sets_should_retire(self):
        """When the process exits unexpectedly, should_retire=True."""
        session, mock_client = _make_session()

        call_count = [0]
        def req_side_effect(method, params=None, timeout=30):
            call_count[0] += 1
            if method == "session/new":
                return {"sessionId": "sess-crash"}
            if method == "session/prompt":
                # Simulate blocking while process dies
                time.sleep(0.1)
                raise RuntimeError("stdin closed unexpectedly")
            return {}

        mock_client.request.side_effect = req_side_effect
        # Process dies after first poll
        alive_iter = iter([True, True, False, False])
        mock_client.is_alive.side_effect = lambda: next(alive_iter, False)

        result = session.run_turn("hello")
        assert result.should_retire is True
        assert result.error is not None

    def test_session_prompt_acp_error_sets_should_retire_for_negative_code(self):
        """ACPClientError with negative code (system error) -> should_retire."""
        session, mock_client = _make_session()

        def req_side_effect(method, params=None, timeout=30):
            if method == "session/new":
                return {"sessionId": "sess-err"}
            if method == "session/prompt":
                time.sleep(0.02)
                raise ACPClientError(code=-32603, message="internal error")
            return {}

        mock_client.request.side_effect = req_side_effect

        result = session.run_turn("hello")
        assert result.error is not None
        assert "session/prompt failed" in result.error
        assert result.should_retire is True

    def test_session_prompt_timeout_sets_should_retire(self):
        """TimeoutError from session/prompt sets should_retire."""
        session, mock_client = _make_session()

        def req_side_effect(method, params=None, timeout=30):
            if method == "session/new":
                return {"sessionId": "sess-timeout"}
            if method == "session/prompt":
                raise TimeoutError("ACP method timed out")
            return {}

        mock_client.request.side_effect = req_side_effect

        result = session.run_turn("hello")
        assert result.should_retire is True
        assert result.error is not None


# ---------------------------------------------------------------------------
# Tests: server request handling (fs/terminal decline -- unchanged behaviour)
# ---------------------------------------------------------------------------


class TestServerRequestHandling:
    def test_permission_request_grants_allow_once(self):
        """Fix 3: Permission requests use ACP outcome shape and grant allow_once."""
        session, mock_client = _make_session()
        mock_client.request.side_effect = [
            {"sessionId": "sess-perm"},  # session/new
        ]

        session.ensure_started()
        req = {
            "id": 42,
            "method": "session/request_permission",
            "params": {
                "sessionId": "sess-perm",
                "toolCall": {"toolName": "bash"},
                "options": [
                    {"optionId": "allow-once-id", "name": "Allow once", "kind": "allow_once"},
                    {"optionId": "deny-id", "name": "Deny", "kind": "reject_once"},
                ],
            },
        }
        session._handle_server_request(req)

        # Must use ACP spec outcome shape, not {granted: ...}
        mock_client.respond.assert_called_once()
        payload = mock_client.respond.call_args[0][1]
        assert "outcome" in payload
        assert "granted" not in payload
        assert payload["outcome"]["outcome"] == "selected"
        assert payload["outcome"]["optionId"] == "allow-once-id"

    def test_fs_write_declined_with_error(self):
        """fs/write_text_file is declined with respond_error."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "sess-fs"}
        session.ensure_started()

        req = {"id": 7, "method": "fs/write_text_file", "params": {"path": "/etc/passwd", "content": "bad"}}
        session._handle_server_request(req)

        mock_client.respond_error.assert_called_once()
        call_args = mock_client.respond_error.call_args
        # respond_error(rid, code=..., message=...) -- rid is positional
        assert call_args[0][0] == 7
        assert call_args[1]["code"] == -32601  # method not supported

    def test_unknown_server_request_declined_with_error(self):
        """Unknown server requests receive respond_error."""
        session, mock_client = _make_session()
        mock_client.request.return_value = {"sessionId": "sess-unk"}
        session.ensure_started()

        req = {"id": 99, "method": "some/unknown_method", "params": {}}
        session._handle_server_request(req)
        mock_client.respond_error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_text_from_text_chunk(self):
        params = {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "hello"},
            },
        }
        assert _extract_text_from_update(params) == "hello"

    def test_extract_text_from_non_text_chunk_returns_empty(self):
        params = {
            "sessionId": "s",
            "update": {
                "sessionUpdate": "tool_call_update",
                "content": {"type": "image"},
            },
        }
        assert _extract_text_from_update(params) == ""

    def test_is_tool_iteration_for_tool_call_update(self):
        params = {"update": {"sessionUpdate": "tool_call_update"}}
        assert _is_tool_iteration(params) is True

    def test_is_tool_iteration_for_agent_message_returns_false(self):
        params = {"update": {"sessionUpdate": "agent_message_chunk"}}
        assert _is_tool_iteration(params) is False

    def test_coerce_user_input_string(self):
        assert _coerce_user_input("hello") == "hello"

    def test_coerce_user_input_list_of_text_blocks(self):
        result = _coerce_user_input([{"type": "text", "text": "hello"}])
        assert result == "hello"

    def test_coerce_user_input_image_block_replaced(self):
        result = _coerce_user_input([{"type": "image"}])
        assert "[image attached]" in result

    def test_coerce_user_input_none(self):
        assert _coerce_user_input(None) == ""

    def test_coerce_user_input_integer(self):
        assert _coerce_user_input(42) == "42"


# ---------------------------------------------------------------------------
# Tests: MCP server forwarding -- translator + session/new plumbing
# ---------------------------------------------------------------------------


class TestTranslateMcpServers:
    """Unit tests for _translate_mcp_servers() covering all ACP wire shapes.

    Ground truth (empirically probed against claude-agent-acp v0.39):
      stdio  (NO type field): {name, command, args:[str], env:[{name,value}]}
      http:  {type:"http", name, url, headers:[{name,value}]}
      sse:   {type:"sse",  name, url, headers:[{name,value}]}
    env/headers must always be present as arrays ([] when empty).
    """

    def test_stdio_with_env_dict_converted_to_array(self):
        """env dict -> [{name, value}] array; no 'type' field in output."""
        result = _translate_mcp_servers({
            "myserver": {
                "command": "npx",
                "args": ["-y", "@my/mcp-server"],
                "env": {"API_KEY": "secret", "DEBUG": "1"},
            }
        })
        assert len(result) == 1
        srv = result[0]
        assert srv["name"] == "myserver"
        assert srv["command"] == "npx"
        assert srv["args"] == ["-y", "@my/mcp-server"]
        # env must be an array, not a dict
        assert isinstance(srv["env"], list)
        env_map = {e["name"]: e["value"] for e in srv["env"]}
        assert env_map == {"API_KEY": "secret", "DEBUG": "1"}
        # stdio must NOT have a "type" field
        assert "type" not in srv

    def test_stdio_with_no_env_emits_empty_array(self):
        """env absent in config -> env:[] in output (REQUIRED by ACP spec)."""
        result = _translate_mcp_servers({
            "bare": {"command": "node", "args": ["index.js"]},
        })
        assert len(result) == 1
        srv = result[0]
        assert srv["env"] == []   # must be [] not missing
        assert "type" not in srv

    def test_stdio_with_explicit_empty_env_emits_empty_array(self):
        """env:{} in config -> env:[] in output."""
        result = _translate_mcp_servers({
            "srv": {"command": "python3", "args": ["-m", "mcp"], "env": {}},
        })
        assert result[0]["env"] == []

    def test_http_with_headers_dict_converted_to_array(self):
        """url + headers dict -> type:http + headers array."""
        result = _translate_mcp_servers({
            "remote": {
                "url": "https://example.com/mcp",
                "headers": {"X-Api-Key": "abc123", "Accept": "application/json"},
            }
        })
        assert len(result) == 1
        srv = result[0]
        assert srv["type"] == "http"
        assert srv["name"] == "remote"
        assert srv["url"] == "https://example.com/mcp"
        assert isinstance(srv["headers"], list)
        hdr_map = {h["name"]: h["value"] for h in srv["headers"]}
        assert hdr_map == {"X-Api-Key": "abc123", "Accept": "application/json"}
        # http must NOT have env or command
        assert "env" not in srv
        assert "command" not in srv

    def test_http_with_no_headers_emits_empty_array(self):
        """headers absent -> headers:[] (REQUIRED by ACP spec)."""
        result = _translate_mcp_servers({
            "pub": {"url": "https://pub.example.com/mcp"},
        })
        srv = result[0]
        assert srv["type"] == "http"
        assert srv["headers"] == []

    def test_sse_transport_hint_sets_type_sse(self):
        """Hermes 'transport: sse' -> ACP type:'sse'."""
        result = _translate_mcp_servers({
            "events": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
                "headers": {},
            }
        })
        srv = result[0]
        assert srv["type"] == "sse"
        assert srv["name"] == "events"
        assert srv["url"] == "http://localhost:8000/sse"
        assert srv["headers"] == []

    def test_sse_via_type_key_also_accepted(self):
        """Hermes 'type: sse' (alternative key) -> ACP type:'sse'."""
        result = _translate_mcp_servers({
            "events2": {"url": "http://localhost:9000/sse", "type": "sse"},
        })
        assert result[0]["type"] == "sse"

    def test_malformed_no_command_no_url_skipped(self):
        """Entry with neither command nor url is skipped, not an error."""
        result = _translate_mcp_servers({
            "bad": {"timeout": 30, "auth": {"token": "x"}},
        })
        assert result == []

    def test_malformed_entry_does_not_block_valid_entries(self):
        """Malformed entry is skipped; valid sibling entries are still translated."""
        result = _translate_mcp_servers({
            "bad": {"timeout": 30},
            "good": {"command": "npx", "args": []},
        })
        assert len(result) == 1
        assert result[0]["name"] == "good"

    def test_both_command_and_url_prefers_stdio(self):
        """Both command+url set -> stdio wins (no type field)."""
        result = _translate_mcp_servers({
            "ambig": {
                "command": "my-cmd",
                "url": "https://remote.example.com",
                "env": {},
            }
        })
        srv = result[0]
        assert "type" not in srv
        assert srv["command"] == "my-cmd"

    def test_hermes_only_keys_dropped(self):
        """timeout/connect_timeout/auth/sampling are NOT forwarded to ACP."""
        result = _translate_mcp_servers({
            "srv": {
                "command": "node",
                "args": [],
                "env": {},
                "timeout": 30,
                "connect_timeout": 5,
                "auth": {"type": "oauth"},
                "sampling": True,
            }
        })
        srv = result[0]
        for dropped in ("timeout", "connect_timeout", "auth", "sampling"):
            assert dropped not in srv

    def test_empty_config_returns_empty_list(self):
        """No servers configured -> []."""
        assert _translate_mcp_servers({}) == []

    def test_none_config_returns_empty_list(self):
        """None config -> []."""
        assert _translate_mcp_servers(None) == []

    def test_multiple_servers_all_translated(self):
        """Multiple entries all appear in the output list."""
        result = _translate_mcp_servers({
            "stdio1": {"command": "cmd1", "args": []},
            "http1":  {"url": "https://a.com"},
            "sse1":   {"url": "https://b.com/sse", "transport": "sse"},
        })
        names = {s["name"] for s in result}
        assert names == {"stdio1", "http1", "sse1"}


class TestMcpServersPlumbedIntoSessionNew:
    """Assert the translated mcp_servers list reaches session/new."""

    def _make_acp_session(self, mcp_servers):
        mock_client = MagicMock()
        mock_client.is_alive.return_value = True
        mock_client.initialize.return_value = {"protocolVersion": 1}
        mock_client.take_notification.return_value = None
        mock_client.take_server_request.return_value = None
        mock_client.stderr_tail.return_value = []
        # session/new returns a sessionId
        mock_client.request.return_value = {"sessionId": "sess-mcp"}

        session = ACPClientSession(
            command="fake-acp",
            mcp_servers=mcp_servers,
            client_factory=lambda **kw: mock_client,
        )
        return session, mock_client

    def test_translated_servers_forwarded_in_session_new(self):
        """session/new receives the exact translated list, not []."""
        servers = [
            {"name": "srv1", "command": "npx", "args": [], "env": []},
            {"type": "http", "name": "srv2", "url": "https://x.com", "headers": []},
        ]
        session, mock_client = self._make_acp_session(mcp_servers=servers)
        session.ensure_started(cwd="/tmp")

        call = mock_client.request.call_args_list[0]
        assert call[0][0] == "session/new"
        params = call[0][1]
        assert params["mcpServers"] == servers

    def test_empty_mcp_servers_sends_empty_list(self):
        """None/[] -> mcpServers:[] in session/new (preserved original behavior)."""
        session, mock_client = self._make_acp_session(mcp_servers=None)
        session.ensure_started(cwd="/tmp")

        params = mock_client.request.call_args_list[0][0][1]
        assert params["mcpServers"] == []

    def test_end_to_end_translation_to_session_new(self):
        """Translator output -> ACPClientSession -> session/new mcpServers roundtrip."""
        hermes_cfg = {
            "fs-mcp": {"command": "uvx", "args": ["mcp-server-filesystem", "/data"],
                       "env": {"HOME": "/root"}},
        }
        translated = _translate_mcp_servers(hermes_cfg)
        session, mock_client = self._make_acp_session(mcp_servers=translated)
        session.ensure_started(cwd="/tmp")

        params = mock_client.request.call_args_list[0][0][1]
        mcp_list = params["mcpServers"]
        assert len(mcp_list) == 1
        srv = mcp_list[0]
        assert srv["name"] == "fs-mcp"
        assert srv["command"] == "uvx"
        assert srv["args"] == ["mcp-server-filesystem", "/data"]
        assert srv["env"] == [{"name": "HOME", "value": "/root"}]
        assert "type" not in srv  # stdio: no type field
