"""Tests for agent.claude_code_tmux_bridge.

All tmux subprocess calls are mocked; no real tmux session is required.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import agent.claude_code_tmux_bridge as bridge
from agent.claude_code_tmux_bridge import (
    BridgeResult,
    BridgeStatus,
    WorkerConfig,
    _extract_response,
    _is_auth_failure,
    _redact_credentials,
    _strip_ansi,
)

FAKE_RUN_ID = "cafebabe" * 4  # 32 hex chars


@pytest.fixture(autouse=True)
def _clear_session_locks():
    bridge._session_locks.clear()
    yield
    bridge._session_locks.clear()


# ── Pure helper unit tests ────────────────────────────────────────────────────

class TestStripAnsi:
    def test_removes_sgr_codes(self):
        assert _strip_ansi("\x1b[32mgreen\x1b[0m") == "green"

    def test_removes_osc_sequences(self):
        assert _strip_ansi("\x1b]0;title\x07text") == "text"

    def test_removes_carriage_return(self):
        assert _strip_ansi("a\rb") == "ab"

    def test_passthrough_plain_text(self):
        assert _strip_ansi("hello world") == "hello world"

    def test_empty_string(self):
        assert _strip_ansi("") == ""


class TestRedactCredentials:
    def test_redacts_bearer_token(self):
        text = "Authorization: Bearer sk-abcdef1234567890abcdef1234567890"
        out = _redact_credentials(text)
        assert "[REDACTED]" in out
        assert "sk-abcdef" not in out

    def test_redacts_api_key_value(self):
        text = "api_key=supersecretlongvalue12345"
        out = _redact_credentials(text)
        assert "[REDACTED]" in out

    def test_leaves_short_values_alone(self):
        # Values shorter than 16 chars are not credential-shaped
        assert _redact_credentials("token: abc") == "token: abc"

    def test_leaves_unrelated_text_alone(self):
        text = "The answer is 42."
        assert _redact_credentials(text) == text


class TestIsAuthFailure:
    @pytest.mark.parametrize("text", [
        "Authentication required",
        "OAuth token expired",
        "Please sign in to continue",
        "Unauthorized",
        "Session expired",
        "Login required",
        "credentials expired",
        "Please authenticate",
    ])
    def test_detects_known_patterns(self, text):
        assert _is_auth_failure(text)

    def test_ignores_plain_text(self):
        assert not _is_auth_failure("The function returns a boolean value.")

    def test_strips_ansi_before_matching(self):
        assert _is_auth_failure("\x1b[31mAuthentication required\x1b[0m")

    def test_case_insensitive(self):
        assert _is_auth_failure("AUTHENTICATION REQUIRED")


class TestExtractResponse:
    def test_extracts_lines_after_pre_count_up_to_marker(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = "\n".join([
            "old line 1",
            "old line 2",
            "The answer is 42.",
            done,
            "stuff after",
        ])
        result = _extract_response(pre_line_count=2, post_raw=pane, done_marker=done)
        assert result is not None
        assert "The answer is 42." in result
        assert "old line" not in result
        assert done not in result

    def test_returns_none_when_marker_absent(self):
        assert _extract_response(0, "line 1\nline 2", "NO_MARKER") is None

    def test_strips_ansi_from_extracted_text(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = f"preamble\n\x1b[32mgreen text\x1b[0m\n{done}"
        result = _extract_response(1, pane, done)
        assert result is not None
        assert "\x1b" not in result
        assert "green text" in result

    def test_returns_empty_string_when_nothing_between(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = f"pre\n{done}"
        result = _extract_response(1, pane, done)
        assert result == ""

    def test_redacts_credential_in_response(self):
        done = f"HERMES_BRIDGE_DONE_{FAKE_RUN_ID}"
        pane = f"pre\nBearer sk-supersecretlongtokenvalue1234567\n{done}"
        result = _extract_response(1, pane, done)
        assert result is not None
        assert "[REDACTED]" in result
        assert "sk-supersecret" not in result


# ── Integration tests with mocked tmux helpers ────────────────────────────────

def _setup_session(monkeypatch, *, exists: bool = True, alive: bool = True) -> None:
    monkeypatch.setattr(bridge, "_session_exists", lambda s: exists)
    monkeypatch.setattr(bridge, "_pane_alive", lambda s: alive)


def _setup_io(
    monkeypatch,
    *,
    pre_pane: str = "",
    post_pane: str = "",
    send_ok: bool = True,
) -> None:
    """Mock _capture_pane to return pre_pane on first call, post_pane thereafter."""
    call_count = [0]

    def _mock_capture(s, **kw):
        call_count[0] += 1
        return pre_pane if call_count[0] == 1 else post_pane

    monkeypatch.setattr(bridge, "_capture_pane", _mock_capture)
    monkeypatch.setattr(bridge, "_send_keys", lambda s, t: send_ok)


class TestSessionVerification:
    def test_missing_session_returns_session_missing(self, monkeypatch):
        _setup_session(monkeypatch, exists=False)
        result = bridge.submit_prompt("claude-momentum", "hello")
        assert result.status == BridgeStatus.SESSION_MISSING
        assert "claude-momentum" in result.error

    def test_dead_session_returns_session_dead(self, monkeypatch):
        _setup_session(monkeypatch, exists=True, alive=False)
        result = bridge.submit_prompt("claude-momentum", "hello")
        assert result.status == BridgeStatus.SESSION_DEAD

    def test_tmux_not_found_returns_failure(self, monkeypatch):
        def _raise(s):
            raise FileNotFoundError("tmux")
        monkeypatch.setattr(bridge, "_session_exists", _raise)
        result = bridge.submit_prompt("claude-momentum", "hello")
        assert result.status == BridgeStatus.FAILURE
        assert "not found" in result.error.lower()

    def test_os_error_returns_failure(self, monkeypatch):
        def _raise(s):
            raise OSError("permission denied")
        monkeypatch.setattr(bridge, "_session_exists", _raise)
        result = bridge.submit_prompt("claude-momentum", "hello")
        assert result.status == BridgeStatus.FAILURE

    def test_unknown_session_uses_default_config(self, monkeypatch):
        """Ad-hoc sessions not in _WORKERS are accepted with defaults."""
        _setup_session(monkeypatch, exists=False)
        result = bridge.submit_prompt("some-other-session", "hello")
        assert result.status == BridgeStatus.SESSION_MISSING


class TestLocking:
    def test_busy_when_lock_already_held(self, monkeypatch):
        _setup_session(monkeypatch)
        lock = bridge._get_session_lock("claude-momentum")
        lock.acquire()
        try:
            result = bridge.submit_prompt("claude-momentum", "hello")
            assert result.status == BridgeStatus.BUSY
            assert "claude-momentum" in result.error
        finally:
            lock.release()

    def test_lock_released_after_successful_call(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = "history\n"
        post = f"history\nresponse line\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "hi", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS

        # Lock must be free after the call
        lock = bridge._get_session_lock("claude-momentum")
        assert lock.acquire(blocking=False)
        lock.release()

    def test_lock_released_after_send_failure(self, monkeypatch):
        _setup_session(monkeypatch)
        monkeypatch.setattr(bridge, "_capture_pane", lambda s, **kw: "")
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: False)

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=5.0)
        assert result.status == BridgeStatus.FAILURE

        lock = bridge._get_session_lock("claude-momentum")
        assert lock.acquire(blocking=False)
        lock.release()

    def test_separate_sessions_use_separate_locks(self, monkeypatch):
        lock_a = bridge._get_session_lock("session-a")
        lock_b = bridge._get_session_lock("session-b")
        assert lock_a is not lock_b

        lock_a.acquire()
        try:
            # session-b lock is independent
            assert lock_b.acquire(blocking=False)
            lock_b.release()
        finally:
            lock_a.release()


class TestSuccessPath:
    def test_returns_success_with_response(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = "prior history\n"
        post = f"prior history\nThe answer is 42.\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "What is 6×7?", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS
        assert "The answer is 42." in result.response

    def test_excludes_prior_pane_history(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        pre = "old output line 1\nold output line 2\n"
        post = f"old output line 1\nold output line 2\nnew response\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS
        assert "old output" not in result.response
        assert "new response" in result.response

    def test_ansi_codes_stripped_from_response(self, monkeypatch):
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        post = f"\x1b[32mColoured output\x1b[0m\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane="", post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "color?", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS
        assert "\x1b" not in result.response
        assert "Coloured output" in result.response


class TestTimeoutAndAuthFailure:
    def _fast_monotonic(self):
        ticks = [0.0]
        def _advance():
            ticks[0] += 10.0
            return ticks[0]
        return _advance

    def test_returns_timeout_when_marker_never_appears(self, monkeypatch):
        _setup_session(monkeypatch)
        monkeypatch.setattr(bridge, "_capture_pane", lambda s, **kw: "no marker here")
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: True)
        monkeypatch.setattr(bridge.time, "monotonic", self._fast_monotonic())

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=5.0)
        assert result.status == BridgeStatus.TIMEOUT
        assert "5s" in result.error

    def test_returns_auth_failure_when_detected_at_timeout(self, monkeypatch):
        _setup_session(monkeypatch)
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: True)
        monkeypatch.setattr(bridge.time, "monotonic", self._fast_monotonic())

        call_count = [0]
        def _mock_capture(s, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return ""  # pre-send snapshot
            return "Claude: Authentication required. Please sign in."

        monkeypatch.setattr(bridge, "_capture_pane", _mock_capture)

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=5.0)
        assert result.status == BridgeStatus.AUTH_FAILURE

    def test_no_false_auth_failure_on_success(self, monkeypatch):
        """Old pane history with auth text must not prevent SUCCESS."""
        run_id = FAKE_RUN_ID
        done = f"HERMES_BRIDGE_DONE_{run_id}"
        # Pre-existing history contains an old auth message
        pre = "Authentication required (old history)\n"
        post = f"Authentication required (old history)\nNew response\n{done}\n"

        monkeypatch.setattr(bridge.uuid, "uuid4", lambda: MagicMock(hex=run_id))
        _setup_session(monkeypatch)
        _setup_io(monkeypatch, pre_pane=pre, post_pane=post)
        monkeypatch.setattr(bridge.time, "sleep", lambda x: None)

        result = bridge.submit_prompt("claude-momentum", "hi", timeout=30.0)
        assert result.status == BridgeStatus.SUCCESS


class TestSendFailure:
    def test_failure_when_send_keys_returns_false(self, monkeypatch):
        _setup_session(monkeypatch)
        monkeypatch.setattr(bridge, "_capture_pane", lambda s, **kw: "")
        monkeypatch.setattr(bridge, "_send_keys", lambda s, t: False)

        result = bridge.submit_prompt("claude-momentum", "hello", timeout=5.0)
        assert result.status == BridgeStatus.FAILURE
        assert "send-keys" in result.error


class TestWorkerRegistry:
    def test_default_worker_registered(self):
        w = bridge.get_worker("claude-momentum")
        assert w is not None
        assert w.workspace == "/home/michael/code"

    def test_register_worker_roundtrip(self):
        cfg = WorkerConfig(session="test-session", workspace="/tmp/test", timeout=60.0)
        bridge.register_worker(cfg)
        assert bridge.get_worker("test-session") is cfg

    def test_get_worker_returns_none_for_unknown(self):
        assert bridge.get_worker("nonexistent-xyz") is None
