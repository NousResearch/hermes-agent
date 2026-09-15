"""Tests for CodexAppServerSession — drive turns through a mock client.

The session adapter has the most complex behavior of the three new modules:
notification draining, server-request handling (approvals), interrupt,
deadline timeouts. These tests pin all of that without spawning real codex.
"""

from __future__ import annotations

import time
from unittest.mock import patch
from typing import Any, Optional

import pytest

import agent.transports.codex_app_server_session as session_mod
from agent.transports.codex_app_server_session import (
    CodexAppServerSession,
    _ServerRequestRouting,
    _approval_choice_to_codex_decision,
    _coerce_turn_input_text,
)


class FakeClient:
    """Stand-in for CodexAppServerClient that records calls and lets the test
    drive the notification / server-request streams synchronously."""

    def __init__(self, *, codex_bin: str = "codex", codex_home=None) -> None:
        self.codex_bin = codex_bin
        self.codex_home = codex_home
        self.requests: list[tuple[str, dict]] = []
        self.notifications_responses: list[dict] = []
        self.responses: list[tuple[Any, dict]] = []
        self.error_responses: list[tuple[Any, int, str]] = []
        self._initialized = False
        self._closed = False
        self._notifications: list[dict] = []
        self._server_requests: list[dict] = []
        self._request_handler = None  # Optional[Callable[[str, dict], dict]]

    # API matching CodexAppServerClient
    def initialize(self, **kwargs):
        self._initialized = True
        return {"userAgent": "fake/0.0.0", "codexHome": "/tmp",
                "platformOs": "linux", "platformFamily": "unix"}

    def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0):
        self.requests.append((method, params or {}))
        if self._request_handler is not None:
            return self._request_handler(method, params or {})
        # Sensible defaults for protocol methods used by the session
        if method == "thread/start":
            return {"thread": {"id": "thread-fake-001"},
                    "activePermissionProfile": {"id": "workspace-write"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-fake-001"}}
        if method == "turn/interrupt":
            return {}
        if method == "turn/steer":
            return {"turnId": (params or {}).get("expectedTurnId")}
        return {}

    def notify(self, method: str, params=None):
        pass

    def respond(self, request_id, result):
        self.responses.append((request_id, result))

    def respond_error(self, request_id, code, message, data=None):
        self.error_responses.append((request_id, code, message))

    def take_notification(self, timeout: float = 0.0):
        if self._notifications:
            return self._notifications.pop(0)
        # Honor a tiny sleep so the loop doesn't hot-spin; the real client
        # blocks on a queue. For tests we want determinism.
        if timeout > 0:
            time.sleep(min(timeout, 0.001))
        return None

    def take_server_request(self, timeout: float = 0.0):
        if self._server_requests:
            return self._server_requests.pop(0)
        return None

    def close(self):
        self._closed = True

    def is_alive(self) -> bool:
        # Fake is "alive" until close() is called; tests that want a dead
        # subprocess can patch this attribute or call close() directly.
        return not self._closed

    def stderr_tail(self, n: int = 20):
        return list(getattr(self, "_stderr_tail", []))[-n:]

    # Test helpers
    def queue_notification(self, method: str, **params):
        # Keep legacy fixture shorthand aligned with the IDs returned by the
        # fake thread/start and turn/start responses.
        if params.get("threadId") in {"t", "th"}:
            params["threadId"] = "thread-fake-001"
        if params.get("turnId") == "tu1":
            params["turnId"] = "turn-fake-001"
        turn = params.get("turn")
        if isinstance(turn, dict) and turn.get("id") == "tu1":
            turn = dict(turn)
            turn["id"] = "turn-fake-001"
            params["turn"] = turn
        self._notifications.append({"method": method, "params": params})

    def queue_server_request(self, method: str, request_id: Any = "srv-1", **params):
        self._server_requests.append({"id": request_id, "method": method, "params": params})

    def set_stderr_tail(self, lines):
        """Test helper: seed stderr_tail() output for OAuth-refresh classifier tests."""
        self._stderr_tail = list(lines)


def make_session(client: FakeClient, **kwargs) -> CodexAppServerSession:
    return CodexAppServerSession(
        cwd="/tmp",
        client_factory=lambda **kw: client,
        **kwargs,
    )


# ---- choice mapping ----

class TestApprovalChoiceMapping:
    @pytest.mark.parametrize("choice,expected", [
        ("once", "accept"),
        ("session", "acceptForSession"),
        ("always", "acceptForSession"),
        ("deny", "decline"),
        ("anything-else", "decline"),
    ])
    def test_mapping(self, choice, expected):
        assert _approval_choice_to_codex_decision(choice) == expected


class TestTurnInputCoercion:
    def test_list_content_keeps_text_and_marks_images(self):
        text = _coerce_turn_input_text([
            {"type": "text", "text": "caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])
        assert text == "caption\n\n[image attached]"


# ---- lifecycle ----

class TestLifecycle:
    def test_ensure_started_is_idempotent(self):
        client = FakeClient()
        s = make_session(client)
        tid_a = s.ensure_started()
        tid_b = s.ensure_started()
        assert tid_a == tid_b == "thread-fake-001"
        # thread/start should be called exactly once
        method_calls = [m for (m, _) in client.requests if m == "thread/start"]
        assert len(method_calls) == 1

    def test_thread_start_passes_cwd_only(self):
        """thread/start carries cwd. We intentionally do NOT pass `permissions`
        on this codex version (experimentalApi-gated + requires matching
        config.toml [permissions] table). Letting codex use its default
        (read-only unless user configures otherwise) is the documented path."""
        client = FakeClient()
        s = make_session(client, permission_profile="workspace-write")
        s.ensure_started()
        method, params = next(r for r in client.requests if r[0] == "thread/start")
        assert params["cwd"] == "/tmp"
        assert "permissions" not in params  # see session.ensure_started() comment

    def test_close_idempotent(self):
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        s.close()
        s.close()
        assert client._closed is True


# ---- turn loop ----

class TestRunTurn:
    def test_simple_text_turn_returns_final_message(self):
        client = FakeClient()
        client.queue_notification("turn/started", threadId="t", turn={"id": "tu1"})
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "hello world"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.final_text == "hello world"
        assert r.interrupted is False
        assert r.error is None
        assert any(m["role"] == "assistant" and m.get("content") == "hello world"
                   for m in r.projected_messages)
        # turn_id propagated for downstream session-DB linkage
        assert r.turn_id == "turn-fake-001"



    def test_result_records_the_exact_submitted_input(self):
        client = FakeClient()
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        rich_input = [
            {"type": "text", "text": "caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        result = make_session(client).run_turn(rich_input, turn_timeout=2.0)
        _, params = next(request for request in client.requests if request[0] == "turn/start")
        assert result.submitted_user_text == params["input"][0]["text"]
        assert result.submitted_user_text != rich_input

    def test_foreign_completion_in_server_request_drain_is_ignored(self):
        """Approval draining must not project a child result into the parent."""
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval",
            request_id="approval-1",
            command="pwd",
            cwd="/tmp",
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-child-001",
            turnId="turn-child-001",
            item={
                "type": "agentMessage",
                "id": "child-message",
                "text": "child drain summary",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-child-001",
            turn={
                "id": "turn-child-001",
                "status": "completed",
                "error": None,
            },
        )

        original_respond = client.respond

        def respond_and_release_parent(request_id, response):
            original_respond(request_id, response)
            client.queue_notification(
                "item/completed",
                threadId="thread-fake-001",
                turnId="turn-fake-001",
                item={
                    "type": "agentMessage",
                    "id": "parent-message",
                    "text": "parent after approval",
                },
            )
            client.queue_notification(
                "turn/completed",
                threadId="thread-fake-001",
                turn={
                    "id": "turn-fake-001",
                    "status": "completed",
                    "error": None,
                },
            )

        client.respond = respond_and_release_parent
        session = make_session(
            client,
            request_routing=_ServerRequestRouting(auto_approve_exec=True),
        )

        result = session.run_turn("delegate then continue", turn_timeout=2.0)

        assert client.responses == [("approval-1", {"decision": "accept"})]
        assert result.final_text == "parent after approval"
        assert result.projected_messages == [
            {"role": "assistant", "content": "parent after approval"}
        ]



    def test_tool_iteration_counter_ticks(self):
        client = FakeClient()
        # Two completed exec items + one final agent message
        for i, item_id in enumerate(("ex1", "ex2"), start=1):
            client.queue_notification(
                "item/completed",
                item={
                    "type": "commandExecution", "id": item_id,
                    "command": f"cmd{i}", "cwd": "/tmp",
                    "status": "completed", "aggregatedOutput": "ok",
                    "exitCode": 0, "commandActions": [],
                },
                threadId="t", turnId="tu1",
            )
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "done"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("do stuff", turn_timeout=2.0)
        assert r.tool_iterations == 2
        # Each tool item produces (assistant, tool) — 2*2 + final assistant = 5 msgs
        assert len(r.projected_messages) == 5


    def test_turn_start_failure_attaches_redacted_stderr_tail(self):
        """When codex stderr has content (non-OAuth), the tail gets attached
        to the user-facing error so config/provider problems are debuggable
        instead of just 'Internal error'. Credential-shaped values in stderr
        are redacted via agent.redact(force=True); web-URL query params pass
        through (see fix(redact): pass web URLs through unchanged)."""
        client = FakeClient()
        client.set_stderr_tail([
            "ERROR: provider auth failed",
            "Authorization: Bearer sk-live-deadbeefdeadbeef",
            "url=https://api.example.com/v1?token=querysecret12345",
        ])
        from agent.transports.codex_app_server import CodexAppServerError

        def boom(method, params):
            if method == "turn/start":
                raise CodexAppServerError(code=-32603, message="Internal error")
            return {"thread": {"id": "t"}, "activePermissionProfile": {"id": "x"}}

        client._request_handler = boom
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "turn/start failed" in r.error
        assert "Internal error" in r.error
        # Stderr tail attached
        assert "codex stderr" in r.error
        assert "provider auth failed" in r.error
        # Credential-shaped values still redacted (sk- prefix + Bearer header)
        assert "sk-live-deadbeefdeadbeef" not in r.error
        # Non-OAuth → should NOT retire (subprocess JSON-RPC is still healthy).
        assert r.should_retire is False

    def test_turn_start_timeout_attaches_redacted_stderr_tail(self):
        """A non-OAuth TimeoutError on turn/start surfaces with codex stderr
        context attached and marks the session for retirement."""
        client = FakeClient()
        client.set_stderr_tail([
            "WARN: provider request stalled",
            "Authorization: Bearer sk-stalled-secret-abc123",
        ])

        def stall(method, params):
            if method == "turn/start":
                raise TimeoutError("codex method 'turn/start' timed out after 10s")
            return {"thread": {"id": "t"}, "activePermissionProfile": {"id": "x"}}

        client._request_handler = stall
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "turn/start timed out" in r.error
        assert "provider request stalled" in r.error
        assert "sk-stalled-secret-abc123" not in r.error
        assert r.should_retire is True




    def test_steer_appends_input_to_active_turn(self):
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        with s._active_turn_lock:
            s._active_turn_id = "turn-live-123"

        assert s.request_steer("Use Postgres instead") is True
        method, params = client.requests[-1]
        assert method == "turn/steer"
        assert params == {
            "threadId": "thread-fake-001",
            "input": [{"type": "text", "text": "Use Postgres instead"}],
            "expectedTurnId": "turn-live-123",
        }







class TestCompactThread:
    def test_compact_thread_sends_rpc_and_waits_for_completion(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={"type": "contextCompaction", "id": "compact-item-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={"type": "agentMessage", "id": "m1", "text": "compacted"},
        )
        client.queue_notification(
            "thread/tokenUsage/updated",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            tokenUsage={
                "last": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
                "total": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
                "modelContextWindow": 200000,
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1", "status": "completed", "error": None},
        )

        r = make_session(client).compact_thread(turn_timeout=2.0)

        assert ("thread/compact/start", {"threadId": "thread-fake-001"}) in client.requests
        assert r.error is None
        assert r.thread_id == "thread-fake-001"
        assert r.turn_id == "compact-turn-1"
        assert r.compacted is True
        assert r.final_text == "compacted"
        assert r.token_usage_last["totalTokens"] == 12
        assert r.model_context_window == 200000

    def test_compact_thread_ignores_foreign_child_completion(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-child-001",
            turn={"id": "child-compact-turn"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-child-001",
            turnId="child-compact-turn",
            item={
                "type": "agentMessage",
                "id": "child-compact-message",
                "text": "child compact summary",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-child-001",
            turn={
                "id": "child-compact-turn",
                "status": "completed",
                "error": None,
            },
        )
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={
                "type": "agentMessage",
                "id": "parent-compact-message",
                "text": "parent compacted",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={
                "id": "compact-turn-1",
                "status": "completed",
                "error": None,
            },
        )

        result = make_session(client).compact_thread(turn_timeout=2.0)

        assert result.error is None
        assert result.turn_id == "compact-turn-1"
        assert result.final_text == "parent compacted"
        assert result.projected_messages == [
            {"role": "assistant", "content": "parent compacted"}
        ]





# ---- approval bridge ----

class TestServerRequestRouting:



    def test_unknown_server_request_replied_with_error(self):
        client = FakeClient()
        client.queue_server_request("totally/unknown", request_id="req-3")
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        s.run_turn("hi", turn_timeout=1.0)
        assert any(
            rid == "req-3" and code == -32601
            for (rid, code, _msg) in client.error_responses
        )

    def test_on_event_fires_during_approval_drain(self):
        """When a server-initiated approval request arrives, the session
        drains up to 8 pending notifications first so per-turn state
        (e.g. _pending_file_changes for fileChange approvals) is current.
        Those drained notifications must also reach the on_event display
        hook — otherwise tool bubbles around approvals silently disappear.

        Regression for the issue where item/started events that landed
        in the queue alongside (or just before) an approval request got
        projected into messages but never displayed.
        """
        client = FakeClient()
        # An item/started notification is queued first, then a server
        # request — the session sees both during a single drain loop.
        client.queue_notification(
            "item/started",
            item={
                "type": "commandExecution",
                "id": "exec-1",
                "command": "echo drained",
                "cwd": "/tmp",
            },
        )
        client.queue_server_request(
            "item/commandExecution/requestApproval", request_id="req-d",
            command="echo drained",
            cwd="/tmp",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        events: list[dict] = []

        def cb(command, description, *, allow_permanent=True):
            return "once"

        s = make_session(
            client,
            approval_callback=cb,
            on_event=events.append,
        )
        s.run_turn("hi", turn_timeout=1.0)

        # The on_event hook must have seen the item/started even though
        # it was drained as part of the approval roundtrip — not just
        # events that arrive on the main notification path.
        item_started_events = [
            e for e in events
            if e.get("method") == "item/started"
        ]
        assert item_started_events, (
            "item/started drained alongside the approval was not "
            "forwarded to on_event — display will miss tool bubbles "
            "around approvals"
        )



    def test_routing_auto_approve_bypass(self):
        client = FakeClient()
        client.queue_server_request("item/commandExecution/requestApproval", request_id="r1",
                                    command="ls", cwd="/")
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        # No callback, but routing says auto-approve. Should approve.
        s = make_session(client, request_routing=_ServerRequestRouting(
            auto_approve_exec=True))
        s.run_turn("hi", turn_timeout=1.0)
        assert ("r1", {"decision": "accept"}) in client.responses



# ---- enriched approval prompts ----

class TestApprovalPromptEnrichment:
    """Quirk #4: apply_patch prompt should show what's changing.
    Quirk #10: exec prompt should never show empty cwd."""

    def test_exec_falls_back_to_session_cwd(self):
        """When codex omits cwd from the approval params, the prompt shows
        the session cwd, not an empty string."""
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval", request_id="r1",
            command="ls",  # no cwd
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["description"] = description
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Session cwd is /tmp by default in make_session()
        assert "/tmp" in captured["description"]
        assert "Codex requests exec in <unknown>" not in captured["description"]

    def test_apply_patch_prompt_summarizes_pending_changes(self):
        """When the projector has cached the fileChange item from item/started,
        the approval prompt surfaces the change summary."""
        client = FakeClient()
        # item/started fires first (carries the changes), then approval request
        client.queue_notification(
            "item/started",
            item={"type": "fileChange", "id": "fc-1",
                  "changes": [
                      {"kind": {"type": "add"}, "path": "/tmp/new.py"},
                      {"kind": {"type": "update"}, "path": "/tmp/old.py"},
                  ]},
            threadId="t", turnId="tu1",
        )
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="req-2",
            itemId="fc-1", turnId="tu1", threadId="t",
            startedAtMs=1234567890,
            reason="add and update files",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["command"] = command
            captured["description"] = description
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Both add and update kinds should be in the summary
        assert "1 add" in captured["command"] or "1 add" in captured["description"]
        assert "1 update" in captured["command"] or "1 update" in captured["description"]
        # And at least one of the paths
        joined = captured["command"] + " " + captured["description"]
        assert "/tmp/new.py" in joined or "/tmp/old.py" in joined

    def test_apply_patch_prompt_works_without_cached_summary(self):
        """When approval arrives before item/started (or without changes
        info), prompt falls back to whatever codex provided."""
        client = FakeClient()
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="req-2",
            itemId="fc-orphan", turnId="tu1", threadId="t",
            startedAtMs=1234567890,
            reason="apply some changes",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["command"] = command
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Falls back to the reason
        assert "apply some changes" in captured["command"]


# ---- openclaw beta.8 parity: retire/wedge/oauth/abort marker ----

class TestSessionRetirement:
    """Mirrors openclaw beta.8's resilience fixes:
      - retire timed-out app-server clients (should_retire on deadline)
      - post-tool completion watchdog (don't burn the full deadline after a
        tool result if codex goes silent)
      - <turn_aborted> raw marker as terminal (don't wait for turn/completed
        that never comes)
      - OAuth refresh failure classification (suggest `codex login` instead
        of raw RPC error strings)
      - dead subprocess detection between iterations
    """



    def test_final_agent_message_without_turn_completed_is_recovered(self):
        """A completed assistant item is still a usable terminal response when
        codex omits turn/completed and then goes quiet.
        """
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "done"},
            threadId="t",
            turnId="tu1",
        )
        s = make_session(client)
        r = s.run_turn(
            "hi",
            turn_timeout=0.05,
            notification_poll_timeout=0.01,
        )
        assert r.final_text == "done"
        assert r.interrupted is False
        assert r.error is None
        assert r.should_retire is False
        assert any(
            msg["role"] == "assistant" and msg.get("content") == "done"
            for msg in r.projected_messages
        )
        assert not any(method == "turn/interrupt" for method, _ in client.requests)


    def test_post_tool_watchdog_uses_monotonic_clock(self):
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "commandExecution", "id": "ex1",
                "command": "echo hi", "cwd": "/tmp",
                "status": "completed", "aggregatedOutput": "hi",
                "exitCode": 0, "commandActions": [],
            },
            threadId="t", turnId="tu1",
        )
        s = make_session(client)
        monotonic_values = iter([1000.0, 999.0, 999.0, 999.0, 1000.2])
        with patch.object(
            session_mod.time,
            "monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            r = s.run_turn(
                "tool then silence",
                turn_timeout=5.0,
                notification_poll_timeout=0.0,
                post_tool_quiet_timeout=0.15,
            )
        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "silent" in r.error

    def test_post_tool_watchdog_resets_on_further_activity(self):
        """A tool completion followed by an agent message should NOT trip
        the watchdog — further activity = codex still alive."""
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "commandExecution", "id": "ex1",
                "command": "echo hi", "cwd": "/tmp",
                "status": "completed", "aggregatedOutput": "hi",
                "exitCode": 0, "commandActions": [],
            },
            threadId="t", turnId="tu1",
        )
        # Non-tool activity immediately after — resets watchdog.
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "tool finished"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn(
            "tool then talk", turn_timeout=2.0,
            notification_poll_timeout=0.01,
            post_tool_quiet_timeout=0.05,
        )
        # Tool ran, then text reset the watchdog, then turn/completed.
        # Should NOT be a retirement case.
        assert r.tool_iterations == 1
        assert r.final_text == "tool finished"
        assert r.should_retire is False
        assert r.interrupted is False







    def test_dead_subprocess_detected_between_iterations(self):
        """If codex dies (segfault, OOM, killed by its auth refresh
        thread), the inter-iteration is_alive check breaks the loop
        instead of waiting on a queue that will never fill."""
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        # Simulate subprocess death by setting _closed (FakeClient's
        # is_alive returns False when closed).
        client._closed = True
        client.set_stderr_tail([
            "thread 'tokio-runtime-worker' panicked at 'oauth: invalid_grant'",
        ])
        r = s.run_turn("x", turn_timeout=2.0,
                       notification_poll_timeout=0.01)
        assert r.should_retire is True
        # Stderr-derived auth hint takes precedence over generic message
        assert r.error and "codex login" in r.error


# ---- thread/start cross-fill ----

class TestThreadStartCrossFill:
    """Mirrors openclaw beta.8's tolerance for thread.id/sessionId aliasing."""

    def test_thread_id_under_thread_key(self):
        client = FakeClient()
        s = make_session(client)
        tid = s.ensure_started()
        assert tid == "thread-fake-001"



    def test_missing_thread_id_raises(self):
        from agent.transports.codex_app_server import CodexAppServerError

        client = FakeClient()
        client._request_handler = lambda method, params: (
            {"thread": {}, "activePermissionProfile": {"id": "x"}}
            if method == "thread/start" else
            {"turn": {"id": "tu1"}}
        )
        s = make_session(client)
        with pytest.raises(CodexAppServerError, match="no thread id"):
            s.ensure_started()


class TestHasTurnAbortedMarker:
    """Unit coverage for the marker matcher itself."""

    def test_empty_string(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("") is False
        assert _has_turn_aborted_marker(None) is False  # type: ignore[arg-type]

    def test_plain_text_no_marker(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("normal response with no markers") is False

    def test_open_marker(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("blah <turn_aborted> blah") is True



class TestClassifyOAuthFailure:
    """Unit coverage for the OAuth classifier; conservative on purpose."""



    def test_401_classified(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        hint = _classify_oauth_failure("HTTP 401 Unauthorized")
        assert hint is not None


    def test_empty_inputs(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        assert _classify_oauth_failure() is None
        assert _classify_oauth_failure("") is None
        assert _classify_oauth_failure("", None) is None  # type: ignore[arg-type]


# ---- documented approval lifecycle hooks on the codex transport ----


def _record_approval_hooks(monkeypatch) -> list[tuple[str, dict]]:
    """Capture the approval lifecycle events a plugin would receive.

    ``tools.approval_context._fire_approval_hook`` resolves ``invoke_hook`` at call time, so
    patching the lifecycle module observes exactly what a registered plugin gets.
    """
    events: list[tuple[str, dict]] = []

    def fake_invoke_hook(name: str, **kwargs):
        events.append((name, kwargs))
        return None

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", fake_invoke_hook)
    return events


class TestApprovalLifecycleHooks:
    """hooks.md documents ``pre_approval_request`` / ``post_approval_response`` as the observer pair
    for an approval a human is asked about. Codex approvals never reach ``tools/approval*.py`` — the
    transport asks its callback directly — so the pair has to be dispatched at the transport's
    single approval seam, with the same payload shape and redaction as the other surfaces.
    """

    def _session_with_exec_request(self, client, callback, **session_kwargs):
        client.queue_server_request(
            "item/commandExecution/requestApproval", request_id="r1",
            command="rm -rf /tmp/build", cwd="/tmp", reason="needs to clean the build dir",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        return make_session(client, approval_callback=callback, **session_kwargs)

    def test_exec_approval_dispatches_pre_hook_once(self, monkeypatch):
        events = _record_approval_hooks(monkeypatch)
        seen = {}

        def cb(command, description, *, allow_permanent=True):
            seen["command"] = command
            return "once"

        session = self._session_with_exec_request(FakeClient(), cb)
        session.run_turn("hi", turn_timeout=1.0)

        pre = [kw for name, kw in events if name == "pre_approval_request"]
        assert len(pre) == 1, f"expected exactly one pre_approval_request, recorded: {events}"
        assert pre[0]["command"] == "rm -rf /tmp/build"
        assert pre[0]["pattern_key"] == "codex_runtime"
        assert pre[0]["pattern_keys"] == ["codex_runtime"]
        assert pre[0]["surface"] == "cli"
        assert pre[0]["session_key"]  # observers scope their marks off this
        assert seen["command"] == "rm -rf /tmp/build"

    def test_exec_approval_dispatches_post_with_the_user_choice(self, monkeypatch):
        events = _record_approval_hooks(monkeypatch)
        session = self._session_with_exec_request(
            FakeClient(), lambda command, description, **kw: "once")
        session.run_turn("hi", turn_timeout=1.0)

        assert [name for name, _ in events] == ["pre_approval_request", "post_approval_response"]
        assert events[1][1]["choice"] == "once"
        assert events[1][1]["command"] == "rm -rf /tmp/build"

    def test_apply_patch_approval_dispatches_the_same_pair(self, monkeypatch):
        events = _record_approval_hooks(monkeypatch)
        client = FakeClient()
        client.queue_notification(
            "item/started",
            item={"type": "fileChange", "id": "fc-1",
                  "changes": [{"kind": {"type": "update"}, "path": "/tmp/a.py"}]},
            threadId="t", turnId="tu1",
        )
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="r2", itemId="fc-1",
            threadId="t", turnId="tu1", startedAtMs=1234567890, reason="update a.py",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        session = make_session(client, approval_callback=lambda command, description, **kw: "session")
        session.run_turn("hi", turn_timeout=1.0)

        assert [name for name, _ in events] == ["pre_approval_request", "post_approval_response"]
        assert "apply_patch" in events[0][1]["command"]
        assert events[1][1]["choice"] == "session"

    @pytest.mark.parametrize("choice", ["once", "session", "always", "deny", "timeout"])
    def test_one_request_is_exactly_one_pair_whatever_the_choice(self, monkeypatch, choice):
        """A scope decision or a timeout is the same request's outcome, not a second event."""
        events = _record_approval_hooks(monkeypatch)
        session = self._session_with_exec_request(
            FakeClient(), lambda command, description, **kw: choice)
        session.run_turn("hi", turn_timeout=1.0)

        assert [name for name, _ in events] == ["pre_approval_request", "post_approval_response"]
        assert events[1][1]["choice"] == choice

    def test_hook_return_value_cannot_veto_the_decision(self, monkeypatch):
        """Observers are documented return-ignored (plugins.py); a plugin answering "deny" must not
        change what the user's own approval produced."""
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kwargs: "deny")
        client = FakeClient()
        session = self._session_with_exec_request(
            client, lambda command, description, **kw: "once")
        session.run_turn("hi", turn_timeout=1.0)

        assert client.responses == [("r1", {"decision": "accept"})]

    def test_hook_exception_cannot_break_the_approval(self, monkeypatch):
        def exploding_invoke_hook(name, **kwargs):
            raise RuntimeError("observer blew up")

        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", exploding_invoke_hook)
        client = FakeClient()
        session = self._session_with_exec_request(
            client, lambda command, description, **kw: "once")
        session.run_turn("hi", turn_timeout=1.0)

        assert client.responses == [("r1", {"decision": "accept"})]

    def test_hook_payload_is_redacted_while_the_approval_sees_the_real_command(self, monkeypatch):
        """hooks.md: an approval command may carry secrets. The observer payload must never carry
        the raw text; the approval decision itself still sees what will run."""
        secret = "sk-live-51H8xQ2eZvKYlo2Cabcdefgh"
        events = _record_approval_hooks(monkeypatch)
        seen = {}
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval", request_id="r1",
            command=f"curl -H 'Authorization: Bearer {secret}' https://api.example.com",
            cwd="/tmp",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        def cb(command, description, **kwargs):
            seen["command"] = command
            return "once"

        session = make_session(client, approval_callback=cb)
        session.run_turn("hi", turn_timeout=1.0)

        assert secret in seen["command"]
        pre = [kw for name, kw in events if name == "pre_approval_request"][0]
        assert secret not in pre["command"], "the observer payload leaked the raw secret"
        assert "***" in pre["command"]

    def test_callback_exception_reports_no_decision_not_a_denial(self, monkeypatch):
        """A raising callback obtained no approval response: the post event must carry the documented
        no-decision failure, never ``deny`` — an exception is not a human denial."""
        events = _record_approval_hooks(monkeypatch)

        def exploding_callback(command, description, **kwargs):
            raise RuntimeError("prompt UI blew up")

        client = FakeClient()
        session = self._session_with_exec_request(client, exploding_callback)
        session.run_turn("hi", turn_timeout=1.0)

        assert client.responses == [("r1", {"decision": "decline"})]
        assert [name for name, _ in events] == ["pre_approval_request", "post_approval_response"]
        post = events[1][1]
        assert post["choice"] == "notify_failed"
        assert post["choice"] not in {"once", "session", "always", "deny", "timeout"}

    def test_no_hooks_when_no_human_is_asked(self, monkeypatch):
        """Parity with the native gates: a bypassed approval and the fail-closed no-callback path
        prompt nobody, so they must not emit lifecycle events."""
        events = _record_approval_hooks(monkeypatch)

        bypassed = FakeClient()
        bypass_session = self._session_with_exec_request(
            bypassed, None,
            request_routing=_ServerRequestRouting(auto_approve_exec=True,
                                                  auto_approve_apply_patch=True))
        bypass_session.run_turn("hi", turn_timeout=1.0)
        assert bypassed.responses == [("r1", {"decision": "accept"})]
        assert events == []

        headless = FakeClient()
        headless_session = self._session_with_exec_request(headless, None)
        headless_session.run_turn("hi", turn_timeout=1.0)
        assert headless.responses == [("r1", {"decision": "decline"})]
        assert events == []

