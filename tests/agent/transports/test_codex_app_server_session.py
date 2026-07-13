"""Tests for CodexAppServerSession — drive turns through a mock client.

The session adapter has the most complex behavior of the three new modules:
notification draining, server-request handling (approvals), interrupt,
deadline timeouts. These tests pin all of that without spawning real codex.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Optional
from unittest.mock import patch

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

    def test_thread_start_bridges_developer_context_dynamic_tools_and_history(self):
        client = FakeClient()
        dynamic_tools = [{
            "type": "namespace",
            "name": "hermes",
            "description": "Hermes live tools",
            "tools": [],
        }]
        history = [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "earlier answer"}],
        }]
        s = make_session(
            client,
            developer_instructions="HERMES SOUL AND MEMORY",
            dynamic_tools=dynamic_tools,
            initial_history_items=history,
        )

        s.ensure_started()

        assert client._initialized is True
        method, params = next(r for r in client.requests if r[0] == "thread/start")
        assert method == "thread/start"
        assert params["developerInstructions"] == "HERMES SOUL AND MEMORY"
        assert params["dynamicTools"] == dynamic_tools
        inject = next(r for r in client.requests if r[0] == "thread/inject_items")
        assert inject[1] == {"threadId": "thread-fake-001", "items": history}

    @pytest.mark.parametrize("invalid_thread_id", ["   ", 123, {}])
    def test_thread_start_rejects_malformed_thread_identity(self, invalid_thread_id):
        client = FakeClient()

        def request_handler(method, params):
            if method == "thread/start":
                return {"thread": {"id": invalid_thread_id}}
            return {}

        client._request_handler = request_handler
        s = make_session(client)

        with pytest.raises(session_mod.CodexAppServerError, match="thread id"):
            s.ensure_started()

    def test_dynamic_tool_server_request_is_dispatched_to_hermes(self):
        client = FakeClient()
        seen = []

        def handler(params):
            seen.append(params)
            return {
                "success": True,
                "contentItems": [{"type": "inputText", "text": "remembered"}],
            }

        client.queue_server_request(
            "item/tool/call",
            request_id="dyn-1",
            threadId="thread-fake-001",
            turnId="turn-fake-001",
            callId="call-1",
            namespace="hermes",
            tool="session_search",
            arguments={"query": "old decision"},
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )

        s = make_session(client, dynamic_tool_handler=handler)
        s.run_turn("remember", turn_timeout=1.0)

        assert seen == [{
            "threadId": "thread-fake-001",
            "turnId": "turn-fake-001",
            "callId": "call-1",
            "namespace": "hermes",
            "tool": "session_search",
            "arguments": {"query": "old decision"},
        }]
        assert client.responses == [("dyn-1", {
            "success": True,
            "contentItems": [{"type": "inputText", "text": "remembered"}],
        })]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"callId": ""},
            {"callId": "   "},
            {"callId": None},
            {"threadId": "wrong-thread"},
            {"threadId": None},
            {"turnId": "wrong-turn"},
            {"turnId": None},
        ],
    )
    def test_dynamic_tool_server_request_rejects_unbound_identity(self, overrides):
        client = FakeClient()
        seen = []
        params = {
            "threadId": "thread-fake-001",
            "turnId": "turn-fake-001",
            "callId": "call-1",
            "namespace": "hermes",
            "tool": "todo",
            "arguments": {},
        }
        params.update(overrides)
        client.queue_server_request("item/tool/call", request_id="dyn-1", **params)
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )

        s = make_session(client, dynamic_tool_handler=lambda payload: seen.append(payload))
        s.run_turn("remember", turn_timeout=1.0)

        assert seen == []
        assert client.responses[0][0] == "dyn-1"
        assert client.responses[0][1]["success"] is False

    @pytest.mark.parametrize(
        "notification_thread_id",
        ["thread-fake-001", "foreign-thread"],
    )
    def test_stale_or_foreign_turn_started_cannot_rebind_dynamic_dispatch(
        self, notification_thread_id
    ):
        client = FakeClient()
        seen = []
        client.queue_notification(
            "turn/started",
            threadId=notification_thread_id,
            turn={"id": "stale-turn"},
        )
        client.queue_server_request(
            "item/tool/call",
            request_id="dyn-stale",
            threadId="thread-fake-001",
            turnId="stale-turn",
            callId="call-stale",
            namespace="hermes",
            tool="todo",
            arguments={},
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )

        s = make_session(client, dynamic_tool_handler=lambda payload: seen.append(payload))
        s.run_turn("remember", turn_timeout=1.0)

        assert seen == []
        assert client.responses == [("dyn-stale", {
            "success": False,
            "contentItems": [{
                "type": "inputText",
                "text": "Hermes dynamic tool request was rejected",
            }],
        })]

    def test_turn_started_cannot_activate_tools_without_turn_start_response_id(self):
        client = FakeClient()
        seen = []

        def request_handler(method, params):
            if method == "thread/start":
                return {"thread": {"id": "thread-fake-001"}}
            if method == "turn/start":
                return {"turn": {}}
            return {}

        client._request_handler = request_handler
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "notification-only-turn"},
        )
        client.queue_server_request(
            "item/tool/call",
            request_id="dyn-unbound",
            threadId="thread-fake-001",
            turnId="notification-only-turn",
            callId="call-unbound",
            namespace="hermes",
            tool="todo",
            arguments={},
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "notification-only-turn", "status": "completed", "error": None},
        )

        s = make_session(client, dynamic_tool_handler=lambda payload: seen.append(payload))
        s.run_turn("remember", turn_timeout=0.05)

        assert seen == []
        assert client.responses[0][0] == "dyn-unbound"
        assert client.responses[0][1]["success"] is False

    def test_duplicate_dynamic_call_id_is_replayed_without_redispatch(self):
        client = FakeClient()
        seen = []
        params = {
            "threadId": "thread-fake-001",
            "turnId": "turn-fake-001",
            "callId": "call-1",
            "namespace": "hermes",
            "tool": "todo",
            "arguments": {},
        }
        client.queue_server_request("item/tool/call", request_id="dyn-1", **params)
        client.queue_server_request("item/tool/call", request_id="dyn-2", **params)
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )

        def handler(payload):
            seen.append(payload)
            return {
                "success": True,
                "contentItems": [{"type": "inputText", "text": "once"}],
            }

        s = make_session(client, dynamic_tool_handler=handler)
        s.run_turn("remember", turn_timeout=1.0)

        assert len(seen) == 1
        assert client.responses == [
            ("dyn-1", {"success": True, "contentItems": [{"type": "inputText", "text": "once"}]}),
            ("dyn-2", {"success": True, "contentItems": [{"type": "inputText", "text": "once"}]}),
        ]

    def test_close_idempotent(self):
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        s.close()
        s.close()
        assert client._closed is True


# ---- turn loop ----

class TestRunTurn:
    @pytest.mark.parametrize(
        "notification_thread_id,notification_turn_id",
        [
            ("foreign-thread", "turn-fake-001"),
            ("thread-fake-001", "foreign-turn"),
            ("", "turn-fake-001"),
            ("thread-fake-001", ""),
            (None, "turn-fake-001"),
            ("thread-fake-001", None),
        ],
    )
    def test_foreign_or_malformed_turn_completed_does_not_end_active_turn(
        self, notification_thread_id, notification_turn_id
    ):
        client = FakeClient()
        client.queue_notification(
            "turn/completed",
            threadId=notification_thread_id,
            turn={
                "id": notification_turn_id,
                "status": "completed",
                "error": None,
            },
        )
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "real result"},
            threadId="thread-fake-001",
            turnId="turn-fake-001",
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )

        result = make_session(client).run_turn("hi", turn_timeout=1.0)

        assert result.final_text == "real result"
        assert result.turn_id == "turn-fake-001"
        assert result.interrupted is False
        assert result.error is None

    def test_simple_text_turn_returns_final_message(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001"},
        )
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "hello world"},
            threadId="thread-fake-001", turnId="turn-fake-001",
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.final_text == "hello world"
        assert r.interrupted is False
        assert r.error is None
        assert any(m["role"] == "assistant" and m.get("content") == "hello world"
                   for m in r.projected_messages)
        # turn_id remains bound to the authoritative turn/start response.
        assert r.turn_id == "turn-fake-001"

    def test_extra_skill_roots_are_set_and_force_reloaded_before_turn_start(self):
        client = FakeClient()
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        s = make_session(client, extra_skill_roots=["/tmp/hermes-skills"])

        s.run_turn("hi", turn_timeout=2.0)

        methods = [method for method, _ in client.requests]
        assert methods.index("thread/start") < methods.index("skills/extraRoots/set")
        assert methods.index("skills/extraRoots/set") < methods.index("skills/list")
        assert methods.index("skills/list") < methods.index("turn/start")
        assert ("skills/extraRoots/set", {"extraRoots": ["/tmp/hermes-skills"]}) in client.requests
        assert (
            "skills/list",
            {"cwds": ["/tmp"], "forceReload": True},
        ) in client.requests

    def test_changed_hermes_skill_is_attached_to_the_next_turn(self, tmp_path):
        skill_path = tmp_path / "skills" / "testing" / "bridge" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("initial", encoding="utf-8")

        client = FakeClient()
        original_request = client.request

        def request(method, params=None, timeout=10.0):
            if method == "skills/list":
                return {
                    "data": [
                        {
                            "cwd": "/tmp",
                            "errors": [],
                            "skills": [
                                {
                                    "name": "bridge",
                                    "path": str(skill_path),
                                    "enabled": True,
                                }
                            ],
                        }
                    ]
                }
            return original_request(method, params, timeout)

        client.request = request
        s = make_session(client, extra_skill_roots=[str(tmp_path / "skills")])

        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        s.run_turn("first", turn_timeout=2.0)

        skill_path.write_text("changed content", encoding="utf-8")
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-002", "status": "completed", "error": None},
        )
        s.run_turn("second", turn_timeout=2.0)

        turn_starts = [params for method, params in client.requests if method == "turn/start"]
        assert turn_starts[0]["input"] == [{"type": "text", "text": "first"}]
        assert turn_starts[1]["input"] == [
            {"type": "skill", "name": "bridge", "path": str(skill_path)},
            {"type": "text", "text": "second"},
        ]

    def test_changed_hermes_skill_is_retried_after_turn_start_rejection(
        self, tmp_path
    ):
        from agent.transports.codex_app_server import CodexAppServerError

        skill_path = tmp_path / "skills" / "testing" / "bridge" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("initial", encoding="utf-8")

        client = FakeClient()
        original_request = client.request
        reject_next_turn = [False]

        def request(method, params=None, timeout=10.0):
            if method == "skills/list":
                return {
                    "data": [
                        {
                            "cwd": "/tmp",
                            "errors": [],
                            "skills": [
                                {
                                    "name": "bridge",
                                    "path": str(skill_path),
                                    "enabled": True,
                                }
                            ],
                        }
                    ]
                }
            if method == "turn/start" and reject_next_turn[0]:
                reject_next_turn[0] = False
                client.requests.append((method, params or {}))
                raise CodexAppServerError(code=-32600, message="rejected")
            return original_request(method, params, timeout)

        client.request = request
        s = make_session(client, extra_skill_roots=[str(tmp_path / "skills")])

        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        s.run_turn("first", turn_timeout=2.0)

        skill_path.write_text("changed content", encoding="utf-8")
        reject_next_turn[0] = True
        rejected = s.run_turn("second", turn_timeout=2.0)
        assert rejected.error is not None
        assert rejected.should_retire is False

        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-002", "status": "completed", "error": None},
        )
        retried = s.run_turn("third", turn_timeout=2.0)
        assert retried.error is None

        turn_starts = [
            params for method, params in client.requests if method == "turn/start"
        ]
        expected_skill = {
            "type": "skill",
            "name": "bridge",
            "path": str(skill_path),
        }
        assert turn_starts[1]["input"][0] == expected_skill
        assert turn_starts[2]["input"][0] == expected_skill

    def test_extra_skill_roots_registration_retries_on_the_next_turn(self, tmp_path):
        from agent.transports.codex_app_server import CodexAppServerError

        skill_path = tmp_path / "skills" / "testing" / "bridge" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("content", encoding="utf-8")

        client = FakeClient()
        original_request = client.request
        registration_attempts = [0]

        def request(method, params=None, timeout=10.0):
            if method == "skills/extraRoots/set":
                client.requests.append((method, params or {}))
                registration_attempts[0] += 1
                if registration_attempts[0] == 1:
                    raise CodexAppServerError(code=-32603, message="temporary failure")
                return {}
            if method == "skills/list":
                client.requests.append((method, params or {}))
                return {
                    "data": [
                        {
                            "cwd": "/tmp",
                            "errors": [],
                            "skills": [
                                {
                                    "name": "bridge",
                                    "path": str(skill_path),
                                    "enabled": True,
                                }
                            ],
                        }
                    ]
                }
            return original_request(method, params, timeout)

        client.request = request
        s = make_session(client, extra_skill_roots=[str(tmp_path / "skills")])

        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        first = s.run_turn("first", turn_timeout=2.0)
        assert first.error is None

        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-002", "status": "completed", "error": None},
        )
        second = s.run_turn("second", turn_timeout=2.0)
        assert second.error is None
        assert registration_attempts[0] == 2

        turn_starts = [
            params for method, params in client.requests if method == "turn/start"
        ]
        assert turn_starts[0]["input"] == [{"type": "text", "text": "first"}]
        assert turn_starts[1]["input"] == [
            {"type": "skill", "name": "bridge", "path": str(skill_path)},
            {"type": "text", "text": "second"},
        ]

    def test_initial_skill_reload_failure_preserves_missing_baseline(self, tmp_path):
        from agent.transports.codex_app_server import CodexAppServerError

        root = tmp_path / "skills"
        root.mkdir()
        skill_path = root / "testing" / "bridge" / "SKILL.md"

        client = FakeClient()
        original_request = client.request
        reload_attempts = [0]

        def request(method, params=None, timeout=10.0):
            if method == "skills/list":
                client.requests.append((method, params or {}))
                reload_attempts[0] += 1
                if reload_attempts[0] == 1:
                    raise CodexAppServerError(
                        code=-32603, message="temporary reload failure"
                    )
                skills = []
                if skill_path.exists():
                    skills.append(
                        {
                            "name": "bridge",
                            "path": str(skill_path),
                            "enabled": True,
                        }
                    )
                return {"data": [{"cwd": "/tmp", "errors": [], "skills": skills}]}
            return original_request(method, params, timeout)

        client.request = request
        s = make_session(client, extra_skill_roots=[str(root)])

        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        first = s.run_turn("first", turn_timeout=2.0)
        assert first.error is None

        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("content", encoding="utf-8")
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-002", "status": "completed", "error": None},
        )
        second = s.run_turn("second", turn_timeout=2.0)
        assert second.error is None

        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-003", "status": "completed", "error": None},
        )
        third = s.run_turn("third", turn_timeout=2.0)
        assert third.error is None

        turn_starts = [
            params for method, params in client.requests if method == "turn/start"
        ]
        assert turn_starts[0]["input"] == [{"type": "text", "text": "first"}]
        assert turn_starts[1]["input"] == [
            {"type": "skill", "name": "bridge", "path": str(skill_path)},
            {"type": "text", "text": "second"},
        ]
        assert turn_starts[2]["input"] == [{"type": "text", "text": "third"}]

    def test_skill_path_outside_extra_root_is_ignored(self, tmp_path):
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside" / "SKILL.md"
        outside.parent.mkdir()
        outside.write_text("outside", encoding="utf-8")

        client = FakeClient()
        original_request = client.request

        def request(method, params=None, timeout=10.0):
            if method == "skills/list":
                return {
                    "data": [
                        {
                            "cwd": "/tmp",
                            "errors": [],
                            "skills": [
                                {
                                    "name": "outside",
                                    "path": str(outside),
                                    "enabled": True,
                                }
                            ],
                        }
                    ]
                }
            return original_request(method, params, timeout)

        client.request = request
        s = make_session(client, extra_skill_roots=[str(root)])
        s.ensure_started()

        changed, fingerprints = s._reload_changed_skills()
        assert changed == []
        assert fingerprints == {}

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_symlinked_skill_escaping_extra_root_is_ignored(self, tmp_path):
        root = tmp_path / "skills"
        symlink_path = root / "testing" / "bridge" / "SKILL.md"
        symlink_path.parent.mkdir(parents=True)
        outside = tmp_path / "outside" / "SKILL.md"
        outside.parent.mkdir()
        outside.write_text("outside", encoding="utf-8")
        symlink_path.symlink_to(outside)

        client = FakeClient()
        original_request = client.request

        def request(method, params=None, timeout=10.0):
            if method == "skills/list":
                return {
                    "data": [
                        {
                            "cwd": "/tmp",
                            "errors": [],
                            "skills": [
                                {
                                    "name": "bridge",
                                    "path": str(symlink_path),
                                    "enabled": True,
                                }
                            ],
                        }
                    ]
                }
            return original_request(method, params, timeout)

        client.request = request
        s = make_session(client, extra_skill_roots=[str(root)])
        s.ensure_started()

        changed, fingerprints = s._reload_changed_skills()
        assert changed == []
        assert fingerprints == {}

    def test_token_usage_notification_is_captured(self):
        client = FakeClient()
        client.queue_notification(
            "thread/tokenUsage/updated",
            threadId="thread-fake-001",
            turnId="turn-fake-001",
            tokenUsage={
                "last": {
                    "totalTokens": 130,
                    "inputTokens": 80,
                    "cachedInputTokens": 20,
                    "outputTokens": 25,
                    "reasoningOutputTokens": 5,
                },
                "total": {
                    "totalTokens": 500,
                    "inputTokens": 300,
                    "cachedInputTokens": 75,
                    "outputTokens": 100,
                    "reasoningOutputTokens": 25,
                },
                "modelContextWindow": 200000,
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        r = make_session(client).run_turn("hi", turn_timeout=2.0)
        assert r.token_usage_last["totalTokens"] == 130
        assert r.token_usage_total["totalTokens"] == 500
        assert r.model_context_window == 200000

    def test_rich_content_turn_is_collapsed_to_text_payload(self):
        client = FakeClient()
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn(
            [
                {
                    "type": "text",
                    "text": "look at this\n\n[Image attached at: /tmp/a.png]",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
            turn_timeout=2.0,
        )
        assert r.error is None
        method, params = next(req for req in client.requests if req[0] == "turn/start")
        assert method == "turn/start"
        text = params["input"][0]["text"]
        assert isinstance(text, str)
        assert "[Image attached at: /tmp/a.png]" in text
        assert "[image attached]" in text

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

    def test_turn_start_failure_returns_error(self):
        client = FakeClient()
        from agent.transports.codex_app_server import CodexAppServerError

        def boom(method, params):
            if method == "turn/start":
                raise CodexAppServerError(code=-32600, message="bad input")
            return {"thread": {"id": "t"}, "activePermissionProfile": {"id": "x"}}

        client._request_handler = boom
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "bad input" in r.error
        assert r.final_text == ""

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

    def test_startup_failure_returns_error_with_stderr(self):
        """Codex thread/start failures during ensure_started() used to bubble
        up as uncaught exceptions. Now they return a TurnResult.error so
        AIAgent surfaces a clean diagnostic instead of crashing the turn."""
        client = FakeClient()
        client.set_stderr_tail([
            "FATAL: model_provider 'azure_foundry' not configured",
        ])
        from agent.transports.codex_app_server import CodexAppServerError

        def boom(method, params):
            if method == "thread/start":
                raise CodexAppServerError(code=-32603, message="Internal error")
            return {}

        client._request_handler = boom
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "startup failed" in r.error
        assert "model_provider 'azure_foundry' not configured" in r.error
        assert r.should_retire is True
        assert r.final_text == ""

    def test_interrupt_during_turn_issues_turn_interrupt(self):
        client = FakeClient()
        # Don't queue turn/completed — the loop has to interrupt out
        client.queue_notification(
            "item/completed",
            item={"type": "commandExecution", "id": "x", "command": "sleep 60",
                  "cwd": "/", "status": "inProgress",
                  "aggregatedOutput": None, "exitCode": None,
                  "commandActions": []},
            threadId="t", turnId="tu1",
        )
        s = make_session(client)
        s.ensure_started()
        # Trip the interrupt before run_turn even consumes the notification.
        # The loop will see interrupt set on its first iteration and bail.
        s.request_interrupt()
        r = s.run_turn("loop forever", turn_timeout=2.0)
        assert r.interrupted is True
        # turn/interrupt was requested with the right turnId
        assert any(
            method == "turn/interrupt" and params.get("turnId") == "turn-fake-001"
            for (method, params) in client.requests
        )

    def test_deadline_exceeded_records_error(self):
        client = FakeClient()
        # No notifications and no completion → must hit deadline
        s = make_session(client)
        r = s.run_turn("never finishes", turn_timeout=0.05,
                       notification_poll_timeout=0.01)
        assert r.interrupted is True
        assert r.error and "timed out" in r.error

    def test_deadline_uses_monotonic_clock(self):
        client = FakeClient()
        s = make_session(client)
        monotonic_values = iter([1000.0, 999.0, 999.0, 1001.0])
        with patch.object(
            session_mod.time,
            "monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            r = s.run_turn(
                "never finishes",
                turn_timeout=0.1,
                notification_poll_timeout=0.0,
            )
        assert r.interrupted is True
        assert r.error and "timed out" in r.error

    def test_failed_turn_records_error_from_turn_completed(self):
        client = FakeClient()
        client.queue_notification(
            "turn/completed", threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "failed",
                  "error": {"message": "model error"}},
        )
        s = make_session(client)
        r = s.run_turn("x", turn_timeout=1.0)
        assert r.error and "model error" in r.error

    def test_run_turn_records_native_compaction_item(self):
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="turn-fake-001",
            item={"type": "contextCompaction", "id": "compact-item-1"},
        )
        client.queue_notification(
            "turn/completed", threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )

        r = make_session(client).run_turn("x", turn_timeout=1.0)

        assert r.compacted is True
        assert r.thread_id == "thread-fake-001"
        assert r.turn_id == "turn-fake-001"

    def test_run_turn_records_deprecated_thread_compacted_notification(self):
        client = FakeClient()
        client.queue_notification(
            "thread/compacted",
            threadId="thread-fake-001",
            turnId="turn-fake-001",
        )
        client.queue_notification(
            "turn/completed", threadId="thread-fake-001",
            turn={"id": "turn-fake-001", "status": "completed", "error": None},
        )

        r = make_session(client).run_turn("x", turn_timeout=1.0)

        assert r.compacted is True


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

    def test_compact_thread_interrupted_returns_non_success(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1", "status": "interrupted", "error": None},
        )

        result = make_session(client).compact_thread(turn_timeout=2.0)

        assert result.interrupted is True
        assert result.error == "compact turn interrupted"

    def test_compact_thread_failure_returns_error(self):
        client = FakeClient()
        from agent.transports.codex_app_server import CodexAppServerError

        def boom(method, params):
            if method == "thread/compact/start":
                raise CodexAppServerError(code=-32603, message="compact unavailable")
            if method == "thread/start":
                return {"thread": {"id": "thread-fake-001"}}
            return {}

        client._request_handler = boom
        r = make_session(client).compact_thread(turn_timeout=2.0)

        assert r.error is not None
        assert "compact unavailable" in r.error
        assert r.should_retire is False


# ---- approval bridge ----

class TestServerRequestRouting:
    def test_exec_approval_with_callback_approves_once(self):
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval", request_id="req-1",
            command="ls /tmp", cwd="/tmp",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        captured: dict = {}

        def cb(command, description, *, allow_permanent=True):
            captured["command"] = command
            captured["description"] = description
            return "once"

        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        assert captured["command"] == "ls /tmp"
        # The session must have responded to the server request with "accept"
        assert ("req-1", {"decision": "accept"}) in client.responses

    def test_exec_approval_no_callback_denies(self):
        client = FakeClient()
        client.queue_server_request("item/commandExecution/requestApproval", request_id="req-1",
                                    command="rm -rf /", cwd="/")
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)  # no approval_callback wired
        s.run_turn("hi", turn_timeout=1.0)
        assert ("req-1", {"decision": "decline"}) in client.responses

    def test_apply_patch_approval_session_maps_to_session_decision(self):
        client = FakeClient()
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="req-2",
            itemId="fc-1",
            turnId="t1",
            threadId="th",
            startedAtMs=1234567890,
            reason="create new file with hello() function",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        def cb(command, description, *, allow_permanent=True):
            return "session"

        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        assert ("req-2", {"decision": "acceptForSession"}) in client.responses

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

    def test_mcp_elicitation_for_hermes_tools_auto_accepts(self):
        """When codex elicits on behalf of hermes-tools (our own callback),
        accept automatically — the user already opted in by enabling the
        runtime."""
        client = FakeClient()
        client.queue_server_request(
            "mcpServer/elicitation/request", request_id="elic-1",
            threadId="t", turnId="tu1",
            serverName="hermes-tools",
            mode="form",
            message="confirm",
            requestedSchema={"type": "object", "properties": {}},
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        s.run_turn("hi", turn_timeout=1.0)
        assert ("elic-1", {"action": "accept", "content": None, "_meta": None}) in client.responses

    def test_mcp_elicitation_for_other_servers_declines(self):
        """For third-party MCP servers we decline by default so users
        explicitly opt in through codex's own UI."""
        client = FakeClient()
        client.queue_server_request(
            "mcpServer/elicitation/request", request_id="elic-2",
            threadId="t", turnId="tu1",
            serverName="some-third-party",
            mode="url",
            message="please log in",
            url="https://example.com/oauth",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        s.run_turn("hi", turn_timeout=1.0)
        assert ("elic-2", {"action": "decline", "content": None, "_meta": None}) in client.responses

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

    def test_callback_raises_falls_back_to_decline(self):
        client = FakeClient()
        client.queue_server_request("item/commandExecution/requestApproval", request_id="r1",
                                    command="ls", cwd="/")
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        def boom(*a, **kw):
            raise RuntimeError("ui crashed")

        s = make_session(client, approval_callback=boom)
        s.run_turn("hi", turn_timeout=1.0)
        # Fail-closed: deny on callback exception
        assert ("r1", {"decision": "decline"}) in client.responses


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

    def test_deadline_marks_session_for_retirement(self):
        client = FakeClient()
        s = make_session(client)
        r = s.run_turn(
            "never finishes",
            turn_timeout=0.05,
            notification_poll_timeout=0.01,
        )
        assert r.interrupted is True
        assert r.error and "timed out" in r.error
        assert r.should_retire is True, (
            "Deadline exhaustion must signal retirement so the next turn "
            "respawns codex instead of riding a wedged subprocess."
        )

    def test_completed_turn_does_not_retire(self):
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "hi"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=1.0)
        assert r.should_retire is False

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

    def test_post_tool_quiet_watchdog_trips_and_retires(self):
        client = FakeClient()
        # One tool completion, then total silence — no further events,
        # no turn/completed. With a tiny post_tool_quiet_timeout the
        # watchdog must fire before the larger turn deadline.
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
        r = s.run_turn(
            "tool then silence",
            turn_timeout=5.0,           # would be miserable to wait
            notification_poll_timeout=0.02,
            post_tool_quiet_timeout=0.15,
        )
        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "silent" in r.error
        # Confirm we issued turn/interrupt to free codex compute
        assert any(method == "turn/interrupt" for (method, _) in client.requests)

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

    def test_turn_aborted_marker_in_text_is_terminal(self):
        """If codex emits `<turn_aborted>` in agent text and never sends
        turn/completed, we still exit promptly instead of burning the
        deadline."""
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "agentMessage", "id": "m1",
                "text": "partial output... <turn_aborted>",
            },
            threadId="t", turnId="tu1",
        )
        # Deliberately NO turn/completed notification queued.
        s = make_session(client)
        r = s.run_turn(
            "abort mid-turn", turn_timeout=2.0,
            notification_poll_timeout=0.01,
        )
        assert r.interrupted is True
        assert r.error and "turn_aborted" in r.error
        # Should have exited fast — not waited for the full 2s deadline.
        # (Can't measure wall clock reliably in CI; presence of the marker
        # error string instead of a "timed out" message is the proxy.)
        assert "timed out" not in r.error

    def test_turn_aborted_self_closing_marker_also_terminal(self):
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1",
                  "text": "<turn_aborted/>"},
            threadId="t", turnId="tu1",
        )
        s = make_session(client)
        r = s.run_turn("x", turn_timeout=2.0,
                       notification_poll_timeout=0.01)
        assert r.interrupted is True
        assert r.error and "turn_aborted" in r.error

    def test_oauth_refresh_failure_on_turn_start_suggests_login(self):
        from agent.transports.codex_app_server import CodexAppServerError

        client = FakeClient()

        def boom(method, params):
            if method == "turn/start":
                raise CodexAppServerError(
                    code=-32603,
                    message="auth refresh failed: invalid_grant",
                )
            return {"thread": {"id": "t"},
                    "activePermissionProfile": {"id": "x"}}

        client._request_handler = boom
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=1.0)
        assert r.error is not None
        assert "codex login" in r.error
        assert r.should_retire is True

    def test_oauth_failure_from_stderr_on_turn_start_failure(self):
        """If the RPC error itself is opaque but stderr shows an auth
        problem, we still classify it as a refresh failure."""
        from agent.transports.codex_app_server import CodexAppServerError

        client = FakeClient()
        client.set_stderr_tail([
            "[2026-05-14T10:00:00Z WARN codex_core::auth] token refresh failed",
            "[2026-05-14T10:00:00Z ERROR codex_core] please log in again",
        ])

        def boom(method, params):
            if method == "turn/start":
                raise CodexAppServerError(code=-32603, message="rpc broke")
            return {"thread": {"id": "t"},
                    "activePermissionProfile": {"id": "x"}}

        client._request_handler = boom
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=1.0)
        assert r.error is not None
        assert "codex login" in r.error
        assert r.should_retire is True

    def test_oauth_failure_in_turn_completed_error(self):
        """A failed turn/completed whose error mentions auth/refresh
        triggers the re-auth hint + retirement."""
        client = FakeClient()
        client.queue_notification(
            "turn/completed", threadId="thread-fake-001",
            turn={
                "id": "turn-fake-001", "status": "failed",
                "error": {"message": "401 Unauthorized: please reauthenticate"},
            },
        )
        s = make_session(client)
        r = s.run_turn("x", turn_timeout=1.0,
                       notification_poll_timeout=0.01)
        assert r.error is not None
        assert "codex login" in r.error
        assert r.should_retire is True

    def test_generic_turn_failure_does_not_trigger_oauth_hint(self):
        """A boring model error must NOT rewrite the message into a fake
        re-auth hint. Conservative classifier."""
        client = FakeClient()
        client.queue_notification(
            "turn/completed", threadId="thread-fake-001",
            turn={
                "id": "turn-fake-001", "status": "failed",
                "error": {"message": "rate limit exceeded"},
            },
        )
        s = make_session(client)
        r = s.run_turn("x", turn_timeout=1.0,
                       notification_poll_timeout=0.01)
        assert r.error is not None
        assert "codex login" not in r.error
        assert "rate limit exceeded" in r.error
        # Generic model failures don't retire — the session itself is fine
        assert r.should_retire is False

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

    def test_thread_session_id_alias_under_thread_key(self):
        client = FakeClient()
        client._request_handler = lambda method, params: (
            {"thread": {"sessionId": "alias-1"},
             "activePermissionProfile": {"id": "x"}}
            if method == "thread/start" else
            {"turn": {"id": "tu1"}} if method == "turn/start" else {}
        )
        s = make_session(client)
        tid = s.ensure_started()
        assert tid == "alias-1"

    def test_top_level_session_id_fallback(self):
        client = FakeClient()
        client._request_handler = lambda method, params: (
            {"sessionId": "top-1"} if method == "thread/start" else
            {"turn": {"id": "tu1"}} if method == "turn/start" else {}
        )
        s = make_session(client)
        tid = s.ensure_started()
        assert tid == "top-1"

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

    def test_self_closing_marker(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("<turn_aborted/>") is True


class TestClassifyOAuthFailure:
    """Unit coverage for the OAuth classifier; conservative on purpose."""

    def test_invalid_grant_classified(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        hint = _classify_oauth_failure("error: invalid_grant returned by server")
        assert hint is not None
        assert "codex login" in hint

    def test_token_refresh_classified(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        hint = _classify_oauth_failure("token refresh failed: network error")
        assert hint is not None
        assert "codex login" in hint

    def test_401_classified(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        hint = _classify_oauth_failure("HTTP 401 Unauthorized")
        assert hint is not None

    def test_generic_error_not_classified(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        assert _classify_oauth_failure("connection reset") is None
        assert _classify_oauth_failure("model returned bad json") is None
        assert _classify_oauth_failure("rate limit exceeded") is None

    def test_empty_inputs(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        assert _classify_oauth_failure() is None
        assert _classify_oauth_failure("") is None
        assert _classify_oauth_failure("", None) is None  # type: ignore[arg-type]

    def test_multi_string_search(self):
        """Hint can come from any of the provided strings."""
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        hint = _classify_oauth_failure(
            "rpc returned -32603",
            "[stderr] token has expired, run codex login",
        )
        assert hint is not None
