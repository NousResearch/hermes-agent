"""Gateway approval bridge — choice mapping, session fallback, smart decisions, observers — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

import asyncio
import logging
import tracemalloc
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from tools import approval_context as approval_ctx
from tools import approval_gateway_wait
from tools import approval_sdk_gateway as sdk_gateway
from tools import approval_smart
from tests.agent.claude_sdk_fakes import (
    _make_session,
    _make_turn,
    _make_agent,
    isolate_provider_config,
    ApprovalBridgeCase,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestGatewayApprovalBridgeDecisions(ApprovalBridgeCase):
    """Choice mapping, ``_create_session`` fallback and relays, smart-mode SDK
    decisions, tool_use_id caps and observer containment (split from
    ``TestGatewayApprovalBridge``)."""

    def test_approve_maps_to_once_and_clamps_durable_choices(self, monkeypatch):
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-mapped")
        try:
            # An older client button can still send "always" — the grant must
            # not outlive the single SDK permission request it answered.
            notify, seen = self._resolve_with(approval_mod, "sess-mapped", "always")
            approval_mod.register_gateway_notify("sess-mapped", notify)
            try:
                cb = sdk_gateway.build_sdk_gateway_approval_callback()
                assert self._call_gateway(cb, "uname") == "once"
            finally:
                approval_mod.unregister_gateway_notify("sess-mapped")
            assert seen[0]["allow_permanent"] is False
            assert seen[0]["allow_session"] is False
            assert seen[0]["command"] == "Bash(command=uname)"
        finally:
            approval_ctx.reset_current_session_key(token)
    def test_deny_choice_denies(self, monkeypatch):
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-denied")
        try:
            notify, _ = self._resolve_with(approval_mod, "sess-denied", "deny")
            approval_mod.register_gateway_notify("sess-denied", notify)
            try:
                cb = sdk_gateway.build_sdk_gateway_approval_callback()
                assert self._call_gateway(cb, "rm x") == {
                    "choice": "deny",
                    "operator_denial": True,
                    "reason": "",
                }
            finally:
                approval_mod.unregister_gateway_notify("sess-denied")
        finally:
            approval_ctx.reset_current_session_key(token)
    def test_create_session_falls_back_to_gateway_bridge(self, monkeypatch):
        # The runtime seam: no thread-local CLI callback + gateway context →
        # the session is constructed with the bridge callback, not None.
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn(
                    projected_messages=[], final_text="ok",
                    token_usage_last=None,
                )

            def close(self):
                pass

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        token = approval_ctx.set_current_session_key("tg:152:bridge")
        try:
            monkeypatch.setattr(
                session_mod, "ClaudeAgentSdkSession", _CapturingSession
            )
            agent = _make_agent()
            agent._claude_sdk_session = None
            run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="task-1",
            )
            assert captured.get("approval_callback") is not None
        finally:
            approval_ctx.reset_current_session_key(token)
    def test_sdk_tool_start_updates_shared_activity_before_progress(self, monkeypatch):
        """SDK lifecycle must drive the shared heartbeat activity contract."""
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn(projected_messages=[], final_text="ok")

            def close(self):
                pass

        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        seen_progress = []
        agent.tool_progress_callback = lambda *args: seen_progress.append(args)
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )

        captured["on_tool_started"]("Bash", "sqlite3 …", {"command": "sqlite3"})

        assert agent._current_tool == "Bash"
        agent._touch_activity.assert_called_once_with("executing tool: Bash")
        assert seen_progress == [("tool.started", "Bash", "sqlite3 …", {"command": "sqlite3"})]
    def test_sdk_tool_result_updates_isolated_live_iteration_counter(self, monkeypatch):
        """SDK result advances its guarded visibility count, not native state."""
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn(projected_messages=[], final_text="ok")

            def close(self):
                pass

        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._current_turn_id = "turn-one"
        agent._api_call_count = 0
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="task-1",
        )
        captured["on_tool_iteration"]()
        assert agent._sdk_visibility_iteration_count == 1
        assert agent._api_call_count == 0
    def test_late_sdk_callback_is_fenced_after_turn_replacement(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}
        class _CapturingSession:
            def __init__(self, **kwargs): captured.update(kwargs)
            def run_turn(self, user_input): return _make_turn(projected_messages=[], final_text="ok")
            def close(self): pass
        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._current_turn_id = "turn-one"
        run_claude_agent_sdk_turn(agent, user_message="one", original_user_message="one", messages=[{"role": "user", "content": "one"}], effective_task_id="one")
        stale = captured["on_tool_iteration"]
        agent._current_turn_id = "turn-two"
        run_claude_agent_sdk_turn(agent, user_message="two", original_user_message="two", messages=[{"role": "user", "content": "two"}], effective_task_id="two")
        stale()
        assert agent._sdk_visibility_iteration_count == 0
    def test_sdk_interim_relay_scrubs_redacts_and_deduplicates(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}
        class _CapturingSession:
            def __init__(self, **kwargs): captured.update(kwargs)
            def run_turn(self, user_input): return _make_turn(projected_messages=[], final_text="ok")
            def close(self): pass
        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._current_turn_id = "turn-one"
        agent._strip_think_blocks.side_effect = lambda text: text.replace("<think>private</think>", "")
        agent._delivered_interim_texts = set()
        agent._interim_text_was_delivered.side_effect = lambda text: text in agent._delivered_interim_texts
        agent._record_delivered_interim_text.side_effect = lambda text: agent._delivered_interim_texts.add(text)
        # Nothing was streamed in this scenario, so the computed flag is False.
        agent._interim_content_was_streamed.return_value = False
        delivered = []
        agent.interim_assistant_callback = lambda text, **kw: delivered.append((text, kw))
        run_claude_agent_sdk_turn(agent, user_message="one", original_user_message="one", messages=[{"role": "user", "content": "one"}], effective_task_id="one")
        relay = captured["on_interim_assistant"]
        relay("<think>private</think> Checking token sk-ant-12345678901234567890")
        relay("<think>private</think> Checking token sk-ant-12345678901234567890")
        assert len(delivered) == 1
        assert "private" not in delivered[0][0]
        assert "12345678901234567890" not in delivered[0][0]
        assert delivered[0][1] == {"already_streamed": False}
    def test_interim_is_not_sealed_when_no_sink_accepted(self, monkeypatch):
        """Sealing a segment nobody painted would drop the prose from the UI.

        The accumulator cannot answer this alone: it records before the sinks
        run on purpose, so a turn whose only sink raises still has content in
        it.
        """
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs): captured.update(kwargs)
            def run_turn(self, user_input): return _make_turn(projected_messages=[], final_text="ok")
            def close(self): pass

        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._current_turn_id = "turn-one"
        agent._strip_think_blocks.side_effect = lambda text: text
        agent._interim_text_was_delivered.return_value = False
        agent._interim_content_was_streamed.return_value = True
        agent._stream_callback = None
        agent.stream_delta_callback = MagicMock(side_effect=RuntimeError("sink down"))
        delivered = []
        agent.interim_assistant_callback = lambda text, **kw: delivered.append((text, kw))
        run_claude_agent_sdk_turn(
            agent,
            user_message="one",
            original_user_message="one",
            messages=[{"role": "user", "content": "one"}],
            effective_task_id="one",
        )

        captured["on_stream_delta"]("never painted")
        captured["on_interim_assistant"]("never painted")
        assert delivered[-1][1] == {"already_streamed": False}

        agent.stream_delta_callback = MagicMock()
        captured["on_stream_delta"]("painted")
        captured["on_interim_assistant"]("painted")
        assert delivered[-1][1] == {"already_streamed": True}
    def test_sdk_interim_relay_reports_already_streamed_prose(self, monkeypatch):
        """Prose the delta sink already painted must be sealed, not re-sent.

        Hardcoding ``already_streamed=False`` made the surface render streamed
        text a second time on top of the streaming buffer.
        """
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs): captured.update(kwargs)
            def run_turn(self, user_input): return _make_turn(projected_messages=[], final_text="ok")
            def close(self): pass

        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._current_turn_id = "turn-one"
        agent._strip_think_blocks.side_effect = lambda text: text
        agent._interim_text_was_delivered.return_value = False
        agent._interim_content_was_streamed.side_effect = lambda text: text == "already painted"
        delivered = []
        agent.interim_assistant_callback = lambda text, **kw: delivered.append((text, kw))
        run_claude_agent_sdk_turn(
            agent,
            user_message="one",
            original_user_message="one",
            messages=[{"role": "user", "content": "one"}],
            effective_task_id="one",
        )
        relay = captured["on_interim_assistant"]
        # A sink accepted a delta this turn; the accumulator decides which
        # prose it covered.
        agent._sdk_stream_sink_accepted = True

        relay("already painted")
        relay("never streamed")

        assert delivered[0] == ("already painted", {"already_streamed": True})
        assert delivered[1] == ("never streamed", {"already_streamed": False})
    def test_create_session_without_gateway_context_keeps_none(self, monkeypatch):
        # CLI/bare-process posture unchanged: no context → callback stays None.
        import agent.transports.claude_agent_sdk_session as session_mod

        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn(
                    projected_messages=[], final_text="ok",
                    token_usage_last=None,
                )

            def close(self):
                pass

        monkeypatch.setattr(
            session_mod, "ClaudeAgentSdkSession", _CapturingSession
        )
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        assert captured.get("approval_callback") is None
    def test_smart_approved_sdk_read_skips_operator_card(self, monkeypatch):
        import json

        sk = "sess-smart-sdk-read"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        notified = []
        smart_calls = []
        prepared = []
        observed = []
        awaited = []
        try:
            approval_mod.register_gateway_notify(sk, notified.append)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(
                approval_gateway_wait,
                "_await_gateway_decision",
                lambda *args, **kwargs: awaited.append((args, kwargs)) or {
                    "resolved": True, "choice": "once",
                },
            )
            monkeypatch.setattr(
                approval_smart,
                "_smart_approve",
                lambda *args, **kwargs: smart_calls.append((args, kwargs)) or "approve",
            )
            monkeypatch.setattr(
                approval_smart,
                "_prepare_smart_approval_observer",
                lambda **kwargs: prepared.append(kwargs) or {"safe": True},
            )
            monkeypatch.setattr(
                approval_smart,
                "_observe_smart_approval_verdict",
                lambda payload, verdict, **kwargs: observed.append(
                    (payload, verdict, kwargs)
                ),
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert cb is not None

            result = self._call_gateway(
                cb,
                tool_name="Read",
                tool_input={"file_path": "/srv/example-app/src/backup.py"},
            )

            canonical = json.dumps(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/srv/example-app/src/backup.py"},
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            assert result == "once"
            assert smart_calls == [((canonical, "Claude requests SDK tool Read"), {})]
            assert prepared == [{
                "command": "Read(path=/srv/example-app/src/backup.py)",
                "description": "Claude requests SDK tool Read",
                "pattern_key": "claude_sdk_tool",
                "pattern_keys": ["claude_sdk_tool"],
                "session_key": sk,
            }]
            assert observed == [({"safe": True}, "approve", {})]
            assert awaited == []
            assert notified == []
            with approval_mod._lock:
                assert not approval_mod._gateway_queues.get(sk)
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_smart_denied_sdk_owner_override_card_is_one_shot_and_resets_tally(
        self, monkeypatch,
    ):
        sk = "sess-smart-sdk-deny-owner-override"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        seen = []

        def notify(data):
            seen.append(dict(data))
            approval_mod.resolve_gateway_approval(sk, "once")

        try:
            approval_mod._reset_denials(sk)
            approval_mod.register_gateway_notify(sk, notify)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: "deny")
            cb = sdk_gateway.build_sdk_gateway_approval_callback()

            assert self._call_gateway(
                cb,
                tool_name="Read",
                tool_input={"file_path": "/tmp/card-target"},
                tool_use_id="toolu-card",
            ) == "once"
            assert len(seen) == 1
            request_id = seen[0].pop("request_id")
            assert type(request_id) is str and request_id
            assert seen == [{
                "command": "Read(path=/tmp/card-target)",
                "pattern_key": "claude_sdk_tool",
                "pattern_keys": ["claude_sdk_tool"],
                "description": "Claude requests SDK tool Read",
                "allow_permanent": False,
                "allow_session": False,
                "smart_denied": True,
                "tool_use_id": "toolu-card",
                "no_coalesce": True,
            }]
            with approval_mod._lock:
                assert sk not in approval_mod._denial_tally
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_mod._reset_denials(sk)
            approval_ctx.reset_current_session_key(token)
    def test_sdk_bash_hardline_denies_before_smart_or_card(self, monkeypatch):
        sk = "sess-sdk-hardline"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        notified = []
        smart_calls = []
        try:
            approval_mod.register_gateway_notify(sk, notified.append)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(
                approval_mod,
                "detect_hardline_command",
                lambda command: (command == "dangerous-probe", "test hardline"),
            )
            monkeypatch.setattr(
                approval_smart,
                "_smart_approve",
                lambda *args, **kwargs: smart_calls.append((args, kwargs)) or "approve",
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert cb is not None
            session, _ = _make_session(
                approval_callback=cb,
                permission_mode="default",
                hermes_session_id=sk,
            )

            result = asyncio.run(session._make_can_use_tool()(
                "Bash",
                {"command": "dangerous-probe"},
                SimpleNamespace(tool_use_id="toolu-hardline"),
            ))

            assert type(result).__name__ == "PermissionResultDeny"
            assert result.message == "approval denied by callback"
            assert smart_calls == []
            assert notified == []
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_sdk_gateway_card_drops_mixed_control_tool_use_id(self, monkeypatch):
        sk = "sess-sdk-hostile-id"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            notify, seen = self._resolve_with(approval_mod, sk, "deny")
            approval_mod.register_gateway_notify(sk, notify)
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(
                callback, "true", tool_use_id="toolu_ok\x00\nINJECT",
            )
            assert result == {
                "choice": "deny", "operator_denial": True, "reason": "",
            }
            assert seen[0]["tool_use_id"] == ""
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    @pytest.mark.parametrize(
        ("tool_use_id", "expected"),
        [
            ("x" * 256, "x" * 256),
            ("x" * 257, ""),
            ("é" * 128, "é" * 128),
            (("é" * 128) + "x", ""),
        ],
    )
    def test_gateway_tool_use_id_enforces_exact_utf8_byte_cap(
        self, monkeypatch, tool_use_id, expected,
    ):
        session_key = "sess-sdk-id-byte-cap"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "deny")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                result = self._call_gateway(
                    callback, "true", tool_use_id=tool_use_id,
                )
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert result == {
            "choice": "deny", "operator_denial": True, "reason": "",
        }
        assert [card["tool_use_id"] for card in seen] == [expected]
        if not expected:
            assert tool_use_id not in [card["tool_use_id"] for card in seen]
    def test_gateway_huge_multibyte_tool_use_id_has_bounded_peak_allocation(
        self, monkeypatch,
    ):
        session_key = "sess-sdk-huge-id-allocation"
        huge_id = "é" * (16 * 1024 * 1024)
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "deny")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                self._call_gateway(callback, "true", tool_use_id="warmup")
                seen.clear()
                tracemalloc.start()
                try:
                    result = self._call_gateway(
                        callback, "true", tool_use_id=huge_id,
                    )
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert result == {
            "choice": "deny", "operator_denial": True, "reason": "",
        }
        assert [card["tool_use_id"] for card in seen] == [""]
        assert huge_id not in [card["tool_use_id"] for card in seen]
        assert peak < 2 * 1024 * 1024
    @pytest.mark.parametrize("bypass_source", ["off", "process_yolo", "session_yolo"])
    def test_sdk_bypass_sources_skip_approver_after_bash_floors(
        self, monkeypatch, bypass_source,
    ):
        sk = f"sess-sdk-bypass-{bypass_source}"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            if bypass_source == "off":
                monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "off")
            elif bypass_source == "process_yolo":
                monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", True)
            else:
                monkeypatch.setattr(
                    approval_mod, "is_session_yolo_enabled", lambda key: key == sk,
                )
            monkeypatch.setattr(
                approval_gateway_wait,
                "_await_gateway_decision",
                lambda *_a, **_k: pytest.fail("bypass reached approval queue"),
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert cb is not None
            session, _ = _make_session(
                approval_callback=cb,
                permission_mode="default",
                hermes_session_id=sk,
            )
            result = asyncio.run(session._make_can_use_tool()(
                "Bash",
                {"command": "printf safe"},
                SimpleNamespace(tool_use_id="toolu-bypass"),
            ))
            assert type(result).__name__ == "PermissionResultAllow"
            assert result.updated_input == {"command": "printf safe"}
        finally:
            approval_ctx.reset_current_session_key(token)
    def test_sdk_smart_evaluator_exception_uses_fixed_log(
        self, monkeypatch, caplog,
    ):
        import json

        sk = "sess-sdk-smart-exception"
        marker = "SMART_EXCEPTION_SECRET_7b9"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            from agent import auxiliary_client
            monkeypatch.setattr(
                auxiliary_client, "call_llm",
                lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(marker)),
            )
            monkeypatch.setattr(
                approval_gateway_wait,
                "_await_gateway_decision",
                lambda *_a, **_k: {"resolved": True, "choice": "deny"},
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = json.dumps(
                {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}},
                sort_keys=True, separators=(",", ":"),
            )
            with caplog.at_level(logging.DEBUG, logger="tools.approval"):
                assert cb("untrusted", "untrusted", canonical_tool_input=canonical) == {
                    "choice": "deny", "operator_denial": True, "reason": "",
                }
            assert marker not in caplog.text
            assert "Smart approvals: LLM call failed, escalating" in caplog.text
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_hostile_exact_session_key_never_reaches_sdk_log(
        self, monkeypatch, caplog,
    ):
        import json

        marker = "CTX_SESSION_SECRET_7c91"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        cb = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: {"gateway": True, "session_key": marker},
        )
        canonical = json.dumps(
            {"tool_name": "Bash", "tool_input": {"command": "true"}},
            sort_keys=True, separators=(",", ":"),
        )
        with caplog.at_level(logging.WARNING, logger="tools.approval"):
            result = cb("untrusted", "untrusted", canonical_tool_input=canonical)
        assert result == {
            "choice": "deny", "reason": "no approver available (background context)",
        }
        assert marker not in caplog.text
    def test_sdk_smart_observer_dispatch_exceptions_use_fixed_stage_logs(
        self, monkeypatch, caplog,
    ):
        import json

        from hermes_cli import lifecycle

        sk = "SDK_SESSION_SECRET_65982"
        marker = "SDK_OBSERVER_EXCEPTION_SECRET_65982"
        presentation = "Read(path=/tmp/visible-path)"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)

        def broken_dispatch(hook_name, **kwargs):
            raise RuntimeError(
                f"{marker} stage={hook_name} session={kwargs['session_key']} "
                f"command={kwargs['command']}"
            )

        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: "approve")
            monkeypatch.setattr(lifecycle, "invoke_hook", broken_dispatch)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = json.dumps(
                {"tool_name": "Read", "tool_input": {"file_path": "/tmp/visible-path"}},
                sort_keys=True, separators=(",", ":"),
            )
            with caplog.at_level(logging.DEBUG, logger="tools.approval"):
                assert cb("untrusted", "untrusted", canonical_tool_input=canonical) == "once"

            assert marker not in caplog.text
            assert sk not in caplog.text
            assert presentation not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
            assert [record.getMessage() for record in caplog.records] == [
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
            ]
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_sdk_smart_builtin_observer_failure_is_contained_at_inner_sink(
        self, monkeypatch, caplog,
    ):
        from agent import relay_runtime
        from hermes_cli import plugins
        from hermes_cli.observability import relay_shared_metrics

        sk = "SDK_BUILTIN_SESSION_SECRET_65982"
        marker = "SDK_BUILTIN_OBSERVER_SECRET_65982"
        path = "/tmp/SDK_BUILTIN_PATH_SECRET_65982"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)

        class BrokenRuntime:
            def record_approval(self, kwargs):
                raise RuntimeError(f"{marker} payload={kwargs!r}")

        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: "approve")
            monkeypatch.setattr(relay_shared_metrics, "handles_hook", lambda _name: True)
            monkeypatch.setattr(
                relay_runtime, "relay_instrumentation_enabled", lambda: True,
            )
            monkeypatch.setattr(relay_shared_metrics, "_get_runtime", lambda: BrokenRuntime())
            monkeypatch.setattr(plugins, "get_plugin_manager", lambda: plugins.PluginManager())
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            with caplog.at_level(logging.DEBUG):
                assert self._call_gateway(
                    cb, tool_name="Read", tool_input={"file_path": path},
                ) == "once"

            assert marker not in caplog.text
            assert sk not in caplog.text
            assert path not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
            # handles_hook is forced True for every hook, so main's table dispatch
            # (relay_shared_metrics._HOOK_HANDLERS) fails the pre hook as well as the
            # post hook; both failures must surface as the fixed lines only.
            assert [
                record.getMessage() for record in caplog.records
                if "SDK Smart" in record.getMessage()
            ] == [
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
            ]
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    @pytest.mark.parametrize(
        ("verdict", "expected", "expected_logs"),
        [
            ("approve", "once", [
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
            ]),
            ("deny", {"choice": "deny", "operator_denial": True, "reason": ""}, [
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
            ]),
        ],
    )
    def test_sdk_smart_real_plugin_callbacks_contain_inner_runtime_failure(
        self, monkeypatch, caplog, verdict, expected, expected_logs,
    ):
        from hermes_cli import plugins
        from hermes_cli.observability import relay_shared_metrics

        sk = "SDK_PLUGIN_SESSION_SECRET_65982"
        marker = "SDK_PLUGIN_OBSERVER_SECRET_65982"
        path = "/tmp/SDK_PLUGIN_PATH_SECRET_65982"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        manager = plugins.PluginManager()

        def broken_plugin(**kwargs):
            raise RuntimeError(f"{marker} payload={kwargs!r}")

        manager._hooks["pre_approval_request"] = [broken_plugin]
        manager._hooks["post_approval_response"] = [broken_plugin]
        try:
            approval_mod.register_gateway_notify(
                sk,
                lambda data: approval_mod.resolve_gateway_approval(
                    sk, "deny", tool_use_id=data["tool_use_id"],
                ),
            )
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: verdict)
            monkeypatch.setattr(relay_shared_metrics, "observe_lifecycle", lambda *_a, **_k: None)
            monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            with caplog.at_level(logging.DEBUG):
                assert self._call_gateway(
                    cb, tool_name="Read", tool_input={"file_path": path},
                    tool_use_id="toolu-plugin",
                ) == expected

            assert marker not in caplog.text
            assert sk not in caplog.text
            assert path not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
            assert [
                record.getMessage() for record in caplog.records
                if "SDK Smart" in record.getMessage()
            ] == expected_logs
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_sdk_smart_observer_redactor_failure_uses_fixed_pre_stage_log(
        self, monkeypatch, caplog,
    ):
        from agent import redact

        sk = "SDK_REDACTOR_SESSION_SECRET_65982"
        marker = "SDK_REDACTOR_EXCEPTION_SECRET_65982"
        path = "/tmp/SDK_REDACTOR_PATH_SECRET_65982"
        presentation = f"Read(path={path})"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        real_redact = redact.redact_sensitive_text

        def selective_redactor(value, *args, **kwargs):
            if value == presentation and kwargs.get("force") is True:
                raise RuntimeError(f"{marker} value={value}")
            return real_redact(value, *args, **kwargs)

        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: "approve")
            monkeypatch.setattr(redact, "redact_sensitive_text", selective_redactor)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            with caplog.at_level(logging.DEBUG):
                assert self._call_gateway(
                    cb, tool_name="Read", tool_input={"file_path": path},
                ) == "once"

            assert marker not in caplog.text
            assert sk not in caplog.text
            assert path not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
            assert [
                record.getMessage() for record in caplog.records
                if "SDK Smart" in record.getMessage()
            ] == ["SDK Smart pre-approval observer dispatch failed"]
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
    def test_generic_observer_failures_keep_legacy_raw_logging(
        self, monkeypatch, caplog,
    ):
        from agent import redact
        from hermes_cli import observability, plugins

        builtin_marker = "GENERIC_BUILTIN_RAW_COMPAT_65982"
        plugin_marker = "GENERIC_PLUGIN_RAW_COMPAT_65982"
        redactor_marker = "GENERIC_REDACTOR_RAW_COMPAT_65982"
        manager = plugins.PluginManager()

        def broken_plugin(**_kwargs):
            raise RuntimeError(plugin_marker)

        manager._hooks["pre_approval_request"] = [broken_plugin]
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
        from hermes_cli.observability import relay_shared_metrics

        monkeypatch.setattr(
            relay_shared_metrics, "observe_lifecycle",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError(builtin_marker)),
        )
        with caplog.at_level(logging.DEBUG):
            observability.observe_lifecycle("pre_approval_request", session_key="generic-session")
            manager.invoke_hook("pre_approval_request", session_key="generic-session")
            monkeypatch.setattr(
                redact,
                "redact_sensitive_text",
                lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError(redactor_marker)),
            )
            assert approval_smart._prepare_smart_approval_observer(
                command="generic command",
                description="generic description",
                pattern_key="generic",
                pattern_keys=["generic"],
                session_key="generic-session",
            ) is None

        assert builtin_marker in caplog.text
        assert plugin_marker in caplog.text
        assert redactor_marker in caplog.text
        assert any(record.exc_info for record in caplog.records)
        assert "SDK Smart" not in caplog.text
    def test_generic_shared_metrics_sink_keeps_legacy_raw_logging(
        self, monkeypatch, caplog,
    ):
        from agent import relay_runtime
        from hermes_cli.observability import relay_shared_metrics

        shared_marker = "GENERIC_SHARED_METRICS_RAW_COMPAT_65982"

        class BrokenSharedRuntime:
            def record_approval(self, kwargs):
                raise RuntimeError(f"{shared_marker} payload={kwargs!r}")

        monkeypatch.setattr(relay_shared_metrics, "handles_hook", lambda _name: True)
        monkeypatch.setattr(relay_runtime, "relay_instrumentation_enabled", lambda: True)
        monkeypatch.setattr(
            relay_shared_metrics, "_get_runtime", lambda: BrokenSharedRuntime(),
        )
        with caplog.at_level(logging.DEBUG):
            relay_shared_metrics.observe_lifecycle(
                "post_approval_response", session_key="generic-shared-session",
            )

        assert shared_marker in caplog.text
        assert any(record.exc_info for record in caplog.records)
        assert "SDK Smart" not in caplog.text
    def test_sdk_safe_and_generic_observer_dispatches_do_not_bleed_between_threads(
        self, monkeypatch, caplog,
    ):
        from concurrent.futures import ThreadPoolExecutor
        import threading

        from hermes_cli import plugins
        from hermes_cli.observability import relay_shared_metrics

        sdk_session = "SDK_CONCURRENT_SESSION_SECRET_65982"
        sdk_path = "/tmp/SDK_CONCURRENT_PATH_SECRET_65982"
        sdk_marker = "SDK_CONCURRENT_EXCEPTION_SECRET_65982"
        generic_session = "GENERIC_CONCURRENT_SESSION_COMPAT_65982"
        generic_marker = "GENERIC_CONCURRENT_EXCEPTION_COMPAT_65982"
        barrier = threading.Barrier(2)
        manager = plugins.PluginManager()

        def broken_plugin(**kwargs):
            barrier.wait(timeout=5)
            marker = sdk_marker if kwargs["session_key"] == sdk_session else generic_marker
            raise RuntimeError(f"{marker} payload={kwargs!r}")

        manager._hooks["pre_approval_request"] = [broken_plugin]
        manager._hooks["post_approval_response"] = [broken_plugin]
        monkeypatch.setattr(relay_shared_metrics, "observe_lifecycle", lambda *_a, **_k: None)
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

        def dispatch(session_key, command, fixed):
            from contextlib import nullcontext

            from hermes_cli.lifecycle import observer_failure_log

            pre = observer_failure_log(approval_ctx.SDK_PRE_OBSERVER_FAILURE_LOG) if fixed else nullcontext()
            post = observer_failure_log(approval_ctx.SDK_POST_OBSERVER_FAILURE_LOG) if fixed else nullcontext()
            with pre:
                payload = approval_smart._prepare_smart_approval_observer(
                    command=command,
                    description="bounded description",
                    pattern_key="claude_sdk_tool" if fixed else "generic",
                    pattern_keys=["claude_sdk_tool" if fixed else "generic"],
                    session_key=session_key,
                )
            with post:
                approval_smart._observe_smart_approval_verdict(payload, "approve")

        with caplog.at_level(logging.DEBUG), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(dispatch, sdk_session, f"Read(path={sdk_path})", True),
                pool.submit(dispatch, generic_session, "generic command", False),
            ]
            for future in futures:
                future.result(timeout=10)

        assert sdk_marker not in caplog.text
        assert sdk_session not in caplog.text
        assert sdk_path not in caplog.text
        assert generic_marker in caplog.text
        assert generic_session in caplog.text
        assert [
            record.getMessage() for record in caplog.records
            if "SDK Smart" in record.getMessage()
        ] == [
            "SDK Smart pre-approval observer dispatch failed",
            "SDK Smart post-approval observer dispatch failed",
        ]
    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [
            ("approve", "once"),
            ("deny", {"choice": "deny", "operator_denial": True, "reason": ""}),
            ("escalate", {"choice": "deny", "operator_denial": True, "reason": ""}),
        ],
    )
    def test_successful_real_sdk_observer_gets_bounded_fields_without_changing_decision(
        self, monkeypatch, verdict, expected,
    ):
        from hermes_cli import plugins
        from hermes_cli.observability import relay_shared_metrics

        sk = f"sdk-success-{verdict}"
        path = "/tmp/sdk-success-path"
        seen = []
        manager = plugins.PluginManager()
        manager._hooks["pre_approval_request"] = [
            lambda **kwargs: seen.append(("pre", kwargs))
        ]
        manager._hooks["post_approval_response"] = [
            lambda **kwargs: seen.append(("post", kwargs))
        ]
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            approval_mod.register_gateway_notify(
                sk,
                lambda _data: approval_mod.resolve_gateway_approval(sk, "deny"),
            )
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: verdict)
            monkeypatch.setattr(relay_shared_metrics, "observe_lifecycle", lambda *_a, **_k: None)
            monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert self._call_gateway(
                cb, tool_name="Read", tool_input={"file_path": path},
            ) == expected

            smart_seen = [item for item in seen if item[1].get("surface") == "smart"]
            expected_stages = ["pre"] if verdict == "escalate" else ["pre", "post"]
            assert [stage for stage, _kwargs in smart_seen] == expected_stages
            pre = smart_seen[0][1]
            assert pre["command"] == f"Read(path={path})"
            assert pre["description"] == "Claude requests SDK tool Read"
            assert pre["pattern_key"] == "claude_sdk_tool"
            assert pre["pattern_keys"] == ["claude_sdk_tool"]
            assert pre["session_key"] == sk
            assert pre["surface"] == "smart"
            assert {"turn_id", "tool_call_id", "telemetry_schema_version"} <= pre.keys()
            if verdict != "escalate":
                assert smart_seen[1][1]["choice"] == f"smart_{verdict}"
                assert smart_seen[1][1]["decided_by"] == "aux_llm"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)
