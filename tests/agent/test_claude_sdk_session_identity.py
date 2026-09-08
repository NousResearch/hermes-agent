"""Session identity and continuity — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

from unittest.mock import MagicMock

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from tests.agent.claude_sdk_fakes import (
    ResultMessage,
    _make_session,
    _make_turn,
    _make_agent,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestHermesSessionIdPlumbing:
    def test_session_id_rides_mcp_env(self):
        session, holder = _make_session(
            script=[ResultMessage(result="ok")], hermes_session_id="sess-42"
        )
        try:
            session.run_turn("ping")
        finally:
            session.close()
        env = holder["client"].options["mcp_servers"]["hermes-tools"]["env"]
        assert env["HERMES_SESSION_ID"] == "sess-42"
        # The invented pre-fix name must never come back: the shim consumer
        # reads only the canonical HERMES_SESSION_ID.
        assert "HERMES_MCP_SESSION_ID" not in env

    def test_no_session_id_no_env_var(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        env = holder["client"].options["mcp_servers"]["hermes-tools"]["env"]
        assert "HERMES_SESSION_ID" not in env
        assert "HERMES_MCP_SESSION_ID" not in env

    def test_runtime_passes_agent_session_id(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        assert captured.get("hermes_session_id") == "sess-1"

    def test_runtime_passes_context_to_append_builder(self, monkeypatch):
        # W2: the append builder receives the agent's platform/session/model
        # so the session line and platform hint reflect the live session. The
        # SDK process uses the validated runtime cwd, while prompt discovery
        # keeps None as the native fallback sentinel so the install-tree guard
        # can distinguish fallback from an operator-selected directory.
        import agent.claude_sdk_runtime_session as rt
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        captured = {}
        session_captured = {}

        def fake_append(**kwargs):
            captured.update(kwargs)
            return "APPEND-UNDER-TEST"

        class SpySession:
            def __init__(self, **kwargs):
                session_captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(rt, "build_system_prompt_append", fake_append)
        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        import agent.runtime_cwd as runtime_cwd

        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: "/resolved-workspace")
        monkeypatch.setattr(runtime_cwd, "resolve_context_cwd", lambda: None)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent.session_cwd = "/unvalidated-stale-workspace"
        agent.skip_context_files = True
        agent.platform = "telegram"
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        # Per-key pins (not whole-dict equality): the contract is that the
        # builder receives the live session's platform/session/model — a new
        # kwarg added later must not break these unrelated assertions.
        assert captured["platform"] == "telegram"
        assert captured["session_id"] == "sess-1"
        assert captured["model"] == "claude-opus-4-8"
        assert captured["cwd"] is None
        assert captured["include_project_context"] is False
        assert session_captured["cwd"] == "/resolved-workspace"

        # A validated, explicitly configured context cwd is forwarded as an
        # explicit path and project files return to their default-on posture.
        captured.clear()
        session_captured.clear()
        agent._claude_sdk_session = None
        agent.skip_context_files = False
        monkeypatch.setattr(
            runtime_cwd,
            "resolve_context_cwd",
            lambda: "/configured-workspace",
        )
        run_claude_agent_sdk_turn(
            agent,
            user_message="again",
            original_user_message="again",
            messages=[{"role": "user", "content": "again"}],
            effective_task_id="task-2",
        )
        assert captured["cwd"] == "/configured-workspace"
        assert captured["include_project_context"] is True
        assert session_captured["cwd"] == "/resolved-workspace"

    @staticmethod
    def _run_with_spy_session(monkeypatch, config_block):
        """Drive one runtime turn with a kwargs-capturing session and the
        given agent.claude_agent_sdk config block; returns captured kwargs."""
        import agent.claude_sdk_runtime_session as rt
        import agent.transports.claude_agent_sdk_session as sdk_session_mod
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": config_block}},
            raising=False,
        )
        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(rt, "build_system_prompt_append", lambda **k: None)
        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        return captured

    def test_max_budget_usd_config_reaches_the_session(self, monkeypatch):
        captured = self._run_with_spy_session(
            monkeypatch, {"max_budget_usd": 2.5}
        )
        assert captured["max_budget_usd"] == 2.5

    def test_max_budget_usd_default_is_no_budget(self, monkeypatch):
        captured = self._run_with_spy_session(monkeypatch, {})
        assert captured["max_budget_usd"] is None

    def test_max_budget_usd_invalid_values_ignored(self, monkeypatch):
        # A typo or a nonsense cap (0 would fail every turn instantly) must
        # never become a silent behavior change — no budget is passed.
        for bad in ("not-a-number", 0, -3, True):
            captured = self._run_with_spy_session(
                monkeypatch, {"max_budget_usd": bad}
            )
            assert captured["max_budget_usd"] is None, bad


# ---------- continuity: resume + digest fallback (W3) ----------


class TestContinuity:
    """Retire matrix under test:
      /new, expiry      → new Hermes session row → no persisted id → FRESH
      restart/eviction  → same row, id persisted → RESUME
      error retire      → persisted id CLEARED → next turn fresh + digest
      stale resume      → retire → clear → ONE fresh retry with digest
    """

    @staticmethod
    def _bound_id(session_id, cwd=None):
        from agent.claude_sdk_runtime_continuity import _encode_sdk_resume_binding

        return _encode_sdk_resume_binding(session_id, cwd=cwd)

    @staticmethod
    def _db_agent(persisted_sdk_id=None):
        from agent.runtime_cwd import resolve_agent_cwd

        agent = _make_agent()
        agent._claude_sdk_session = None
        db = MagicMock()
        db.get_session.return_value = {
            "claude_sdk_session_id": persisted_sdk_id,
            "cwd": str(resolve_agent_cwd()),
        }
        agent._session_db = db
        agent._session_db_created = True
        return agent, db

    @staticmethod
    def _spy_sessions(monkeypatch, behaviors):
        """Install a SpySession whose Nth instance behaves per behaviors[N]:
        a TurnResult-like object to return, or an Exception to raise."""
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        instances = []

        class SpySession:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.inputs = []
                instances.append(self)

            def run_turn(self, user_input):
                self.inputs.append(user_input)
                behavior = behaviors[len(instances) - 1]
                if isinstance(behavior, Exception):
                    raise behavior
                return behavior

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        return instances

    def test_creation_resumes_from_persisted_id(self, monkeypatch):
        agent, _db = self._db_agent(
            persisted_sdk_id=self._bound_id("sdk-old-1")
        )
        instances = self._spy_sessions(monkeypatch, [_make_turn()])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert instances[0].kwargs.get("resume_session_id") == "sdk-old-1"
        # A resumed session already holds the context — no digest.
        assert instances[0].inputs == ["hi"]

    def test_bound_resume_same_cwd_uses_embedded_sdk_id(self, monkeypatch):
        import agent.runtime_cwd as runtime_cwd

        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: "/workspace/a")
        binding = (
            'hermes-sdk-resume-v1:{"cwd":"/workspace/a","id":"sdk-bound-1"}'
        )
        agent, _db = self._db_agent(persisted_sdk_id=binding)
        instances = self._spy_sessions(monkeypatch, [_make_turn()])

        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        assert instances[0].kwargs.get("resume_session_id") == "sdk-bound-1"
        assert instances[0].kwargs.get("cwd") == "/workspace/a"

    def test_bound_resume_changed_cwd_starts_fresh_without_destroying_old_binding(
        self, monkeypatch
    ):
        import agent.runtime_cwd as runtime_cwd

        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: "/workspace/b")
        binding = (
            'hermes-sdk-resume-v1:{"cwd":"/workspace/a","id":"sdk-bound-1"}'
        )
        agent, db = self._db_agent(persisted_sdk_id=binding)
        instances = self._spy_sessions(monkeypatch, [_make_turn()])
        messages = [
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
            {"role": "user", "content": "continue"},
        ]

        run_claude_agent_sdk_turn(
            agent,
            user_message="continue",
            original_user_message="continue",
            messages=messages,
            effective_task_id="t",
        )

        assert instances[0].kwargs.get("resume_session_id") is None
        assert instances[0].kwargs.get("cwd") == "/workspace/b"
        assert instances[0].inputs[0].startswith("[Continuity digest")
        writes = [call.args for call in db.update_claude_sdk_session_id.call_args_list]
        assert ("sess-1", None) not in writes
        assert writes[-1] == (
            "sess-1",
            'hermes-sdk-resume-v1:{"cwd":"/workspace/b","id":"sdk-session-1"}',
        )

    def test_legacy_resume_matching_mutable_row_cwd_starts_fresh(self, monkeypatch):
        import agent.runtime_cwd as runtime_cwd

        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: "/workspace/a")
        agent, db = self._db_agent(persisted_sdk_id="sdk-legacy-1")
        db.get_session.return_value["cwd"] = "/workspace/a"
        instances = self._spy_sessions(monkeypatch, [_make_turn()])

        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        assert instances[0].kwargs.get("resume_session_id") is None
        db.update_claude_sdk_session_id.assert_any_call("sess-1", None)

    def test_live_session_is_retired_when_workspace_changes(self, monkeypatch):
        import agent.runtime_cwd as runtime_cwd

        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: "/workspace/b")
        agent, db = self._db_agent()
        live = MagicMock()
        live._cwd = "/workspace/a"
        live.run_turn.return_value = _make_turn(thread_id="sdk-old-workspace")
        agent._claude_sdk_session = live
        instances = self._spy_sessions(
            monkeypatch, [_make_turn(thread_id="sdk-new-workspace")]
        )
        messages = [
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
            {"role": "user", "content": "continue in b"},
        ]

        run_claude_agent_sdk_turn(
            agent,
            user_message="continue in b",
            original_user_message="continue in b",
            messages=messages,
            effective_task_id="t",
        )

        live.close.assert_called_once()
        live.run_turn.assert_not_called()
        assert len(instances) == 1
        assert instances[0].kwargs.get("cwd") == "/workspace/b"
        assert instances[0].kwargs.get("resume_session_id") is None
        assert instances[0].inputs[0].startswith("[Continuity digest")
        db.update_claude_sdk_session_id.assert_called_with(
            "sess-1",
            'hermes-sdk-resume-v1:{"cwd":"/workspace/b","id":"sdk-new-workspace"}',
        )

    @pytest.mark.parametrize(
        "stored_value,row_cwd",
        [
            ("hermes-sdk-resume-v1:{not-json", "/workspace/a"),
            ("sdk-legacy-1", "/workspace/other"),
        ],
    )
    def test_invalid_or_mismatched_legacy_resume_clears_and_starts_fresh(
        self, monkeypatch, stored_value, row_cwd
    ):
        import agent.runtime_cwd as runtime_cwd

        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: "/workspace/a")
        agent, db = self._db_agent(persisted_sdk_id=stored_value)
        db.get_session.return_value["cwd"] = row_cwd
        instances = self._spy_sessions(monkeypatch, [_make_turn()])

        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        assert instances[0].kwargs.get("resume_session_id") is None
        assert ("sess-1", None) in [
            call.args for call in db.update_claude_sdk_session_id.call_args_list
        ]

    def test_successful_turn_persists_cwd_bound_resume_envelope(self, monkeypatch):
        import agent.runtime_cwd as runtime_cwd

        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: "/workspace/a")
        agent, db = self._db_agent()
        self._spy_sessions(monkeypatch, [_make_turn(thread_id="sdk-new-9")])

        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        db.update_claude_sdk_session_id.assert_called_with(
            "sess-1",
            'hermes-sdk-resume-v1:{"cwd":"/workspace/a","id":"sdk-new-9"}',
        )

    def test_unplaceable_id_is_not_written_as_a_binding(self):
        """The write point refuses to invent a workspace.

        Callers sample the workspace; this is the backstop for one that could
        not. Encoding here would canonicalise the persist-time cwd into the
        envelope, producing a binding that always matches itself. Nothing is
        written -- a caller asking to save must not have that turned into a
        delete of a binding an earlier, placeable turn wrote.
        """
        from agent.claude_sdk_runtime_continuity import _store_sdk_session_id

        agent, db = self._db_agent()

        _store_sdk_session_id(agent, "sdk-unplaceable", cwd=None)

        db.update_claude_sdk_session_id.assert_not_called()

    def test_workspace_is_sampled_at_turn_start_not_at_persist(self, monkeypatch):
        """The binding records where the turn ran, not where persist lands.

        The live session is the authority for the turn workspace, but it is not
        always present at that point. The fallback has to be resolved at turn
        start: persist runs after the turn, so resolving there would sample a
        workspace that had already moved, stamp the post-move path onto the id,
        and the binding would then match itself on the next turn -- the check
        would never be able to fail.
        """
        import agent.runtime_cwd as runtime_cwd
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent, db = self._db_agent()
        live = {"cwd": "/workspace/before"}
        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: live["cwd"])

        class MovingWorkspaceSession:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run_turn(self, user_input):
                live["cwd"] = "/workspace/after"
                return _make_turn(thread_id="sdk-new-9")

            def close(self):
                pass

        monkeypatch.setattr(
            sdk_session_mod, "ClaudeAgentSdkSession", MovingWorkspaceSession
        )

        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        db.update_claude_sdk_session_id.assert_called_with(
            "sess-1",
            'hermes-sdk-resume-v1:{"cwd":"/workspace/before","id":"sdk-new-9"}',
        )

    def test_successful_turn_persists_thread_id(self, monkeypatch):
        from agent.claude_sdk_runtime_continuity import _encode_sdk_resume_binding

        agent, db = self._db_agent()
        self._spy_sessions(monkeypatch, [_make_turn(thread_id="sdk-new-9")])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        db.update_claude_sdk_session_id.assert_called_with(
            "sess-1", _encode_sdk_resume_binding("sdk-new-9")
        )

    def test_error_retire_clears_persisted_id(self, monkeypatch):
        agent, db = self._db_agent()
        self._spy_sessions(monkeypatch, [_make_turn(
            should_retire=True, error="turn timed out", projected_messages=[],
            final_text="", token_usage_last=None,
        )])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        db.update_claude_sdk_session_id.assert_called_with("sess-1", None)

    def test_eligible_handoff_keeps_failed_session_id_durably_cleared(
        self, monkeypatch
    ):
        agent, db = self._db_agent(
            persisted_sdk_id=self._bound_id("sdk-failed-8")
        )
        self._spy_sessions(monkeypatch, [_make_turn(
            thread_id="sdk-failed-8",
            should_retire=False,
            error="HTTP 503 service overloaded",
            projected_messages=[],
            tool_iterations=0,
            final_text="",
            token_usage_last=None,
        )])

        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        assert result["failover_reason"] == "overloaded"
        writes = db.update_claude_sdk_session_id.call_args_list
        assert writes[-1].args == ("sess-1", None)
        assert all(call.args != ("sess-1", "sdk-failed-8") for call in writes)

    def test_digest_prepended_on_fresh_session_with_history(self, monkeypatch):
        agent, _db = self._db_agent(persisted_sdk_id=None)
        instances = self._spy_sessions(monkeypatch, [_make_turn()])
        messages = [
            {"role": "user", "content": "the linter flags shadowed imports"},
            {"role": "assistant", "content": "Fixed by renaming the local."},
            {"role": "user", "content": "and the tests?"},
        ]
        run_claude_agent_sdk_turn(
            agent, user_message="and the tests?", original_user_message="and the tests?",
            messages=messages, effective_task_id="t",
        )
        sent = instances[0].inputs[0]
        assert sent.startswith("[Continuity digest")
        assert "shadowed imports" in sent
        assert sent.endswith("and the tests?")

    def test_projected_bg_row_excluded_from_continuity_digest(self):
        # Binding amendment (sdk-echo-approval-fixes): rows projected by the
        # background-result lane are the agent's OWN delivered answers —
        # the digest re-presenting them is double-presentation, the exact
        # pathology the lane fixes. Marked rows never enter the digest.
        from agent.claude_sdk_runtime_continuity import _render_continuity_digest

        digest = _render_continuity_digest([
            {"role": "user", "content": "run the research"},
            {
                "role": "assistant",
                "content": "the full background report",
                "display_kind": "sdk_background_result",
            },
            {"role": "assistant", "content": "a normal reply"},
        ])
        assert "the full background report" not in digest
        assert "run the research" in digest
        assert "a normal reply" in digest

    def test_no_digest_on_brand_new_conversation(self, monkeypatch):
        agent, _db = self._db_agent(persisted_sdk_id=None)
        instances = self._spy_sessions(monkeypatch, [_make_turn()])
        run_claude_agent_sdk_turn(
            agent, user_message="hello", original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}], effective_task_id="t",
        )
        assert instances[0].inputs == ["hello"]

    def test_stale_resume_retires_then_retries_fresh_with_digest(self, monkeypatch):
        # The Pi probe: a stale resume id fails the session. The runtime
        # must clear the id and retry ONCE fresh (digest included) — the
        # user gets an answer, not an error.
        agent, db = self._db_agent(
            persisted_sdk_id=self._bound_id("sdk-stale-7")
        )
        instances = self._spy_sessions(monkeypatch, [
            _make_turn(should_retire=True, error="resume failed",
                       projected_messages=[], final_text="", token_usage_last=None),
            _make_turn(final_text="fresh answer",
                       projected_messages=[{"role": "assistant", "content": "fresh answer"}]),
        ])
        messages = [
            {"role": "user", "content": "earlier context line"},
            {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "current question"},
        ]
        result = run_claude_agent_sdk_turn(
            agent, user_message="current question",
            original_user_message="current question",
            messages=messages, effective_task_id="t",
        )
        assert result["final_response"] == "fresh answer"
        assert len(instances) == 2
        assert instances[0].kwargs.get("resume_session_id") == "sdk-stale-7"
        assert instances[1].kwargs.get("resume_session_id") is None
        assert instances[1].inputs[0].startswith("[Continuity digest")
        db.update_claude_sdk_session_id.assert_any_call("sess-1", None)

    def test_cold_short_circuit_consumes_live_session_event_too(self, monkeypatch):
        # Validator C1: an interrupt racing turn completion sets BOTH the
        # agent flag and the live session's event. The short-circuit consumed
        # only the flag — the NEXT legit message then died on the stale
        # session event with no model call. Honoring must consume both.
        agent, _db = self._db_agent()
        live = MagicMock()
        agent._claude_sdk_session = live
        agent._interrupt_requested = True
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["partial"] is True
        live.consume_interrupt.assert_called_once()
        live.run_turn.assert_not_called()

    def test_resume_id_persisted_after_flush_and_gated_on_persist_disabled(self, monkeypatch):
        # Validator C9: the resume-id UPDATE ran BEFORE the flush that
        # (re)creates the session row after a transient turn-start lock —
        # silently discarding continuity. Order must be flush-then-store.
        agent, db = self._db_agent()
        order = []
        agent._flush_messages_to_session_db = MagicMock(
            side_effect=lambda *a, **k: order.append("flush"))
        db.update_claude_sdk_session_id.side_effect = (
            lambda *a, **k: order.append("store"))
        self._spy_sessions(monkeypatch, [_make_turn(thread_id="sdk-z-1")])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert "store" in order and "flush" in order
        assert order.index("flush") < order.index("store")
        # And a fork with persistence disabled must never touch the parent row.
        agent2, db2 = self._db_agent(persisted_sdk_id="sdk-parent-1")
        agent2._persist_disabled = True
        self._spy_sessions(monkeypatch, [_make_turn(thread_id="sdk-fork-9")])
        run_claude_agent_sdk_turn(
            agent2, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        db2.update_claude_sdk_session_id.assert_not_called()

    def test_interrupted_turn_retires_client_but_persists_resume_id(self, monkeypatch):
        # Adversarial-review HIGH: breaking out of receive_response() on
        # interrupt leaves the interrupted turn's ResultMessage queued in the
        # client's stream — a REUSED client would serve it as the NEXT turn's
        # answer. The runtime must retire the client (clean stream) while
        # persisting the SDK id, so the next turn RESUMES the conversation.
        from agent.claude_sdk_runtime_continuity import _encode_sdk_resume_binding

        agent, db = self._db_agent()
        self._spy_sessions(monkeypatch, [_make_turn(
            interrupted=True, final_text="partial answer", thread_id="sdk-live-3",
        )])
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert agent._claude_sdk_session is None  # client retired
        db.update_claude_sdk_session_id.assert_called_with(
            "sess-1", _encode_sdk_resume_binding("sdk-live-3")
        )
        assert result["partial"] is True

    def test_fresh_retire_does_not_retry(self, monkeypatch):
        # Only a RESUMED session earns the retry — a fresh session that
        # retires is a real error and must surface, never loop.
        agent, _db = self._db_agent(persisted_sdk_id=None)
        instances = self._spy_sessions(monkeypatch, [_make_turn(
            should_retire=True, error="boom", projected_messages=[],
            final_text="", token_usage_last=None,
        )])
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert len(instances) == 1
        assert result["partial"] is True

    def test_interrupted_retire_does_not_retry(self, monkeypatch):
        # A RESUMED session that retires on an INTERRUPTED turn (a /stop
        # that killed the CLI, a hard watchdog trip) must NOT re-run the
        # turn — the retry would evaporate the stop and deliver the answer
        # anyway. Only non-interrupted resume failures earn the retry.
        agent, _db = self._db_agent(
            persisted_sdk_id=self._bound_id("sdk-live-1")
        )
        instances = self._spy_sessions(monkeypatch, [_make_turn(
            should_retire=True, interrupted=True,
            error="SDK message stream ended before this turn's result",
            projected_messages=[], final_text="", token_usage_last=None,
        )])
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert len(instances) == 1  # no second full-budget run
        assert result["partial"] is True

    def test_late_stop_after_terminal_retire_does_not_retry(self, monkeypatch):
        agent, _db = self._db_agent(
            persisted_sdk_id=self._bound_id("sdk-live-1")
        )
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        instances = []

        class LateStopRetireSession:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.inputs = []
                instances.append(self)

            def run_turn(self, user_input):
                self.inputs.append(user_input)
                agent._interrupt_requested = True
                return _make_turn(
                    should_retire=True,
                    terminal_result_accepted=True,
                    error="SDK result error (subtype=error): session retired",
                    api_error_status=500,
                    projected_messages=[],
                    final_text="",
                    token_usage_last=None,
                )

            def consume_interrupt(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(
            sdk_session_mod, "ClaudeAgentSdkSession", LateStopRetireSession
        )
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        assert len(instances) == 1
        assert instances[0].inputs == ["hi"]
        assert result["interrupted"] is True
        assert result["failed"] is False
        assert result.get("failover_reason") is None

    def test_raising_resumed_turn_with_stop_does_not_retry(self, monkeypatch):
        agent, _db = self._db_agent(
            persisted_sdk_id=self._bound_id("sdk-live-1")
        )
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        instances = []

        class RaisingStoppedSession:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.inputs = []
                instances.append(self)

            def run_turn(self, user_input):
                self.inputs.append(user_input)
                agent._interrupt_requested = True
                raise RuntimeError("resumed SDK transport exploded")

            def close(self):
                pass

        monkeypatch.setattr(
            sdk_session_mod, "ClaudeAgentSdkSession", RaisingStoppedSession
        )
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        assert len(instances) == 1
        assert instances[0].inputs == ["hi"]
        assert result["interrupted"] is True
        assert result["failed"] is False
        assert agent._interrupt_requested is False

    def test_stop_admitted_during_raising_close_does_not_replay(self, monkeypatch):
        # Same promise as the test above, but the /stop lands one window
        # later: INSIDE close(), after the interrupt flag was sampled and
        # before the retry decision reads it. Tearing the transport down is
        # not instantaneous, so this is a real window, and a stop admitted
        # in it is still a stop against this turn. Deciding the retry on the
        # stale pre-close sample replays the prompt the user just asked to
        # abandon, destroys the resume binding on the way, and leaves the
        # agent-level flag set to reject the next message.
        import agent.runtime_cwd as runtime_cwd

        # A literal envelope rather than self._bound_id: this case only needs
        # a resumable id, and spelling it out keeps the test independent of
        # the binding helper.
        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: "/workspace/a")
        agent, db = self._db_agent(
            persisted_sdk_id='hermes-sdk-resume-v1:{"cwd":"/workspace/a","id":"sdk-live-1"}'
        )
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        instances = []

        class StopDuringCloseSession:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.inputs = []
                instances.append(self)

            def run_turn(self, user_input):
                self.inputs.append(user_input)
                raise RuntimeError("resumed SDK transport exploded")

            def close(self):
                # The stop is admitted while the transport is being reaped.
                agent._interrupt_requested = True

        monkeypatch.setattr(
            sdk_session_mod, "ClaudeAgentSdkSession", StopDuringCloseSession
        )
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        # No fresh-resume replay of the abandoned prompt.
        assert len(instances) == 1
        assert instances[0].inputs == ["hi"]
        # The replay path is what clears the binding; it must not have run.
        assert db.update_claude_sdk_session_id.call_args_list == []
        assert result["interrupted"] is True
        assert result["failed"] is False
        assert agent._interrupt_requested is False

    def test_pre_turn_interrupt_short_circuit_reports_interrupted(self):
        # The top-of-turn short-circuit consumes a pre-turn /stop without a
        # model call. Its result dict must carry interrupted=True — without
        # the key the gateway's empty-response normalizer has NO branch to
        # take (api_calls 0, partial True) and the user's message dies in
        # total silence.
        agent, _db = self._db_agent()
        agent._claude_sdk_session = None
        agent._interrupt_requested = True
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["interrupted"] is True
        assert result["api_calls"] == 0
        assert agent._interrupt_requested is False  # consumed

    def test_effective_prompt_snapshot_replaces_native_one(self, monkeypatch):
        # The prologue persists Hermes' native composed prompt — a prompt
        # this runtime never sends. The runtime overwrites the snapshot with
        # the EFFECTIVE prompt so the audit trail tells the truth.
        agent, db = self._db_agent()
        self._spy_sessions(monkeypatch, [_make_turn()])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        args = db.update_system_prompt.call_args
        assert args is not None
        assert args.args[0] == "sess-1"
        assert args.args[1].startswith("[claude_code preset]")


    def test_reuse_turn_survives_an_unresolvable_workspace(self, monkeypatch):
        """The workspace fence is an optimisation, not a safety gate.

        ``resolve_agent_cwd()`` raises for real -- a deleted launch directory,
        or a refusal terminal scope -- and the session-reuse path never
        resolved a cwd before the fence existed. An unresolvable workspace must
        leave the live session alone, not kill the turn; the resume binding
        still declines a foreign session on its own.
        """
        import agent.runtime_cwd as runtime_cwd

        def _deleted_launch_dir():
            raise FileNotFoundError("launch directory was removed")

        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", _deleted_launch_dir)

        agent = _make_agent()
        agent._claude_sdk_session._cwd = "/workspace/a"
        agent._claude_sdk_session.run_turn.return_value = _make_turn()

        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )

        assert result["final_response"] == "SDK_ASSISTANT"
        assert agent._claude_sdk_session is not None


class TestSessionResumeField:
    def test_resume_rides_options_when_set(self):
        session, holder = _make_session(
            script=[ResultMessage(result="ok")], resume_session_id="sdk-abc"
        )
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["resume"] == "sdk-abc"

    def test_no_resume_field_when_unset(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert "resume" not in holder["client"].options


# ---------- agent close() releases the SDK session ----------


class TestAgentCloseClosesSdkSession:
    """AIAgent.close() runs on /new, session expiry, and agent-cache
    eviction. Without an explicit disconnect the SDK client (and its CLI
    subprocess) is dropped to GC — a leak. (#25267)"""

    @staticmethod
    def _make_real_agent():
        from run_agent import AIAgent

        return AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    def test_close_disconnects_claude_sdk_session(self):
        agent = self._make_real_agent()
        sdk_session = MagicMock()
        agent._claude_sdk_session = sdk_session
        agent.close()
        sdk_session.close.assert_called_once()
        assert agent._claude_sdk_session is None

    def test_close_without_sdk_session_stays_safe(self):
        # Negative control: an agent that never created an SDK session (or
        # already closed it) must close without raising — idempotency.
        agent = self._make_real_agent()
        agent.close()
        agent._claude_sdk_session = None
        agent.close()
