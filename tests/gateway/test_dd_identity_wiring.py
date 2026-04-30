"""Tests for DecisionData identity wiring on gateway runs.

Covers:
- ``_attach_dd_context_for_turn`` writes a fresh per-turn DD identity
  onto the (possibly cached/reused) AIAgent and returns the live
  context dict the gateway can inspect after ``run_conversation``.
- Reusing the helper on the same agent across turns overwrites the
  prior run's identity and resets the per-call emitted flag — a cached
  agent must never inherit the previous turn's DD context.
- ``_resolve_dd_session_key`` honors ``X-DD-Session-Key`` from the
  hermes-dispatcher and falls back to the derived
  ``agent:hermes:api_server:<session_id>`` only when the header is
  missing/blank.
"""

from __future__ import annotations

from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Phase 2 — gateway/run.py: _attach_dd_context_for_turn
# ---------------------------------------------------------------------------


class TestAttachDdContextForTurn:
    def _import_helper(self):
        from gateway.run import _attach_dd_context_for_turn
        return _attach_dd_context_for_turn

    def test_sets_dd_attributes_on_agent(self):
        attach = self._import_helper()
        agent = SimpleNamespace()

        ctx = attach(
            agent,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
            parent_run_id=None,
            wts_task_id=None,
        )

        assert agent._dd_run_id == "hermes-sess-g1"
        assert agent._dd_session_key == "agent:hermes:gateway:sess"
        assert agent._dd_parent_run_id is None
        assert agent._dd_wts_task_id is None
        assert agent._dd_context is ctx

    def test_returns_fresh_per_turn_context_dict(self):
        attach = self._import_helper()
        agent = SimpleNamespace()

        ctx = attach(
            agent,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
        )

        assert ctx["run_id"] == "hermes-sess-g1"
        assert ctx["session_key"] == "agent:hermes:gateway:sess"
        assert ctx["agent_class"] == "Hermes"
        assert ctx["ingest_source"] == "hermes-gateway"
        assert ctx["per_call_generation_emitted"] is False

    def test_optional_parent_and_wts_task_ids_propagate(self):
        attach = self._import_helper()
        agent = SimpleNamespace()

        ctx = attach(
            agent,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
            parent_run_id="parent-run-xyz",
            wts_task_id="wts-task-abc",
        )

        assert ctx["parent_run_id"] == "parent-run-xyz"
        assert ctx["wts_task_id"] == "wts-task-abc"
        assert agent._dd_parent_run_id == "parent-run-xyz"
        assert agent._dd_wts_task_id == "wts-task-abc"

    def test_reused_agent_gets_fresh_context_each_turn(self):
        """A cached AIAgent reused across turns must NOT inherit the
        previous turn's DD identity or the previous turn's
        per_call_generation_emitted flag — that would either misattribute
        new generations or silently skip the synthetic completion fallback
        on a turn that never emitted per-call rows.
        """
        attach = self._import_helper()
        agent = SimpleNamespace()

        ctx_turn_1 = attach(
            agent,
            run_id="hermes-sess-g1",
            session_key="agent:hermes:gateway:sess",
        )
        ctx_turn_1["per_call_generation_emitted"] = True

        ctx_turn_2 = attach(
            agent,
            run_id="hermes-sess-g2",
            session_key="agent:hermes:gateway:sess",
        )

        assert ctx_turn_2 is not ctx_turn_1
        assert ctx_turn_2["per_call_generation_emitted"] is False
        assert ctx_turn_2["run_id"] == "hermes-sess-g2"
        assert agent._dd_run_id == "hermes-sess-g2"
        # The prior turn's context dict must remain untouched (a still-
        # in-flight per-call logger from turn 1 might still hold a
        # reference and write to it).
        assert ctx_turn_1["run_id"] == "hermes-sess-g1"
        assert ctx_turn_1["per_call_generation_emitted"] is True


# ---------------------------------------------------------------------------
# Phase 7 — api_server.py: _resolve_dd_session_key
# ---------------------------------------------------------------------------


class TestResolveDdSessionKey:
    def _import_helper(self):
        from gateway.platforms.api_server import _resolve_dd_session_key
        return _resolve_dd_session_key

    def test_uses_header_value_when_provided(self):
        resolve = self._import_helper()
        headers = {"X-DD-Session-Key": "agent:hermes:gateway:dispatcher-sess"}

        result = resolve(headers, "abc123")

        assert result == "agent:hermes:gateway:dispatcher-sess"

    def test_falls_back_to_derived_when_header_absent(self):
        resolve = self._import_helper()
        headers: dict[str, str] = {}

        result = resolve(headers, "abc123")

        assert result == "agent:hermes:api_server:abc123"

    def test_falls_back_to_derived_when_header_blank(self):
        resolve = self._import_helper()
        headers = {"X-DD-Session-Key": "   "}

        result = resolve(headers, "abc123")

        assert result == "agent:hermes:api_server:abc123"

    def test_strips_surrounding_whitespace_on_header(self):
        resolve = self._import_helper()
        headers = {"X-DD-Session-Key": "  agent:hermes:gateway:dispatcher-sess  "}

        result = resolve(headers, "abc123")

        assert result == "agent:hermes:gateway:dispatcher-sess"
