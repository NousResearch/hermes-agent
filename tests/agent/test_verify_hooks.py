"""Unit tests for the verification-loop policy (agent/verify_hooks.py).

The `pre_verify` user-hook aggregation lives in `hermes_cli.plugins`
(`get_pre_verify_continue_message`) and is tested in
`tests/hermes_cli/test_plugins.py`, alongside `get_pre_tool_call_block_message`.
"""

from __future__ import annotations

from agent import verify_hooks


class TestMaxVerifyNudges:
    def test_default_when_unset(self):
        assert (
            verify_hooks.max_verify_nudges({})
            == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES
        )
        assert (
            verify_hooks.max_verify_nudges({"agent": {}})
            == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES
        )

    def test_reads_and_coerces(self):
        assert verify_hooks.max_verify_nudges({"agent": {"max_verify_nudges": 5}}) == 5
        assert verify_hooks.max_verify_nudges({"agent": {"max_verify_nudges": "2"}}) == 2
        assert verify_hooks.max_verify_nudges({"agent": {"max_verify_nudges": -1}}) == 0

    def test_bad_value_falls_back(self):
        assert (
            verify_hooks.max_verify_nudges({"agent": {"max_verify_nudges": "x"}})
            == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES
        )


class TestCodingVerifyGuidance:
    def test_enabled_by_default(self):
        assert (
            verify_hooks.coding_verify_guidance({})
            == verify_hooks.CODING_VERIFY_GUIDANCE
        )
        assert (
            verify_hooks.coding_verify_guidance({"agent": {}})
            == verify_hooks.CODING_VERIFY_GUIDANCE
        )

    def test_reads_truthy_config(self):
        cfg = {"agent": {"verify_guidance": "yes"}}
        assert verify_hooks.coding_verify_guidance(cfg) == verify_hooks.CODING_VERIFY_GUIDANCE

    def test_opt_out_via_config(self):
        off = {"agent": {"verify_guidance": False}}
        assert verify_hooks.coding_verify_guidance(off) is None


class TestPreVerifyOnNoEditTurns:
    """The no-edit gate is opt-in: default off, config-driven, never env-driven."""

    def test_off_by_default(self):
        assert verify_hooks.pre_verify_on_no_edit_turns({}) is False
        assert verify_hooks.pre_verify_on_no_edit_turns({"agent": {}}) is False

    def test_reads_truthy_config(self):
        for raw in (True, "true", "yes", 1):
            cfg = {"agent": {"pre_verify_on_no_edit_turns": raw}}
            assert verify_hooks.pre_verify_on_no_edit_turns(cfg) is True, raw

    def test_falsey_and_garbage_stay_off(self):
        for raw in (False, "false", "no", 0, "banana", None):
            cfg = {"agent": {"pre_verify_on_no_edit_turns": raw}}
            assert verify_hooks.pre_verify_on_no_edit_turns(cfg) is False, raw

    def test_config_default_ships_disabled(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["agent"]["pre_verify_on_no_edit_turns"] is False


class TestEnforcedPreVerifyVerdict:
    """The enforced half of the ``pre_verify`` directive (delivered-answer contract).

    Behaviour, not shape: what a caller receives back for a given pending
    verdict. ``agent/turn_finalizer.py`` applies this AFTER the output
    transforms, which is what makes the outcome transform-order independent.
    """

    class _Agent:
        pass

    def test_inert_without_a_pending_verdict(self):
        a = self._Agent()
        assert verify_hooks.apply_pre_verify_verdict(a, "plain answer") == "plain answer"
        assert verify_hooks.apply_pre_verify_verdict(a, "") == ""
        assert verify_hooks.apply_pre_verify_verdict(a, None) is None

    def test_non_string_attribute_can_never_inject_text(self):
        a = self._Agent()
        a._pre_verify_final_verdict = object()
        assert verify_hooks.apply_pre_verify_verdict(a, "answer") == "answer"

    def test_pending_verdict_leads_the_delivered_answer(self):
        a = self._Agent()
        verify_hooks.record_pre_verify_verdict(a, "VERDICT: work remains")
        out = verify_hooks.apply_pre_verify_verdict(a, "all good, nothing to report")
        assert out.startswith("VERDICT: work remains")
        assert "all good, nothing to report" in out

    def test_applied_exactly_once_and_deduplicated(self):
        a = self._Agent()
        verify_hooks.record_pre_verify_verdict(a, "VERDICT: work remains")
        # Already present verbatim (e.g. a plugin transform emitted it): untouched.
        same = verify_hooks.apply_pre_verify_verdict(a, "x VERDICT: work remains y")
        assert same == "x VERDICT: work remains y"
        # Consumed: a second call in the same/next turn adds nothing.
        assert verify_hooks.apply_pre_verify_verdict(a, "later") == "later"

    def test_empty_answer_becomes_the_verdict(self):
        a = self._Agent()
        verify_hooks.record_pre_verify_verdict(a, "VERDICT: work remains")
        assert verify_hooks.apply_pre_verify_verdict(a, "   ") == "VERDICT: work remains"

    def test_interrupt_wins(self):
        a = self._Agent()
        verify_hooks.record_pre_verify_verdict(a, "VERDICT: work remains")
        assert verify_hooks.apply_pre_verify_verdict(
            a, "partial", interrupted=True) == "partial"
        # and the pending value is consumed, so it cannot leak into a later turn
        assert verify_hooks.apply_pre_verify_verdict(a, "next") == "next"

    def test_recording_empty_clears_a_previous_verdict(self):
        a = self._Agent()
        verify_hooks.record_pre_verify_verdict(a, "VERDICT: work remains")
        verify_hooks.record_pre_verify_verdict(a, None)
        assert verify_hooks.apply_pre_verify_verdict(a, "answer") == "answer"


class TestPreVerifyDirectiveResolution:
    """``hermes_cli.plugins.get_pre_verify_directive`` — both halves, one pass."""

    def _with_results(self, monkeypatch, results):
        calls = []

        def fake_invoke(hook_name, **kwargs):
            calls.append(hook_name)
            return results

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke)
        return calls

    def test_old_continue_only_shape_is_unchanged(self, monkeypatch):
        from hermes_cli import plugins

        self._with_results(monkeypatch, [{"action": "continue", "message": "go on"}])
        assert plugins.get_pre_verify_directive() == {
            "message": "go on", "final_verdict": ""}
        assert plugins.get_pre_verify_continue_message() == "go on"

    def test_no_hooks_means_no_directive(self, monkeypatch):
        from hermes_cli import plugins

        self._with_results(monkeypatch, [])
        assert plugins.get_pre_verify_directive() == {"message": "", "final_verdict": ""}
        assert plugins.get_pre_verify_continue_message() is None

    def test_terminal_verdict_without_continuation(self, monkeypatch):
        from hermes_cli import plugins

        self._with_results(monkeypatch, [{"action": "final", "message": "VERDICT"}])
        d = plugins.get_pre_verify_directive()
        assert d == {"message": "", "final_verdict": "VERDICT"}
        # A terminal verdict must NOT be mistaken for a continuation.
        assert plugins.get_pre_verify_continue_message() is None

    def test_continue_and_verdict_in_one_directive(self, monkeypatch):
        from hermes_cli import plugins

        self._with_results(monkeypatch, [
            {"action": "continue", "message": "keep going", "final_verdict": "VERDICT"},
        ])
        assert plugins.get_pre_verify_directive() == {
            "message": "keep going", "final_verdict": "VERDICT"}

    def test_hooks_are_consulted_once_per_resolution(self, monkeypatch):
        from hermes_cli import plugins

        calls = self._with_results(monkeypatch, [{"action": "continue", "message": "x"}])
        plugins.get_pre_verify_directive()
        assert calls == ["pre_verify"]

    def test_invalid_returns_are_ignored(self, monkeypatch):
        from hermes_cli import plugins

        self._with_results(monkeypatch, [
            None, "nope", 7, {"action": "continue"}, {"final_verdict": 5},
            {"action": "continue", "message": "  "},
        ])
        assert plugins.get_pre_verify_directive() == {"message": "", "final_verdict": ""}
