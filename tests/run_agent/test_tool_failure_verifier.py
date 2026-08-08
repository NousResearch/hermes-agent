"""Tests for the per-turn generic tool-failure verifier footer.

Covers the three moving pieces introduced for #54722:

1. ``AIAgent._record_tool_failure`` — records an unresolved tool failure
   keyed by tool name, keeping the first error for a tool.
2. ``AIAgent._record_tool_success`` — reconciles a failure away when the
   same tool later succeeds, so a failed probe that was retried and
   resolved does NOT force the final warning.
3. ``AIAgent._format_tool_failure_footer`` — renders the remaining
   (unresolved) failures as a user-visible advisory.

The file-mutation verifier (#35584) tracks per-path state for
write_file/patch.  This generalises the same "structural challenge to an
over-confident success summary" idea to ANY tool call that ended in an
error and was not subsequently resolved.
"""

from __future__ import annotations

import json

import pytest

from run_agent import AIAgent, _extract_error_preview


def _bare_agent() -> AIAgent:
    """Skip __init__ and only attach the per-turn tool-failure state dict.

    AIAgent.__init__ touches network, auth, and the filesystem.  These
    tests only need the two record helpers and the footer formatter, so we
    build a bare instance via ``object.__new__`` — the same pattern the
    file-mutation verifier tests use.
    """
    agent = object.__new__(AIAgent)
    agent._turn_tool_failures = {}
    return agent


# ---------------------------------------------------------------------------
# _record_tool_failure — unresolved-failure semantics
# ---------------------------------------------------------------------------


class TestRecordToolFailure:
    def test_failure_recorded(self):
        agent = _bare_agent()
        agent._record_tool_failure("terminal", json.dumps({"exit_code": 1, "error": "boom"}))
        state = agent._turn_tool_failures
        assert "terminal" in state
        assert state["terminal"]["tool"] == "terminal"
        assert "boom" in state["terminal"]["error_preview"]

    def test_repeated_failure_keeps_first_error(self):
        agent = _bare_agent()
        agent._record_tool_failure("terminal", json.dumps({"exit_code": 1, "error": "first"}))
        agent._record_tool_failure("terminal", json.dumps({"exit_code": 1, "error": "second"}))
        assert agent._turn_tool_failures["terminal"]["error_preview"] == "first"

    def test_no_state_dict_silent_noop(self):
        agent = object.__new__(AIAgent)  # no state attached
        # Should not raise when called outside run_conversation.
        agent._record_tool_failure("terminal", "Error: x")
        agent._record_tool_success("terminal")


# ---------------------------------------------------------------------------
# _record_tool_success — reconciliation
# ---------------------------------------------------------------------------


class TestRecordToolSuccess:
    def test_success_reconciles_prior_failure(self):
        agent = _bare_agent()
        agent._record_tool_failure("read_file", json.dumps({"error": "File not found: x"}))
        assert "read_file" in agent._turn_tool_failures
        # Retried successfully later in the turn.
        agent._record_tool_success("read_file")
        assert agent._turn_tool_failures == {}

    def test_success_does_not_touch_other_tools(self):
        agent = _bare_agent()
        agent._record_tool_failure("terminal", json.dumps({"exit_code": 1}))
        agent._record_tool_success("read_file")
        assert "terminal" in agent._turn_tool_failures


# ---------------------------------------------------------------------------
# _format_tool_failure_footer
# ---------------------------------------------------------------------------


class TestFormatFooter:
    def test_empty_returns_empty_string(self):
        assert AIAgent._format_tool_failure_footer({}) == ""

    def test_single_failure(self):
        out = AIAgent._format_tool_failure_footer(
            {"terminal": {"tool": "terminal", "error_preview": "exit 1"}},
        )
        assert "Tool-verifier:" in out
        assert "terminal" in out
        assert "exit 1" in out

    def test_truncation_at_5_entries(self):
        failures = {
            f"tool{i}": {"tool": f"tool{i}", "error_preview": "err"}
            for i in range(8)
        }
        out = AIAgent._format_tool_failure_footer(failures)
        assert "8 tool call(s) returned errors" in out
        assert "… and 3 more" in out
        bullet_lines = [ln for ln in out.split("\n") if ln.lstrip().startswith("•")]
        assert len(bullet_lines) == 6  # 5 shown + 1 summary

    def test_reconciled_failures_do_not_appear(self):
        # Only unresolved failures remain in the dict — nothing to render.
        assert AIAgent._format_tool_failure_footer({}) == ""

    def test_paths_are_backtick_wrapped(self):
        out = AIAgent._format_tool_failure_footer(
            {"write_file": {"tool": "write_file", "error_preview": "/home/u/.hermes/config.yaml denied"}},
        )
        assert "`/home/u/.hermes/config.yaml`" in out


# ---------------------------------------------------------------------------
# _tool_failure_verifier_enabled — config-only gate
# ---------------------------------------------------------------------------


class TestVerifierEnabled:
    def test_default_is_enabled(self, monkeypatch):
        import hermes_cli.config as _cfg_mod
        monkeypatch.setattr(_cfg_mod, "load_config", lambda: {})
        assert _bare_agent()._tool_failure_verifier_enabled() is True

    def test_config_disables(self, monkeypatch):
        import hermes_cli.config as _cfg_mod
        monkeypatch.setattr(
            _cfg_mod, "load_config",
            lambda: {"display": {"tool_failure_verifier": False}},
        )
        assert _bare_agent()._tool_failure_verifier_enabled() is False

    def test_config_enables(self, monkeypatch):
        import hermes_cli.config as _cfg_mod
        monkeypatch.setattr(
            _cfg_mod, "load_config",
            lambda: {"display": {"tool_failure_verifier": True}},
        )
        assert _bare_agent()._tool_failure_verifier_enabled() is True

    def test_no_env_override(self, monkeypatch):
        # No HERMES_TOOL_FAILURE_VERIFIER override may exist (AGENTS.md
        # reserves HERMES_*/.env for secrets; display settings live in
        # config.yaml).  Assert the env var is not consulted.
        monkeypatch.setenv("HERMES_TOOL_FAILURE_VERIFIER", "0")
        import hermes_cli.config as _cfg_mod
        monkeypatch.setattr(
            _cfg_mod, "load_config",
            lambda: {"display": {"tool_failure_verifier": True}},
        )
        assert _bare_agent()._tool_failure_verifier_enabled() is True
