"""Regression tests for #31480 — invalid shell-hook event-name warnings.

A user who copies a gateway event name (``agent:end``, ``session:end``,
…) into ``config.yaml``'s ``hooks:`` block sees their hook silently
dropped; ``hermes hooks list`` previously reported only "No shell hooks
configured" with zero hint that anything went wrong.  These tests pin:

* the new ``_GATEWAY_EVENT_HINTS`` mapping and the
  ``_suggest_for_unknown_event`` helper that drives both the logger
  warning and the new CLI surface,
* the public ``validate_hooks_config`` helper used by ``hermes hooks
  list`` / ``hermes hooks doctor`` to surface those warnings to the
  operator,
* the CLI's call-site wiring so future refactors cannot quietly drop
  the warning print without breaking the test.
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import shell_hooks
from hermes_cli import hooks as hooks_cli


# ---------------------------------------------------------------------------
# _suggest_for_unknown_event
# ---------------------------------------------------------------------------


class TestSuggestForUnknownEvent:
    """Direct contract for the gateway-aware suggestion helper."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Gateway events — exact mapping wins over difflib's
            # fuzzy match (which scores ``agent:end`` below the 0.6
            # cutoff and would otherwise return None).
            ("agent:start", "pre_llm_call"),
            ("agent:step", "post_llm_call"),
            ("agent:end", "post_llm_call"),
            ("session:start", "on_session_start"),
            ("session:end", "on_session_end"),
            ("session:reset", "on_session_reset"),
        ],
    )
    def test_gateway_events_get_precise_mapping(self, raw, expected):
        assert shell_hooks._suggest_for_unknown_event(raw) == expected

    def test_gateway_only_events_have_no_shell_analogue(self):
        # ``gateway:startup`` fires only at the gateway layer — there is
        # no shell-hook lifecycle slot that's even close.  The helper
        # must return None so the caller can render "no analogue"
        # rather than a misleading "did you mean foo?".
        assert shell_hooks._suggest_for_unknown_event("gateway:startup") is None

    def test_typo_falls_back_to_difflib_close_match(self):
        # The original difflib path is still the right answer for
        # ordinary typos in the underscore namespace.
        assert (
            shell_hooks._suggest_for_unknown_event("pre_tool_calll")
            == "pre_tool_call"
        )

    def test_completely_unknown_event_returns_none(self):
        assert shell_hooks._suggest_for_unknown_event("xyzzy_made_up") is None

    def test_non_string_input_does_not_crash(self):
        # config.yaml occasionally hands us int/None keys after a YAML
        # parsing accident; the helper must coerce-or-skip rather than
        # raise.
        assert shell_hooks._suggest_for_unknown_event(None) is None
        assert shell_hooks._suggest_for_unknown_event(42) is None


# ---------------------------------------------------------------------------
# validate_hooks_config — public CLI-facing API
# ---------------------------------------------------------------------------


class TestValidateHooksConfig:
    """The new helper feeds ``hermes hooks list`` /
    ``hermes hooks doctor`` so the operator sees what the parser
    rejected.  Pin its output shape."""

    def test_empty_or_missing_block_is_silent(self):
        assert shell_hooks.validate_hooks_config({}) == []
        assert shell_hooks.validate_hooks_config({"hooks": {}}) == []
        assert shell_hooks.validate_hooks_config({"hooks": None}) == []
        assert shell_hooks.validate_hooks_config(None) == []

    def test_fully_valid_config_is_silent(self):
        cfg = {
            "hooks": {
                "post_llm_call": [{"command": "/x/hook.sh"}],
                "on_session_end": [{"command": "/x/hook.sh"}],
            }
        }
        assert shell_hooks.validate_hooks_config(cfg) == []

    def test_gateway_event_emits_targeted_message(self):
        cfg = {"hooks": {"agent:end": [{"command": "/x/hook.sh"}]}}
        warnings = shell_hooks.validate_hooks_config(cfg)
        assert len(warnings) == 1
        msg = warnings[0]
        # Header names the offending event verbatim and labels it as
        # invalid for shell hooks.
        assert "'agent:end' is not a valid shell hook event" in msg
        # Targeted suggestion (precise mapping, NOT the difflib dump).
        assert "Did you mean 'post_llm_call'?" in msg
        # Distinguishes the two namespaces.
        assert "Gateway events" in msg
        assert "config.yaml" in msg

    def test_gateway_only_event_explains_no_analogue(self):
        # ``gateway:startup`` has no shell-hook equivalent: the message
        # must explain the namespace mistake without misleadingly
        # suggesting a random shell event.
        cfg = {"hooks": {"gateway:startup": [{"command": "/x/hook.sh"}]}}
        warnings = shell_hooks.validate_hooks_config(cfg)
        assert len(warnings) == 1
        msg = warnings[0]
        assert "'gateway:startup' is not a valid shell hook event" in msg
        assert "Did you mean" not in msg  # no false suggestion
        assert "Gateway events" in msg

    def test_typo_warns_with_close_match(self):
        cfg = {"hooks": {"pre_tool_calll": [{"command": "/x/hook.sh"}]}}
        warnings = shell_hooks.validate_hooks_config(cfg)
        assert len(warnings) == 1
        assert "Did you mean 'pre_tool_call'?" in warnings[0]
        # No gateway-event noise on a plain typo.
        assert "Gateway events" not in warnings[0]

    def test_completely_unknown_event_lists_valid_options(self):
        cfg = {"hooks": {"xyzzy_no_match": [{"command": "/x/hook.sh"}]}}
        warnings = shell_hooks.validate_hooks_config(cfg)
        assert len(warnings) == 1
        assert "Valid shell hook events:" in warnings[0]
        # The full list must include at least the two events the bug
        # report quotes as the answer the user wanted.
        assert "post_llm_call" in warnings[0]
        assert "on_session_end" in warnings[0]

    def test_non_list_event_value_warns(self):
        cfg = {"hooks": {"pre_tool_call": "not-a-list"}}
        warnings = shell_hooks.validate_hooks_config(cfg)
        assert any("must be a list" in w for w in warnings)

    def test_multiple_problems_each_get_their_own_message(self):
        cfg = {
            "hooks": {
                "agent:end": [{"command": "/x"}],
                "pre_tool_calll": [{"command": "/x"}],
                "post_llm_call": [{"command": "/x"}],  # valid — no warn
            }
        }
        warnings = shell_hooks.validate_hooks_config(cfg)
        # One warning per offending key — the valid ``post_llm_call``
        # entry must NOT contribute a warning of its own.
        assert len(warnings) == 2
        # Each invalid key surfaces as its own header so the operator
        # can scan a multi-error config without parsing concatenations.
        assert any("'agent:end' is not a valid" in w for w in warnings)
        assert any("'pre_tool_calll' is not a valid" in w for w in warnings)
        # No warning header references the valid event — we only allow
        # ``post_llm_call`` to appear as a suggestion under
        # ``agent:end``, never as an offender.
        assert not any(
            "'post_llm_call' is not a valid" in w for w in warnings
        )


# ---------------------------------------------------------------------------
# Logger-level warnings (existing channel; still emitted, now with hint)
# ---------------------------------------------------------------------------


class TestParseHooksBlockLogsTargetedWarning:
    """The existing ``logger.warning`` channel must still fire and now
    carries the gateway-aware hint, so external log scrapers continue
    to see structured data."""

    def test_gateway_event_logs_targeted_message(self, caplog):
        with caplog.at_level(logging.WARNING, logger="agent.shell_hooks"):
            shell_hooks._parse_hooks_block({"agent:end": [{"command": "/x"}]})
        msgs = [r.getMessage() for r in caplog.records]
        joined = "\n".join(msgs)
        assert "agent:end" in joined
        assert "post_llm_call" in joined
        # The new wording explicitly tells the operator gateway hooks
        # live elsewhere — this is the missing context the bug report
        # complains about.
        assert "gateway hooks" in joined or "~/.hermes/hooks" in joined


# ---------------------------------------------------------------------------
# CLI surfacing — hermes hooks list / hermes hooks doctor
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()


def _run_cli(action: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        hooks_cli.hooks_command(SimpleNamespace(hooks_action=action))
    return buf.getvalue()


class TestCliSurfacesWarnings:
    """The bug report quotes the exact ``hermes hooks list`` output the
    user sees — pin the new wording so it can't regress to silence."""

    def test_hooks_list_prints_warning_for_gateway_event(self):
        cfg = {"hooks": {"agent:end": [{"command": "/tmp/hook.sh"}]}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            out = _run_cli("list")

        # The warning must come BEFORE the "No shell hooks configured"
        # line (or in lieu of it) so the user sees it first.
        assert "⚠" in out
        assert "agent:end" in out
        assert "post_llm_call" in out
        assert "Gateway events" in out
        # The original silent-fallback message stays — but only AFTER
        # the warning, so the user sees "your hook was dropped" before
        # "no hooks configured".
        warn_idx = out.find("⚠")
        empty_idx = out.find("No shell hooks configured")
        assert warn_idx >= 0 < empty_idx
        assert warn_idx < empty_idx

    def test_hooks_list_silent_when_block_is_clean(self):
        # No false-positive warnings on a healthy config.
        script = "/tmp/clean-hook.sh"
        cfg = {"hooks": {"post_llm_call": [{"command": script}]}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            out = _run_cli("list")
        assert "⚠" not in out

    def test_hooks_doctor_also_surfaces_warnings(self):
        cfg = {"hooks": {"session:end": [{"command": "/tmp/hook.sh"}]}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            out = _run_cli("doctor")
        # ``session:end`` must map precisely to ``on_session_end`` —
        # it's the sister case of ``agent:end`` from the bug report.
        assert "session:end" in out
        assert "on_session_end" in out
        assert "⚠" in out
