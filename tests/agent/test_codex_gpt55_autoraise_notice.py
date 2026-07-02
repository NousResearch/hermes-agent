"""Tests for the Codex gpt-5.5 autoraise-notice dedupe gate (#54432).

The notice ("auto-compaction was raised to 85%…") must be shown at most once
per profile/config state. Before the fix it re-fired on every agent init, and
because the gateway rebuilds the agent per inbound message it spammed Discord
etc. The gate persists a marker under ``$HERMES_HOME`` (profile-scoped, isolated
to a tempdir by the conftest autouse fixture) keyed on the displayed from→to
percentages, so an unchanged threshold stays silent across restarts while a
changed threshold re-notifies once.
"""

from __future__ import annotations

import pytest

from hermes_constants import get_hermes_home

from agent.agent_init import (
    _codex_gpt55_autoraise_notice_marker,
    _codex_gpt55_autoraise_notice_seen,
    _codex_gpt55_autoraise_notice_state,
    _record_codex_gpt55_autoraise_notice,
)

# The (from, to) ratios agent_init stashes when the Codex gpt-5.5 override fires.
AUTORAISE = {"from": 0.50, "to": 0.85}


def test_marker_lives_under_hermes_home() -> None:
    marker = _codex_gpt55_autoraise_notice_marker()
    assert marker.parent == get_hermes_home()
    assert marker.name == ".codex_gpt55_autoraise_notice"


def test_state_keyed_on_displayed_percentages() -> None:
    # Same percentages the notice text renders (int(round(ratio * 100))).
    assert _codex_gpt55_autoraise_notice_state(AUTORAISE) == "50:85"
    assert _codex_gpt55_autoraise_notice_state({"from": 0.75, "to": 0.85}) == "75:85"


def test_unseen_before_anything_is_recorded() -> None:
    assert _codex_gpt55_autoraise_notice_seen(AUTORAISE) is False


def test_seen_after_record() -> None:
    assert _codex_gpt55_autoraise_notice_seen(AUTORAISE) is False
    _record_codex_gpt55_autoraise_notice(AUTORAISE)
    # A "restart" is just another call: the marker persists on disk.
    assert _codex_gpt55_autoraise_notice_seen(AUTORAISE) is True


def test_changed_threshold_renotifies_once() -> None:
    _record_codex_gpt55_autoraise_notice(AUTORAISE)
    assert _codex_gpt55_autoraise_notice_seen(AUTORAISE) is True
    # User raises their global threshold -> "from" changes -> notice re-fires.
    changed = {"from": 0.60, "to": 0.85}
    assert _codex_gpt55_autoraise_notice_seen(changed) is False
    _record_codex_gpt55_autoraise_notice(changed)
    assert _codex_gpt55_autoraise_notice_seen(changed) is True
    # And the old state is now considered unseen (marker moved forward).
    assert _codex_gpt55_autoraise_notice_seen(AUTORAISE) is False


def test_record_is_idempotent() -> None:
    _record_codex_gpt55_autoraise_notice(AUTORAISE)
    _record_codex_gpt55_autoraise_notice(AUTORAISE)
    assert _codex_gpt55_autoraise_notice_marker().read_text(encoding="utf-8") == "50:85"


def test_malformed_marker_reads_as_unseen() -> None:
    marker = _codex_gpt55_autoraise_notice_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("not-a-state", encoding="utf-8")
    assert _codex_gpt55_autoraise_notice_seen(AUTORAISE) is False


@pytest.mark.parametrize("bad", [{}, {"from": 0.5}, {"from": None, "to": None}])
def test_seen_tolerates_malformed_autoraise(bad) -> None:
    # Never raises even if the stashed dict is missing/garbage keys.
    assert _codex_gpt55_autoraise_notice_seen(bad) is False


def test_full_init_gate_shows_once_then_stays_silent() -> None:
    # Mirror the decision agent_init makes on each build:
    #   show = bool(autoraise) and compression_enabled and not seen(autoraise)
    def decide(compression_enabled: bool) -> bool:
        show = (
            bool(AUTORAISE)
            and compression_enabled
            and not _codex_gpt55_autoraise_notice_seen(AUTORAISE)
        )
        if show:
            _record_codex_gpt55_autoraise_notice(AUTORAISE)
        return show

    # First init (any surface) shows; every subsequent init in this profile
    # stays silent — the gateway spam scenario from the issue.
    assert decide(compression_enabled=True) is True
    assert decide(compression_enabled=True) is False
    assert decide(compression_enabled=True) is False
