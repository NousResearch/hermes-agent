"""Regression tests for the Codex gpt-5.x autoraise notice gate.

Covers two layers:

1. The config display gate (``compression.codex_gpt55_autoraise_notice``) —
   suppresses the banner without disabling the threshold autoraise.
2. The per-profile dedupe marker (#54432) — the notice must show at most once
   per profile/config state. Before the fix it re-fired on every agent init,
   and because the gateway rebuilds the agent per inbound message it spammed
   Discord etc. The gate persists a marker under ``$HERMES_HOME``
   (profile-scoped, isolated to a tempdir by the conftest autouse fixture)
   keyed on the model slug + displayed from→to percentages, so an unchanged
   threshold stays silent across restarts while a changed threshold (or a
   different autoraised Codex model) re-notifies once.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from run_agent import AIAgent

from agent.agent_init import (
    _codex_gpt55_autoraise_notice_marker,
    _codex_gpt55_autoraise_notice_seen,
    _codex_gpt55_autoraise_notice_state,
    _record_codex_gpt55_autoraise_notice,
)

# The dict agent_init stashes when the Codex gpt-5.5 override fires.
AUTORAISE = {"model": "gpt-5.5", "from": 0.50, "to": 0.85}


def _config(*, show_notice: bool) -> dict:
    return {
        "compression": {
            "enabled": True,
            "threshold": 0.50,
            "target_ratio": 0.20,
            "protect_first_n": 3,
            "protect_last_n": 20,
            "codex_gpt55_autoraise": True,
            "codex_gpt55_autoraise_notice": show_notice,
        },
        "prompt_caching": {"cache_ttl": "5m"},
        "sessions": {},
        "bedrock": {},
    }


def _make_codex_agent(
    monkeypatch,
    tmp_path: Path,
    *,
    show_notice: bool,
    model: str = "gpt-5.5",
    context_length: int = 272_000,
):
    """Construct a real Codex agent under an isolated config."""
    import run_agent

    from hermes_cli import config as config_mod
    from agent import context_compressor as compressor_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: _config(show_notice=show_notice))
    monkeypatch.setattr(run_agent, "_hermes_home", get_hermes_home())
    monkeypatch.setattr(
        compressor_mod,
        "get_model_context_length",
        lambda *args, **kwargs: context_length,
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        agent = AIAgent(
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-key",
            provider="openai-codex",
            model=model,
            enabled_toolsets=[],
            disabled_toolsets=[],
            quiet_mode=False,
            skip_memory=True,
            session_db=db,
            session_id="codex-notice-test",
        )

    return agent, stdout.getvalue()


def _threshold_ratio(agent: AIAgent) -> float:
    compressor = getattr(agent, "context_compressor")
    return round(compressor.threshold_tokens / compressor.context_length, 2)


# ── config display gate ──────────────────────────────────────────────────────


def test_codex_gpt55_autoraise_notice_enabled_by_default(monkeypatch, tmp_path):
    agent, stdout = _make_codex_agent(monkeypatch, tmp_path, show_notice=True)

    assert _threshold_ratio(agent) == 0.85
    warning = getattr(agent, "_compression_warning")
    assert warning is not None
    assert "auto-compaction was raised" in warning
    assert "auto-compaction was raised" in stdout


def test_codex_gpt55_autoraise_notice_can_be_suppressed_without_disabling_autoraise(
    monkeypatch, tmp_path
):
    agent, stdout = _make_codex_agent(monkeypatch, tmp_path, show_notice=False)

    assert _threshold_ratio(agent) == 0.85
    assert getattr(agent, "_compression_warning") is None
    assert "auto-compaction was raised" not in stdout


def test_codex_gpt56_uses_372k_window_and_95_percent_autoraise(monkeypatch, tmp_path):
    agent, stdout = _make_codex_agent(
        monkeypatch,
        tmp_path,
        show_notice=True,
        model="gpt-5.6-sol",
        context_length=372_000,
    )

    assert agent.context_compressor.context_length == 372_000
    assert _threshold_ratio(agent) == 0.95
    warning = getattr(agent, "_compression_warning")
    assert "372K" in warning
    assert "95%" in warning
    assert "372K" in stdout


def test_codex_gpt55_autoraise_notice_deduped_across_agent_inits(monkeypatch, tmp_path):
    # Gateway spam scenario (#54432): the gateway rebuilds the agent per
    # inbound message. The first init shows the notice; the second stays
    # silent because the per-profile marker was recorded.
    agent1, stdout1 = _make_codex_agent(monkeypatch, tmp_path, show_notice=True)
    assert "auto-compaction was raised" in stdout1
    assert getattr(agent1, "_compression_warning") is not None

    agent2, stdout2 = _make_codex_agent(monkeypatch, tmp_path, show_notice=True)
    assert _threshold_ratio(agent2) == 0.85  # autoraise still applies
    assert "auto-compaction was raised" not in stdout2
    assert getattr(agent2, "_compression_warning") is None


# ── per-profile dedupe marker (#54432) ───────────────────────────────────────


def test_marker_lives_under_hermes_home() -> None:
    marker = _codex_gpt55_autoraise_notice_marker()
    assert marker.parent == get_hermes_home()
    assert marker.name == ".codex_gpt55_autoraise_notice"


def test_state_keyed_on_model_and_displayed_percentages() -> None:
    # Same percentages the notice text renders (int(round(ratio * 100))),
    # prefixed with the bare model slug.
    assert _codex_gpt55_autoraise_notice_state(AUTORAISE) == "gpt-5.5:50:85"
    assert (
        _codex_gpt55_autoraise_notice_state(
            {"model": "openai/gpt-5.4", "from": 0.75, "to": 0.85}
        )
        == "gpt-5.4:75:85"
    )


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
    changed = {"model": "gpt-5.5", "from": 0.60, "to": 0.85}
    assert _codex_gpt55_autoraise_notice_seen(changed) is False
    _record_codex_gpt55_autoraise_notice(changed)
    assert _codex_gpt55_autoraise_notice_seen(changed) is True
    # And the old state is now considered unseen (marker moved forward).
    assert _codex_gpt55_autoraise_notice_seen(AUTORAISE) is False


def test_changed_model_renotifies_once() -> None:
    # Switching to a different autoraised Codex model re-fires the notice
    # (the banner names the model, so it displays new information).
    _record_codex_gpt55_autoraise_notice(AUTORAISE)
    other_model = {"model": "gpt-5.4", "from": 0.50, "to": 0.85}
    assert _codex_gpt55_autoraise_notice_seen(other_model) is False
    _record_codex_gpt55_autoraise_notice(other_model)
    assert _codex_gpt55_autoraise_notice_seen(other_model) is True


def test_record_is_idempotent() -> None:
    _record_codex_gpt55_autoraise_notice(AUTORAISE)
    _record_codex_gpt55_autoraise_notice(AUTORAISE)
    assert (
        _codex_gpt55_autoraise_notice_marker().read_text(encoding="utf-8")
        == "gpt-5.5:50:85"
    )


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
