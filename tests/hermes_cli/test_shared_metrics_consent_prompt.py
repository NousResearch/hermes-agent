"""Shared-metrics consent is one global answer with per-profile overrides.

The first answer given on ANY surface is recorded at the Hermes root and inherited by every
profile that has not set ``telemetry.shared_metrics.enabled`` itself; an explicit profile
key always wins. The one-time question (CLI, desktop) fires only while nobody has answered
and the current profile is off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_cli.config import read_raw_config, save_config
from hermes_cli.observability import shared_metrics_consent as consent
from hermes_cli.observability.shared_metrics_consent import (
    apply_shared_metrics_choice,
    consent_prompt_pending,
    global_consent_path,
    maybe_prompt_for_consent_cli,
    shared_metrics_state,
)
from hermes_cli.observability.shared_metrics_send_config import resolve_send_config


@pytest.fixture(autouse=True)
def _quiet_store(monkeypatch, _isolate_hermes_home):
    # The consent-window write is exercised elsewhere; keep these tests on the resolution contract.
    monkeypatch.setattr(consent, "record_send_consent_change", lambda **_k: None)


def _profile_dir(name: str) -> Path:
    """A named profile's home under the isolated root (``<root>/profiles/<name>``)."""
    home = global_consent_path().parent / "profiles" / name
    home.mkdir(parents=True)
    return home


def test_nobody_answered_means_off_and_pending():
    assert shared_metrics_state({}) == (False, False, False, "default")
    assert consent_prompt_pending({}) is True
    assert resolve_send_config({}).send is False


def test_first_answer_becomes_the_global_default_other_profiles_inherit():
    config = {}
    apply_shared_metrics_choice(config, enabled=True, send=True)

    assert global_consent_path().exists()
    # The answering profile carries explicit keys ...
    assert shared_metrics_state(config) == (True, True, True, "profile")
    # ... and a profile with NO telemetry block inherits the answer end to end.
    other = {}
    assert shared_metrics_state(other) == (True, True, True, "global")
    assert consent_prompt_pending(other) is False
    assert resolve_send_config(other).send is True


def test_explicit_profile_opt_out_beats_the_global_answer():
    apply_shared_metrics_choice({}, enabled=True, send=True)

    profile = {"telemetry": {"shared_metrics": {"enabled": False}}}
    assert shared_metrics_state(profile) == (False, False, True, "profile")
    assert resolve_send_config(profile).send is False
    # Off but already decided: never asked again.
    assert consent_prompt_pending(profile) is False


def test_later_answers_do_not_rewrite_the_global_default():
    apply_shared_metrics_choice({}, enabled=False, send=False)
    apply_shared_metrics_choice({}, enabled=True, send=True)

    assert shared_metrics_state({}) == (False, False, True, "global")


def test_explicit_opt_out_survives_save_config():
    """Without a False default in DEFAULT_CONFIG the explicit key stays on disk; if it were
    stripped, a profile that opted out would silently fall back to a global 'share'."""
    apply_shared_metrics_choice({}, enabled=True, send=True)
    config = read_raw_config()
    apply_shared_metrics_choice(config, enabled=False, send=False)
    save_config(config)

    on_disk = read_raw_config()
    assert on_disk["telemetry"]["shared_metrics"]["enabled"] is False
    assert shared_metrics_state(on_disk).send is False


def test_deployment_env_answers_for_every_profile_below_a_recorded_answer(monkeypatch):
    """HERMES_SHARED_METRICS is the automated-instance answer: it opts every profile in and
    suppresses the question, but a recorded human answer and an explicit profile key both win."""
    monkeypatch.setenv("HERMES_SHARED_METRICS", "true")
    assert shared_metrics_state({}) == (True, True, True, "env")
    assert consent_prompt_pending({}) is False
    assert resolve_send_config({}).send is True

    profile = {"telemetry": {"shared_metrics": {"enabled": False}}}
    assert shared_metrics_state(profile) == (False, False, True, "profile")

    apply_shared_metrics_choice({}, enabled=False, send=False)  # a person declined somewhere
    assert shared_metrics_state({}) == (False, False, True, "global")


def test_deployment_env_can_pre_decline_and_unset_means_undecided(monkeypatch):
    monkeypatch.setenv("HERMES_SHARED_METRICS", "false")
    assert shared_metrics_state({}) == (False, False, True, "env")
    assert consent_prompt_pending({}) is False

    monkeypatch.setenv("HERMES_SHARED_METRICS", "  ")
    assert shared_metrics_state({}) == (False, False, False, "default")
    assert consent_prompt_pending({}) is True


def test_cli_prompt_asks_once_then_never_for_any_profile(monkeypatch):
    class _Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda _q, default: False)
    monkeypatch.setattr("hermes_cli.cli_output.print_info", lambda *_a, **_k: None)

    assert maybe_prompt_for_consent_cli() is True
    assert global_consent_path().exists()

    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *_a, **_k: pytest.fail("asked twice"))
    assert maybe_prompt_for_consent_cli() is False
    # A brand-new profile under the same root inherits the decision and is not asked either.
    monkeypatch.setenv("HERMES_HOME", str(_profile_dir("second")))
    assert maybe_prompt_for_consent_cli() is False


def test_cli_prompt_never_asks_off_a_terminal(monkeypatch):
    class _Pipe:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _Pipe())
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *_a, **_k: pytest.fail("asked off-tty"))

    assert maybe_prompt_for_consent_cli() is False
    assert consent_prompt_pending(read_raw_config()) is True
