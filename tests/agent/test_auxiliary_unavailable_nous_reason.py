"""The goal judge names a Nous auxiliary auth failure instead of an opaque judge error (#42177).

``_resolve_nous_runtime_api`` swallows the Nous resolver's ``AuthError`` so the ladder can fall
back; the failure must still reach the operator (one WARNING) and the goal-loop status line.
"""
import logging

import yaml

import agent.auxiliary_unavailable as unavailable
from hermes_cli.auth_constants import AuthError


def _reset(monkeypatch):
    monkeypatch.setattr(unavailable, "_last_nous_detail", None)
    monkeypatch.setattr(unavailable, "_warned_nous_details", set())


def test_goal_judge_reason_names_nous_auth_failure_and_still_fails_open(tmp_path, monkeypatch):
    """Real judge_goal → call_llm → ladder with goal_judge pinned to nous and no Nous login."""
    _reset(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "nous", "default": "test-model"},
        "auxiliary": {"goal_judge": {"provider": "nous", "model": "test-model"}},
    }), encoding="utf-8")
    from hermes_cli.goals import judge_goal

    verdict, reason, parse_failed, wait_directive, judge_errored = judge_goal(
        "ship the fix", "edited the file and ran the tests", timeout=5)

    assert (verdict, parse_failed, wait_directive, judge_errored) == ("continue", False, None, True)
    assert reason.startswith("goal_judge auxiliary client unavailable: Nous Portal runtime credentials unavailable:")
    assert "hermes model" in reason, reason
    assert "judge error" not in reason


def test_nous_credential_failure_is_remembered_and_warned_once(caplog, monkeypatch):
    _reset(monkeypatch)
    exc = AuthError("Invalid refresh token", provider="nous", code="invalid_grant", relogin_required=True)
    with caplog.at_level(logging.WARNING, logger="agent.auxiliary_unavailable"):
        detail = unavailable.record_nous_credential_failure(exc)
        unavailable.record_nous_credential_failure(exc)

    assert detail.startswith("Nous Portal runtime credentials unavailable: ")
    assert "invalid_grant" in detail and "hermes model" in detail
    assert unavailable.nous_credential_failure_detail() == detail
    assert sum(detail in rec.getMessage() for rec in caplog.records) == 1
    unavailable.clear_nous_credential_failure()
    assert unavailable.nous_credential_failure_detail() is None


def test_never_logged_in_is_debug_but_a_dead_credential_warns(caplog, monkeypatch, tmp_path):
    """The auto-route walk resolves Nous on every pass; users who never chose Nous must not be nagged."""
    _reset(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    not_logged_in = AuthError("Hermes is not logged into Nous Portal.", provider="nous", relogin_required=True)
    dead = AuthError("Invalid refresh token", provider="nous", code="invalid_grant", relogin_required=True)
    with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_unavailable"):
        quiet = unavailable.record_nous_credential_failure(not_logged_in)
        loud = unavailable.record_nous_credential_failure(dead)

    levels = {rec.levelno for rec in caplog.records if quiet in rec.getMessage()}
    assert levels == {logging.DEBUG}, caplog.records
    assert {rec.levelno for rec in caplog.records if loud in rec.getMessage()} == {logging.WARNING}
    assert "hermes model" in quiet  # the goal judge still gets the remediation text
