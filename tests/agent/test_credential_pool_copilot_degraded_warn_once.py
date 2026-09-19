"""Regression: the Copilot "degraded to RAW token" WARNING must fire once per
source per process, not on every pool load.

``_seed_copilot_singleton`` runs on every ``load_pool("copilot")`` (model
picker, delegation spawns, dashboard polls). While the exchange is down its
negative cache fails fast, so each load degrades again within milliseconds and
the unconditional WARNING stormed errors.log (8 identical lines in 5s). The
warning stays actionable: it re-arms once the exchange recovers.
"""

from __future__ import annotations

import json
import logging

import pytest

_DEGRADED_MSG = "Copilot token exchange degraded to RAW token"


def _degraded_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and _DEGRADED_MSG in r.getMessage()]


@pytest.fixture()
def copilot_env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    import agent.credential_pool as cp
    monkeypatch.setattr(cp, "_COPILOT_DEGRADED_WARNED", set(), raising=False)
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_fake_token_abc123", "gh auth token"),
    )
    exchange = {"ok": False}

    def _get_api_token(raw):
        return ("tid=exchanged;exp=1", "https://api.enterprise.example") if exchange["ok"] else (raw, None)

    monkeypatch.setattr("hermes_cli.copilot_auth.get_copilot_api_token", _get_api_token)
    return exchange


def test_degraded_warning_logged_once_across_pool_loads(copilot_env, caplog):
    from agent.credential_pool import load_pool

    with caplog.at_level(logging.WARNING, logger="agent.credential_pool"):
        for _ in range(5):
            load_pool("copilot")

    assert len(_degraded_records(caplog)) == 1


def test_degraded_warning_rearms_after_exchange_recovers(copilot_env, caplog):
    from agent.credential_pool import load_pool

    with caplog.at_level(logging.WARNING, logger="agent.credential_pool"):
        load_pool("copilot")
        load_pool("copilot")
        copilot_env["ok"] = True
        load_pool("copilot")
        copilot_env["ok"] = False
        load_pool("copilot")
        load_pool("copilot")

    assert len(_degraded_records(caplog)) == 2
