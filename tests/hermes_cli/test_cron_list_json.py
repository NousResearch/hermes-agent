"""`hermes cron list --json` — machine-readable job listing.

Fleet scripts and dashboards need stable JSON, not box-drawing prose.
The JSON path prints one stable-shape object per job and `[]` for an
empty fleet; the human path is unchanged.
"""

import json

import pytest

from cron.jobs import create_job
from hermes_cli.cron import cron_list


def _seed_job(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = create_job(
        prompt="check the deploy",
        schedule="every 60m",
        name="deploy-watch",
        deliver="local",
    )
    return job


def test_json_output_is_array_of_stable_shapes(tmp_path, monkeypatch, capsys):
    _seed_job(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.cron._builtin_gateway_liveness", lambda: True)

    cron_list(show_all=True, json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 1
    entry = payload[0]
    for key in (
        "id", "name", "schedule", "state", "enabled", "repeat_times",
        "repeat_completed", "next_run_at", "last_run_at", "last_status",
        "deliver", "skills", "model", "provider",
    ):
        assert key in entry, f"missing stable key: {key}"
    assert entry["name"] == "deploy-watch"
    assert entry["state"] in {"scheduled", "paused", "completed"}
    assert entry["deliver"] == ["local"]


def test_json_empty_fleet_prints_empty_array(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.cron._builtin_gateway_liveness", lambda: True)

    cron_list(show_all=True, json_output=True)

    assert json.loads(capsys.readouterr().out) == []


def test_human_output_unchanged(tmp_path, monkeypatch, capsys):
    _seed_job(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.cron._builtin_gateway_liveness", lambda: True)

    cron_list(show_all=False, json_output=False)

    out = capsys.readouterr().out
    assert "Scheduled Jobs" in out
    assert "deploy-watch" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
