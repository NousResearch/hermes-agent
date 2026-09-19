"""`hermes cron list --json` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â machine-readable job listing.

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


def test_json_without_all_excludes_disabled_jobs(tmp_path, monkeypatch, capsys):
    """A disabled job must not appear in JSON unless --all is passed."""
    job = _seed_job(tmp_path, monkeypatch)
    from cron.jobs import update_job

    update_job(job["id"], {"enabled": False})
    monkeypatch.setattr("hermes_cli.cron._builtin_gateway_liveness", lambda: True)

    cron_list(show_all=False, json_output=True)
    assert json.loads(capsys.readouterr().out) == []

    cron_list(show_all=True, json_output=True)
    payload = json.loads(capsys.readouterr().out)
    assert [j["id"] for j in payload] == [job["id"]]
    assert payload[0]["enabled"] is False


def test_json_schedule_unknown_is_null_not_question_mark(tmp_path, monkeypatch, capsys):
    """Machines get null for unresolvable schedules, never a literal '?'. """
    job = _seed_job(tmp_path, monkeypatch)
    job.pop("schedule_display", None)
    job["schedule"] = {}
    monkeypatch.setattr(
        "cron.jobs.list_jobs", lambda include_disabled=True: [job]
    )

    cron_list(show_all=True, json_output=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["schedule"] is None


def test_json_explicit_empty_skills_list_respected(tmp_path, monkeypatch, capsys):
    """skills: [] is honored; only None falls back to the legacy singular."""
    job = _seed_job(tmp_path, monkeypatch)
    job["skills"] = []
    job["skill"] = "legacy-skill"
    monkeypatch.setattr(
        "cron.jobs.list_jobs", lambda include_disabled=True: [job]
    )

    cron_list(show_all=True, json_output=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["skills"] == []
