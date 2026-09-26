"""`hermes cron list --json` emits a machine-readable job listing.

Fleet scripts and dashboards need stable JSON, not box-drawing prose. The JSON
path prints one stable-shape object per job and `[]` for an empty fleet, and it
is built from the same job set and the same state resolution as the human
renderer, so the two never drift. The human path is unchanged.
"""

import json

import pytest

from cron.jobs import create_job
from hermes_cli.cron import cron_list

STABLE_KEYS = (
    "id", "name", "schedule", "state", "enabled", "repeat_times",
    "repeat_completed", "next_run_at", "last_run_at", "last_status",
    "deliver", "skills", "model", "provider",
)


def _seed_job(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.cron._builtin_gateway_liveness", lambda: True)
    return create_job(
        prompt="check the deploy",
        schedule="every 60m",
        name="deploy-watch",
        deliver="local",
    )


def test_json_output_is_array_of_stable_shapes(tmp_path, monkeypatch, capsys):
    _seed_job(tmp_path, monkeypatch)

    cron_list(show_all=True, json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 1
    entry = payload[0]
    for key in STABLE_KEYS:
        assert key in entry, f"missing stable key: {key}"
    assert entry["name"] == "deploy-watch"
    assert entry["state"] in {"scheduled", "paused", "completed", "error"}
    assert entry["deliver"] == ["local"]


def test_json_empty_fleet_prints_empty_array(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.cron._builtin_gateway_liveness", lambda: True)

    cron_list(show_all=True, json_output=True)

    assert json.loads(capsys.readouterr().out) == []


def test_human_output_unchanged(tmp_path, monkeypatch, capsys):
    _seed_job(tmp_path, monkeypatch)

    cron_list(show_all=False, json_output=False)

    out = capsys.readouterr().out
    assert "Scheduled Jobs" in out
    assert "deploy-watch" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_json_mirrors_human_visibility_for_a_disabled_job(tmp_path, monkeypatch, capsys):
    """A disabled job must be listed by JSON exactly when the human path lists it.

    ``effective_job_state`` resolves a disabled job to ``paused``, and the human
    listing deliberately shows paused records even without ``--all``; JSON is
    built from that same filtered set, so it has to agree.
    """
    job = _seed_job(tmp_path, monkeypatch)
    from cron.jobs import update_job

    update_job(job["id"], {"enabled": False})

    cron_list(show_all=False, json_output=False)
    human = capsys.readouterr().out
    cron_list(show_all=False, json_output=True)
    payload = json.loads(capsys.readouterr().out)

    assert job["id"] in human
    assert [j["id"] for j in payload] == [job["id"]]
    assert payload[0]["enabled"] is False
    assert payload[0]["state"] == "paused"


def test_json_schedule_unknown_is_null_not_question_mark(tmp_path, monkeypatch, capsys):
    """Machines get null for an unresolvable schedule, never a literal '?'."""
    job = _seed_job(tmp_path, monkeypatch)
    job.pop("schedule_display", None)
    job["schedule"] = {}
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=True: [job])

    cron_list(show_all=True, json_output=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["schedule"] is None


def test_json_explicit_empty_skills_list_respected(tmp_path, monkeypatch, capsys):
    """skills: [] is honoured; only a missing key falls back to the legacy singular."""
    job = _seed_job(tmp_path, monkeypatch)
    job["skills"] = []
    job["skill"] = "legacy-skill"
    monkeypatch.setattr("cron.jobs.list_jobs", lambda include_disabled=True: [job])

    cron_list(show_all=True, json_output=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["skills"] == []