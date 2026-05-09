import json

import pytest

from hermes_cli import proactive


def test_build_reflection_prompt_is_safe_and_silent_by_default():
    prompt = proactive.build_reflection_prompt(
        lookback_days=5,
        max_sessions=12,
        min_confidence="high",
    )

    assert "session_search" in prompt
    assert "last 5 days" in prompt
    assert "up to 12" in prompt
    assert "[SILENT]" in prompt
    assert "at most one proactive message" in prompt
    assert "Do not send emails, posts, DMs, calendar invites" in prompt
    assert "Ask before" in prompt
    assert "high confidence" in prompt


def test_install_creates_idempotent_cron_job(monkeypatch):
    calls = []
    jobs = []

    def fake_list_jobs(include_disabled=True):
        return list(jobs)

    def fake_create_job(**kwargs):
        calls.append(("create", kwargs))
        job = {
            "id": "abc123",
            "name": kwargs["name"],
            "prompt": kwargs["prompt"],
            "schedule_display": kwargs["schedule"],
            "deliver": kwargs["deliver"],
            "enabled_toolsets": kwargs["enabled_toolsets"],
            "state": "scheduled",
        }
        jobs.append(job)
        return job

    def fake_update_job(job_id, updates):
        calls.append(("update", job_id, updates))
        jobs[0].update(updates)
        return dict(jobs[0])

    monkeypatch.setattr(proactive, "list_jobs", fake_list_jobs)
    monkeypatch.setattr(proactive, "create_job", fake_create_job)
    monkeypatch.setattr(proactive, "update_job", fake_update_job)

    first = proactive.install_proactive_job(
        schedule="0 9 * * *",
        deliver="telegram",
        lookback_days=3,
        max_sessions=20,
    )
    second = proactive.install_proactive_job(
        schedule="0 10 * * *",
        deliver="local",
        lookback_days=7,
        max_sessions=40,
    )

    assert first["action"] == "created"
    assert second["action"] == "updated"
    assert calls[0][0] == "create"
    created = calls[0][1]
    assert created["name"] == proactive.DEFAULT_JOB_NAME
    assert created["enabled_toolsets"] == ["session_search", "memory", "todo"]
    assert "last 3 days" in created["prompt"]
    assert created["deliver"] == "telegram"

    assert calls[1][0] == "update"
    assert calls[1][1] == "abc123"
    assert calls[1][2]["schedule"] == "0 10 * * *"
    assert calls[1][2]["deliver"] == "local"
    assert "last 7 days" in calls[1][2]["prompt"]


def test_install_can_create_paused_job(monkeypatch):
    created_jobs = []
    updates = []

    monkeypatch.setattr(proactive, "list_jobs", lambda include_disabled=True: [])

    def fake_create_job(**kwargs):
        job = {"id": "paused1", "name": kwargs["name"], "state": "scheduled", "enabled": True}
        created_jobs.append(kwargs)
        return job

    def fake_update_job(job_id, update):
        updates.append((job_id, update))
        return {"id": job_id, **update}

    monkeypatch.setattr(proactive, "create_job", fake_create_job)
    monkeypatch.setattr(proactive, "update_job", fake_update_job)

    result = proactive.install_proactive_job(paused=True)

    assert result["action"] == "created_paused"
    assert updates == [("paused1", {"enabled": False, "state": "paused", "paused_reason": "created paused for review"})]


def test_cli_prompt_outputs_json_when_requested(capsys):
    rc = proactive.cmd_proactive(
        type(
            "Args",
            (),
            {
                "proactive_command": "prompt",
                "lookback_days": 2,
                "max_sessions": 9,
                "min_confidence": "medium",
                "json": True,
            },
        )()
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lookback_days"] == 2
    assert payload["max_sessions"] == 9
    assert "medium confidence" in payload["prompt"]
