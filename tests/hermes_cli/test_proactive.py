import json
import sys
import time
import types

from hermes_cli import proactive


def test_build_reflection_prompt_is_safe_and_silent_by_default():
    prompt = proactive.build_reflection_prompt(
        lookback_days=5,
        max_sessions=12,
        min_confidence="high",
    )

    assert "Proactive signal scan" in prompt
    assert "last 5 days" in prompt
    assert "up to 12" in prompt
    assert "[SILENT]" in prompt
    assert "at most one proactive message" in prompt
    assert "NEVER send anything outside Hermes/the configured delivery system" in prompt
    assert "Drafting internally is allowed" in prompt
    assert "Ask before" in prompt
    assert "high confidence" in prompt
    assert "personal assistant" in prompt
    assert "Proactive modes" in prompt
    assert "meeting notes" in prompt
    assert "Silence is the correct answer unless the best candidate clears the bar" in prompt
    assert "No generic summaries, status theater" in prompt
    assert "Would Charles plausibly reply" in prompt


def test_collect_proactive_signals_returns_structured_scan(monkeypatch):
    now = time.time()

    class FakeDB:
        def list_sessions_rich(self, **kwargs):
            return [
                {
                    "id": "s1",
                    "source": "telegram",
                    "title": "Sales meeting notes",
                    "last_active": now,
                    "preview": "Charles added meeting notes with a sales-team speech",
                }
            ]

        def search_messages(self, query, **kwargs):
            if "meeting notes" in query:
                return [
                        {
                            "session_id": "s1",
                            "source": "telegram",
                            "timestamp": now,
                            "content": "meeting notes included a sales team speech and delivery coaching opportunity",
                            "snippet": "meeting notes included a sales team speech and delivery coaching opportunity",
                        }
                ]
            if "OwnerPath" in query:
                return [
                        {
                            "session_id": "s2",
                            "source": "telegram",
                            "timestamp": now,
                            "content": "OwnerPath is on hold; do not nudge",
                            "snippet": "OwnerPath is on hold; do not nudge",
                        }
                ]
            return []

    monkeypatch.setitem(sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=FakeDB))
    monkeypatch.setattr(
        proactive,
        "list_jobs",
        lambda include_disabled=True: [
            {"id": "j1", "name": "Broken cron", "last_error": "boom", "state": "scheduled"}
        ],
    )

    report = proactive.collect_proactive_signals(lookback_days=2, max_sessions=5)

    assert report["wakeAgent"] is True
    assert report["recent_sessions"][0]["id"] == "s1"
    assert {s["kind"] for s in report["signals"]} >= {"content_opportunity", "cron_failure"}
    assert report["suppressed_topics"]
    rendered = proactive.render_signal_scan(report)
    assert "## Proactive signal scan" in rendered
    assert json.loads(rendered.splitlines()[-1]) == {"wakeAgent": True}


def test_render_signal_scan_wake_gate_can_skip_agent():
    rendered = proactive.render_signal_scan({"wakeAgent": False, "signals": []})
    assert rendered == '{"wakeAgent": false}'


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
            "script": kwargs["script"],
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
    monkeypatch.setattr(proactive, "_ensure_scanner_script", lambda: "proactive_signal_scan.py")

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
    assert created["script"] == "proactive_signal_scan.py"
    assert created["enabled_toolsets"] == ["memory"]
    assert "last 3 days" in created["prompt"]
    assert created["deliver"] == "telegram"

    assert calls[1][0] == "update"
    assert calls[1][1] == "abc123"
    assert calls[1][2]["schedule"] == "0 10 * * *"
    assert calls[1][2]["deliver"] == "local"
    assert calls[1][2]["script"] == "proactive_signal_scan.py"
    assert "last 7 days" in calls[1][2]["prompt"]


def test_install_can_create_paused_job(monkeypatch):
    created_jobs = []
    updates = []

    monkeypatch.setattr(proactive, "list_jobs", lambda include_disabled=True: [])
    monkeypatch.setattr(proactive, "_ensure_scanner_script", lambda: "proactive_signal_scan.py")

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
    assert created_jobs[0]["script"] == "proactive_signal_scan.py"


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
