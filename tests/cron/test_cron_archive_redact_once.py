"""The cron run archive survives the gateway's storage redaction pass (#153079).

The run document is stamped (``**Prompt Characters:**`` / ``**Response Characters:**``) by
``run_job`` inside the gateway, then stored in state.db through ``prepare_result``, whose
redaction pass used to change the text under those stamps: a second pass re-masks an already
masked ``KEY='exampl...-32c'`` to ``KEY='***'``, and a secret-shaped token in the response was
masked for the first time after stamping. ``_archive_answer`` then found no frame.
"""

import json

import pytest

import cron.scheduler
import run_agent

FIXTURE = "example-fixture-value-not-a-real-secret-32c"
TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"


@pytest.fixture
def redaction_on(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


def _stored_archive(monkeypatch, job, answer):
    """Run ``job`` through the real writer and the gateway's real storage pass."""
    from gateway.session_results import prepare_result

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def run_conversation(self, *args, **kwargs):
            return {"final_response": answer, "completed": True, "failed": False}

    monkeypatch.setattr(run_agent, "AIAgent", Agent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        lambda **kwargs: {"provider": "openai", "api_key": "fixture"})
    result = cron.scheduler.run_job(job)
    assert result[0], result[3]
    captured = {"result": {"final_response": result[2], "cron_result": list(result),
                           "failed": False, "completed": True}, "usage": {}}
    _outcome, stored = prepare_result({"payload": {"text": ""}}, result[2], "completed", captured)
    return stored["result"]["cron_result"][1]


def test_stored_archive_frame_parses_and_secrets_stay_masked(cron_env, monkeypatch, cron_owner, redaction_on):
    from agent.redact import redact_sensitive_text
    from cron.jobs import create_job, save_job_output
    from cron.scheduler_prompt import _archive_answer, _inject_context_from

    # The prompt holds script output that capture already redacted once; the answer carries a
    # token the storage pass would have masked for the first time after the stamps.
    script_line = redact_sensitive_text(f"+const SECRET = '{FIXTURE}'")
    job = create_job(prompt=f"Review this diff:\n{script_line}", schedule="0 8 * * *", context_from="self")
    stored = _stored_archive(monkeypatch, job, f"Looks fine. Rotate {TOKEN} though.")

    assert FIXTURE not in stored and TOKEN not in stored
    answer = _archive_answer(stored)
    assert answer is not None and answer.startswith("Looks fine. Rotate ghp_")
    assert TOKEN not in answer
    save_job_output(job["id"], stored)
    prompt, injected = _inject_context_from(job, "Next task")
    assert injected and "Looks fine." in prompt and "Review this diff" not in prompt


def test_json_answer_quoting_an_assignment_stays_valid_json(cron_env, monkeypatch, cron_owner, redaction_on):
    from cron.jobs import create_job
    from cron.scheduler_prompt import _archive_answer
    from gateway.session_results import _redacted

    payload = json.dumps({"p": f'const SECRET = "{FIXTURE}".', "ok": True})
    # The storage boundary itself keeps an escaped-quote assignment parseable (any stored string).
    plain = json.loads(_redacted(payload))
    assert FIXTURE not in plain["p"] and plain["ok"] is True

    job = create_job(prompt="Report as JSON", schedule="0 8 * * *", context_from="self")
    stored = _stored_archive(monkeypatch, job, payload)
    answer = _archive_answer(stored)
    assert answer is not None
    parsed = json.loads(answer)
    assert FIXTURE not in parsed["p"] and parsed["p"].startswith("const SECRET=") and parsed["ok"] is True
