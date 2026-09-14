"""Unified generation + delivery contract for cron email reports."""

import os

import cron.scheduler as scheduler
from cron.email_contract import (
    CANONICAL_PURPLE_PALETTE,
    CANONICAL_PURPLE_PALETTE_ID,
    EMAIL_CONTRACT_FAILED_MARKER,
    build_repair_prompt,
    render_contract_failure_email,
    resolve_email_contract,
    validate_email_output,
)


VALID_HTML = "<!DOCTYPE html><html><head><title>Monthly Audit — July 2026</title></head><body>ok</body></html>"
PALETTE_STYLE = "<style>:root{" + ";".join(
    f"--{name}:{value}" for name, value in CANONICAL_PURPLE_PALETTE.items()
) + "}.header{background:linear-gradient(135deg,#6C27D7 0%,#4F1E9C 100%)}</style>"
PALETTE_HTML = VALID_HTML.replace("</head>", f"{PALETTE_STYLE}</head>")


def _html_job(**overrides):
    job = {
        "id": "email-job",
        "name": "Monthly Audit",
        "prompt": "Return branded HTML with <!DOCTYPE html> and an email Subject from <title>.",
        "deliver": "email",
        "schedule_display": "manual",
    }
    job.update(overrides)
    return job


def test_existing_html_email_jobs_are_inferred_and_plain_alerts_are_not():
    html_contract = resolve_email_contract(_html_job())
    plain_contract = resolve_email_contract(
        _html_job(prompt="Return one short operational alert.")
    )
    assert html_contract is not None and html_contract.format == "branded_html"
    assert plain_contract is not None and plain_contract.format == "plain_text"
    assert resolve_email_contract(_html_job(deliver="telegram")) is None


def test_valid_html_is_normalized_to_one_document():
    result = validate_email_output(
        _html_job(), f"Here it is:\n```html\n{VALID_HTML}\n```\nfinished"
    )
    assert result.valid
    assert result.content == VALID_HTML


def test_palette_contract_rejects_old_branding_and_accepts_canonical_palette():
    job = _html_job(
        email_contract={
            "format": "branded_html",
            "retries": 1,
            "palette": CANONICAL_PURPLE_PALETTE_ID,
        }
    )
    contract = resolve_email_contract(job)
    assert contract is not None and contract.palette == CANONICAL_PURPLE_PALETTE_ID

    old = validate_email_output(job, VALID_HTML)
    assert not old.valid
    assert "missing canonical palette colors" in " ".join(old.errors)
    assert "header gradient" in " ".join(old.errors)

    canonical = validate_email_output(job, PALETTE_HTML)
    assert canonical.valid

    repair = build_repair_prompt(job, old.errors)
    assert "#6C27D7" in repair
    assert "#4F1E9C" in repair
    assert "Include every palette color" in repair


def test_plain_narration_fails_html_contract():
    result = validate_email_output(_html_job(), "Audit complete. Delivering report.")
    assert not result.valid
    assert "does not start" in " ".join(result.errors)
    assert "no non-empty <title>" in " ".join(result.errors)


def test_generic_or_reply_prefixed_title_is_rejected():
    html = "<!DOCTYPE html><html><head><title>Re: Hermes Agent</title></head><body>x</body></html>"
    result = validate_email_output(_html_job(), html)
    assert not result.valid
    assert "generic or reply-prefixed" in " ".join(result.errors)


def test_failure_alert_is_itself_valid_branded_html():
    alert = render_contract_failure_email(
        _html_job(),
        f"{EMAIL_CONTRACT_FAILED_MARKER} missing title",
        generated="2026-08-31 23:50 +0300",
        run_id="exec-123",
    )
    result = validate_email_output(_html_job(), alert)
    assert result.valid
    assert "Email Report Generation Failed — Monthly Audit" in alert
    assert "malformed response was not delivered" in alert
    assert "linear-gradient(135deg,#6C27D7 0%,#4F1E9C 100%)" in alert
    assert all(value in alert for value in CANONICAL_PURPLE_PALETTE.values())


class _FakeSessionDB:
    def __init__(self, *args, **kwargs):
        pass

    def set_session_title(self, *args, **kwargs):
        return True

    def get_compression_tip(self, session_id):
        return None

    def session_lifecycle_statuses(self, session_ids):
        return {sid: "complete" for sid in session_ids}

    def end_session(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _SequencedAgent:
    responses = []
    prompts = []

    def __init__(self, *args, **kwargs):
        pass

    def run_conversation(self, prompt, **kwargs):
        type(self).prompts.append(prompt)
        response = type(self).responses.pop(0)
        return {
            "completed": True,
            "failed": False,
            "final_response": response,
            "turn_exit_reason": "",
            "messages": [],
        }

    def get_activity_summary(self):
        return {"seconds_since_activity": 0}

    def close(self):
        pass


def _prepare_run_job(monkeypatch, tmp_path, responses):
    import hermes_state
    import run_agent

    _SequencedAgent.responses = list(responses)
    _SequencedAgent.prompts = []
    monkeypatch.setattr(run_agent, "AIAgent", _SequencedAgent)
    monkeypatch.setattr(hermes_state, "SessionDB", _FakeSessionDB)
    monkeypatch.setattr("hermes_constants.resolve_reasoning_config", lambda *_a, **_k: None)
    monkeypatch.setenv("HERMES_TEST_RUNTIME_KEY", "unused-placeholder")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": os.environ["HERMES_TEST_RUNTIME_KEY"],
            "base_url": None,
            "provider": "test-provider",
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(scheduler, "_guard_job_credential_exfil", lambda _job: None)
    monkeypatch.setattr(scheduler, "_preflight_job_config", lambda _job, _cfg: None)


def test_run_job_repairs_invalid_html_once_in_same_agent_session(monkeypatch, tmp_path):
    _prepare_run_job(monkeypatch, tmp_path, ["Audit complete.", VALID_HTML])
    success, _output, final, error = scheduler.run_job(_html_job())
    assert success is True
    assert error is None
    assert final == VALID_HTML
    assert len(_SequencedAgent.prompts) == 2
    assert "EMAIL OUTPUT CONTRACT REPAIR" in _SequencedAgent.prompts[1]
    assert "Do not call tools" in _SequencedAgent.prompts[1]


def test_run_job_fails_closed_after_one_invalid_repair(monkeypatch, tmp_path):
    _prepare_run_job(monkeypatch, tmp_path, ["Audit complete.", "Still not HTML."])
    success, _output, final, error = scheduler.run_job(_html_job())
    assert success is False
    assert final == ""
    assert error is not None and EMAIL_CONTRACT_FAILED_MARKER in error
    assert len(_SequencedAgent.prompts) == 2


def test_run_one_job_delivers_deterministic_html_alert_after_repair_failure(monkeypatch):
    delivered = []
    marked = []
    error = f"RuntimeError: {EMAIL_CONTRACT_FAILED_MARKER} missing title"
    monkeypatch.setattr(scheduler, "create_execution", lambda *_a, **_k: {"id": "exec-alert-123"})
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: {"id": _execution_id})
    monkeypatch.setattr(scheduler, "run_job", lambda *_a, **_k: (False, "bad output", "", error))
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a, **_k: "/tmp/output.md")
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_k: delivered.append(content) or None,
    )
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda *args, **kwargs: marked.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(scheduler, "finish_execution", lambda *_a, **_k: None)
    monkeypatch.setattr(scheduler, "_upsert_incident_for_failure", lambda *_a, **_k: (False, "incident-1"))
    monkeypatch.setattr(scheduler, "_mark_incident_alerted", lambda *_a, **_k: None)

    assert scheduler.run_one_job(_html_job()) is True
    assert len(delivered) == 1
    checked = validate_email_output(_html_job(), delivered[0])
    assert checked.valid
    assert "Report delivery blocked" in delivered[0]
    assert marked[0][0][1] is False
