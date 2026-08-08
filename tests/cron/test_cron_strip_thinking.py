"""Cron delivery uses an explicit FINAL_CRON_OUTPUT marker (#52934 / #53383).

Tagged thinking is already stripped on the standard agent finalization
path; remaining untagged scratchpads must not be cut with a Markdown
``---`` heuristic. Extraction is delivery-only (run_one_job), preserving
the full response in the persisted cron output document.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import (
    CRON_DELIVERABLE_MARKER,
    _build_job_prompt,
    _extract_cron_deliverable_response,
    run_job,
)


def test_build_job_prompt_injects_deliverable_marker():
    prompt = _build_job_prompt(
        {
            "id": "job-marker",
            "name": "daily-brief",
            "prompt": "Write the morning brief",
            "skills": [],
        }
    )
    assert CRON_DELIVERABLE_MARKER in prompt
    assert "Only content after that marker" in prompt


def test_extract_uses_explicit_marker_only():
    body = (
        "All checks complete. Compiling the brief.\n\n"
        "Summary of findings:\n- Weather: peak 12.7°C\n\n"
        f"{CRON_DELIVERABLE_MARKER}\n\n"
        "Good morning.\n\n"
        "🌤 WEATHER\n• Today: 13℃\n"
    )
    delivered = _extract_cron_deliverable_response(body)
    assert "All checks complete" not in delivered
    assert delivered.lstrip().startswith("Good morning.")
    assert "WEATHER" in delivered


def test_extract_same_line_marker_tail():
    body = (
        "Working notes.\n\n"
        f"{CRON_DELIVERABLE_MARKER} Good morning.\n\n"
        "WEATHER\n• Today: 13C\n"
    )
    delivered = _extract_cron_deliverable_response(body)
    assert delivered.startswith("Good morning.")
    assert "Working notes" not in delivered


def test_extract_leaves_unmarked_markdown_hr_alone():
    """False-positive guard: titles using --- must not lose their heading."""
    body = (
        "Daily Brief\n"
        "---\n\n"
        "Weather: clear.\n"
        "Portfolio: unchanged.\n"
    )
    assert _extract_cron_deliverable_response(body) == body.strip()


def test_extract_leaves_unmarked_reasoning_preamble_alone():
    body = (
        "All checks complete. Compiling the brief.\n\n"
        "Summary of findings:\n- Weather: peak 12.7°C\n\n"
        "---\n\n"
        "Good morning.\n\n"
        "WEATHER\n• Today: 13C\n"
    )
    assert _extract_cron_deliverable_response(body) == body.strip()


def test_empty_marker_does_not_suppress_delivery():
    body = f"Working notes.\n\n{CRON_DELIVERABLE_MARKER}\n"
    assert _extract_cron_deliverable_response(body) == body.strip()


def _base_job(**overrides):
    job = {
        "id": "job-think",
        "name": "daily-brief",
        "prompt": "Write the morning brief",
        "model": None,
        "provider": None,
        "provider_snapshot": "openai",
        "base_url": None,
        "schedule": "0 7 * * *",
        "schedule_display": "daily 07:00",
        "enabled": True,
    }
    job.update(overrides)
    return job


def test_run_job_preserves_full_response_for_persisted_output(tmp_path):
    """run_job must not apply delivery extraction (document keeps full text)."""
    body = (
        "scratch notes\n\n"
        f"{CRON_DELIVERABLE_MARKER}\n\n"
        "User brief body\n"
    )
    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openai",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {
            "final_response": body,
            "messages": [],
            "completed": True,
            "failed": False,
        }
        mock_agent_cls.return_value = mock_agent
        success, output, final_response, error = run_job(_base_job())
    assert success is True
    assert error is None
    assert "scratch notes" in final_response
    assert CRON_DELIVERABLE_MARKER in final_response
    assert "User brief body" in final_response
    assert "scratch notes" in output


def test_run_one_job_extracts_marker_before_delivery(monkeypatch, tmp_path):
    """Delivery path only: marker extraction immediately before _deliver_result."""
    import cron.scheduler as s

    body = (
        "internal preamble\n\n"
        f"{CRON_DELIVERABLE_MARKER}\n\n"
        "Good morning.\n"
    )
    delivered = {}

    def fake_run_job(job, *, defer_agent_teardown=None):
        return (True, "# full doc with scratch", body, None)

    def fake_save(jid, out):
        return tmp_path / "out.md"

    def fake_deliver(job, content, adapters=None, loop=None):
        delivered["content"] = content
        return None

    def fake_mark(jid, success, error=None, delivery_error=None):
        return None

    monkeypatch.setattr(s, "claim_dispatch", lambda jid: True)
    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    monkeypatch.setattr(s, "_is_interrupted", lambda jid: False)

    assert s.run_one_job(_base_job()) is True
    assert delivered["content"].lstrip().startswith("Good morning.")
    assert "internal preamble" not in delivered["content"]
