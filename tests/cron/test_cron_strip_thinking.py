"""Cron delivery uses an explicit FINAL_CRON_OUTPUT marker (#52934 / #62511).

Tagged thinking is already stripped on the standard agent finalization path.
Unmarked scratchpads must not be cut with a Markdown ``---`` heuristic.
Extraction is delivery-only, so the persisted cron output keeps the full
model response and upstream sanitizers stay in place.
"""

from __future__ import annotations

from cron.scheduler import (
    CRON_DELIVERABLE_MARKER,
    _build_job_prompt,
    _compose_run_delivery,
    _extract_cron_deliverable_response,
    _final_response_from_result,
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


def test_extract_keeps_provider_fallback_notice():
    """The pre-agent fallback line is system delivery text, not scratch (#74349)."""
    notice = "⚠️ Provider fallback: primary/a unavailable; using fallback/b for this response."
    body = (
        f"{notice}\n\n"
        "scratch notes\n\n"
        f"{CRON_DELIVERABLE_MARKER}\n\n"
        "Good morning.\n"
    )
    delivered = _extract_cron_deliverable_response(body)
    assert delivered.startswith(notice)
    assert "scratch notes" not in delivered
    assert "Good morning." in delivered


class _ExplainerAgent:
    @staticmethod
    def _format_turn_completion_explanation(reason, cause=None, model=""):
        return f"NO-REPLY {reason} {cause or ''} {model}".strip()


def test_final_response_preserves_scratch_and_upstream_sanitizers(monkeypatch):
    """run_job's response builder must not apply delivery extraction."""
    body = (
        "scratch notes\n\n"
        f"{CRON_DELIVERABLE_MARKER}\n\n"
        "User brief body\n"
    )
    result = _final_response_from_result(
        {"final_response": body, "messages": [], "completed": True, "failed": False},
        "job-think",
        "daily-brief",
        _ExplainerAgent,
    )
    assert "scratch notes" in result
    assert CRON_DELIVERABLE_MARKER in result
    assert "User brief body" in result

    cleared = _final_response_from_result(
        {
            "final_response": "(No response generated)",
            "messages": [],
            "completed": True,
            "failed": False,
        },
        "job-think",
        "daily-brief",
        _ExplainerAgent,
    )
    assert cleared == ""

    explainer = _ExplainerAgent._format_turn_completion_explanation(
        "empty_turn", None, model="claude"
    )
    suppressed = _final_response_from_result(
        {
            "final_response": explainer,
            "messages": [],
            "completed": True,
            "failed": False,
            "turn_exit_reason": "empty_turn",
            "model": "claude",
        },
        "job-think",
        "daily-brief",
        _ExplainerAgent,
    )
    assert suppressed == ""

    def _mark_repaired(text, _messages):
        return text + " [repaired]"

    monkeypatch.setattr(
        "gateway.media_repair.repair_explicit_computer_use_media_paths",
        _mark_repaired,
    )
    repaired = _final_response_from_result(
        {"final_response": "see /tmp/shot.png", "messages": [], "completed": True},
        "job-think",
        "daily-brief",
        _ExplainerAgent,
    )
    assert repaired.endswith("[repaired]")


def test_compose_delivery_extracts_marker_only_on_success():
    body = (
        "internal preamble\n\n"
        f"{CRON_DELIVERABLE_MARKER}\n\n"
        "Good morning.\n"
    )
    job = {"id": "job-think", "name": "daily-brief"}
    content, *_rest = _compose_run_delivery(
        job, success=True, error=None, final_response=body, output_file=None,
    )
    assert content.lstrip().startswith("Good morning.")
    assert "internal preamble" not in content

    unmarked = "Daily Brief\n---\n\nWeather: clear.\n"
    kept, *_rest = _compose_run_delivery(
        job, success=True, error=None, final_response=unmarked, output_file=None,
    )
    assert kept == unmarked.strip()


def test_run_one_job_extracts_marker_before_delivery(monkeypatch, tmp_path):
    """The queued delivery is the extract; the saved document stays the full output."""
    import cron.scheduler as s

    body = (
        "internal preamble\n\n"
        f"{CRON_DELIVERABLE_MARKER}\n\n"
        "Good morning.\n"
    )
    saved = {}
    delivered = {}

    def fake_run_job(job, *, defer_agent_teardown=None, **_kwargs):
        return (True, "# full doc with scratch", body, None)

    def fake_save(jid, out):
        saved["output"] = out
        return tmp_path / "out.md"

    def fake_enqueue(execution_id, job, content, **_kwargs):
        delivered["content"] = content
        return {"status": "pending"}

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr("cron.delivery_queue.enqueue", fake_enqueue)
    monkeypatch.setattr(s, "mark_job_run", lambda *args, **kwargs: None)

    assert s.run_one_job(
        {
            "id": "job-think",
            "name": "daily-brief",
            "prompt": "Write the morning brief",
            "deliver": "telegram",
        }
    ) is True
    assert saved["output"] == "# full doc with scratch"
    assert delivered["content"].lstrip().startswith("Good morning.")
    assert "internal preamble" not in delivered["content"]
