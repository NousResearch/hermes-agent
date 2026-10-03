"""A failed review must survive the conversation-to-durable-report boundary."""

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "result, expected_error",
    [
        ({"failed": True, "error": "provider refused", "final_response": ""}, "provider refused"),
        ({"failed": True, "error": "provider refused", "final_response": "partial work"}, "provider refused"),
        ({"failed": True, "final_response": ""}, "review failed"),
        ({"error": "provider refused"}, "provider refused"),
        ({"failed": False, "error": None, "final_response": "nothing to change"}, None),
        ({"final_response": ""}, None),
        ("lock", None),
        ("transient_block", None),
        ({"failed": True, "compression_deferred": True, "error": "provider refused", "final_response": "partial work"}, "provider refused"),
        (RuntimeError("provider raised"), "error: provider raised"),
    ],
)
def test_review_outcome_reaches_durable_reports(result, expected_error, tmp_path, monkeypatch):
    from agent import curator
    from tools import skill_usage
    import run_agent

    if isinstance(result, str):
        from types import SimpleNamespace
        from agent.conversation_loop import _compression_deferred_result

        result = _compression_deferred_result(
            SimpleNamespace(session_id="review-fixture", _flush_status_buffer=lambda: None),
            [], 1, reason=result,
        )

    home = tmp_path / "home"
    skill = home / "skills" / "sample"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: sample\n---\nA sample skill.\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    skill_usage.mark_agent_created("sample")
    captured = {}

    class ReviewAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._session_messages = [{"tool_calls": [{"function": {"name": "skill_view", "arguments": "{}"}}]}]

        def run_conversation(self, **kwargs):
            captured["prompt"] = kwargs["user_message"]
            if isinstance(result, Exception):
                raise result
            return result

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(run_agent, "AIAgent", ReviewAgent)
    monkeypatch.setattr(curator, "_resolve_review_provider", lambda: ({"api_key": "fixture"}, "fixture-model", "custom", {}))
    summaries = []
    curator.run_curator_review(synchronous=True, dry_run=True, consolidate=True, on_summary=summaries.append)
    state = curator.load_state()
    report_dir = Path(state["last_report_path"])
    payload = json.loads((report_dir / "run.json").read_text(encoding="utf-8"))
    markdown = (report_dir / "REPORT.md").read_text(encoding="utf-8")
    assert captured.get("closed") is True, payload
    assert "sample" in captured["prompt"]
    assert payload["llm_error"] == expected_error
    if expected_error:
        assert "error" in payload["llm_summary"].lower()
        assert expected_error in markdown
        assert "llm: no change" not in state["last_run_summary"]
        assert "error" in summaries[-1].lower()
    else:
        assert payload["llm_summary"] == (result["final_response"] or "no change")
        assert "LLM pass error:" not in markdown
        if result.get("compression_deferred"):
            assert result["final_response"] in markdown
            assert result["final_response"] in state["last_run_summary"]
    if not isinstance(result, Exception):
        assert payload["llm_final"] == result.get("final_response", "")
        assert payload["tool_call_counts"] == {"skill_view": 1}
    assert payload["model"] == "fixture-model"
    assert payload["provider"] == "custom"
