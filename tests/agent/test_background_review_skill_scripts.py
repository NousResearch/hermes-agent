"""Fixture coverage for execution-grounded script learning in automatic review."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent import background_review as review
from agent.tool_dispatch_helpers import make_tool_result_message
from tools import skill_manager_tool  # registers skill_manage
from tools.registry import registry
from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin


SCRIPT = 'import sys\n\nprint("\\n".join(sorted(set(sys.stdin.read().splitlines()))))\n'
SKILL = "---\nname: sort-lines\ndescription: Use when sorting and deduplicating text.\n---\n## Procedure\nSort the input lines and remove duplicates.\n"
POINTER = "Run `scripts/sort-lines.py` with the text on stdin; if unavailable, sort and deduplicate the lines directly."


@pytest.mark.parametrize(
    "scenario", ["skills", "combined", "blocked", "memory", "explicit", "failed", "tool-calls", "reused", "orphan", "malformed"],
)
def test_automatic_review_can_save_a_helper_or_keep_prose(tmp_path, monkeypatch, scenario):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"skills": {"write_approval": False}})
    completed = subprocess.run([sys.executable, "-c", SCRIPT], input="pear\napple\npear\n", text=True,
                               capture_output=True, check=True)
    result = {
        "status": "error" if scenario == "failed" else "success",
        "exit_code": completed.returncode,
        "output": completed.stdout,
        "tool_calls_made": int(scenario == "tool-calls"),
        "kernel": {"mode": "session", "reused": scenario == "reused", "execution_count": 1},
    }
    history: list[dict] = [{"role": "user", "content": "Sort and deduplicate these lines."}]
    if scenario != "orphan":
        history.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_sort", "type": "function", "function": {
                "name": "execute_code", "arguments": json.dumps({"code": SCRIPT})},
        }]})
    history.append(make_tool_result_message(
        "execute_code", "[invalid" if scenario == "malformed" else json.dumps(result), "call_sort"))

    def fixture_reviewer(_agent, snapshot, prompt, **_kwargs):
        assert snapshot == history  # Cached conversation prefix is untouched.
        if scenario == "memory":
            assert "Script extraction candidate" not in prompt
            return
        operations = [{"action": "create", "content": SKILL}]
        # Model substitute: emit the existing tool call only when invited by the review request.
        if "Script extraction candidate" in prompt:
            operations[0]["content"] += POINTER + "\n"
            path = "unsupported/sort-lines.py" if scenario == "blocked" else "scripts/sort-lines.py"
            operations.append({"action": "write_file", "file_path": path, "file_content": SCRIPT})
        origin = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            raw = registry.dispatch("skill_manage", {"name": "sort-lines", "operations": operations})
            assert isinstance(raw, str)
            response = json.loads(raw)
            if scenario == "blocked":
                assert len(operations) == 2 and not response["success"]
                assert not (tmp_path / "skills" / "sort-lines" / "SKILL.md").exists()
                raw = registry.dispatch("skill_manage", {"name": "sort-lines", "action": "create", "content": SKILL})
                assert isinstance(raw, str)
                response = json.loads(raw)
        finally:
            reset_current_write_origin(origin)
        assert response["success"], response

    monkeypatch.setattr(review, "_run_review_in_thread", fixture_reviewer)
    target, _ = review.spawn_background_review_thread(
        SimpleNamespace(), history, review_memory=scenario in {"combined", "memory"},
        review_skills=scenario != "memory", explicit=scenario == "explicit", task_cfg={},
    )
    target()
    skill_dir = tmp_path / "skills" / "sort-lines"
    script = skill_dir / "scripts" / "sort-lines.py"
    if scenario in {"skills", "combined"}:
        assert POINTER in (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        reused = subprocess.run([sys.executable, str(script)], input="zebra\nant\nzebra\n", text=True,
                                capture_output=True, check=True)
        assert reused.stdout == "ant\nzebra\n"
    elif scenario == "memory":
        assert not skill_dir.exists()
    else:
        assert not script.exists()
        assert (skill_dir / "SKILL.md").read_text(encoding="utf-8").endswith("remove duplicates.\n")
