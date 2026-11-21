"""skill_view's repeat-view stub after the executor delivered only a spill preview of the first view.

A skill_view result over the per-result budget, or one the per-turn budget spills out of a parallel batch,
reaches the model as a short ``<persisted-output>`` preview plus a file path. A later view of the same
unchanged skill must lead back to the content (a preview again, or the guardrail's reference to the spill
file), never the "unchanged, refer to the earlier skill_view result; it is still current and complete"
stub: the model never held that result.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.skills_tool_dedup import reset_skill_view_dedup
from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, extract_persisted_path

TASK = "skill-spill-task"
STUB_MARKER = '"content_returned": false'


def _write_skill(home, name: str, chars: int) -> None:
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True)
    head = f"---\nname: {name}\ndescription: Spill probe.\n---\n# {name}\n\n"
    line = f"Step for {name}: follow the procedure exactly.\n"
    (skill_dir / "SKILL.md").write_text(head + line * ((chars - len(head)) // len(line)), encoding="utf-8")


def _views(prefix: str, *names: str) -> list:
    return [SimpleNamespace(id=f"{prefix}_{name}", type="function",
                            function=SimpleNamespace(name="skill_view", arguments=json.dumps({"name": name})))
            for name in names]


def _response(tool_calls=None):
    message = SimpleNamespace(content=None if tool_calls else "done", reasoning_content=None, reasoning=None,
                              tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message,
                                                    finish_reason="tool_calls" if tool_calls else "stop")],
                           model="test/model", usage=None)


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    yield home
    reset_skill_view_dedup()


def _run(home, monkeypatch, *batches, dedup_reset_before: tuple[int, ...] = ()) -> dict:
    """Run one turn whose model calls each batch of skill_view calls in turn; tool rows by call id.
    ``dedup_reset_before`` lists model calls (0-based) before which the dedup bucket is reset, as a
    mid-turn compaction does (the guardrail's per-turn counts survive it)."""
    tool_defs = [{"type": "function", "function": {
        "name": "skill_view", "description": "skill_view",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}}]
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **kw: tool_defs)
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **kw: {})
    with patch("agent.process_bootstrap.OpenAI"):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, max_iterations=len(batches) + 2)
    agent.client = MagicMock()
    responses = iter([*(_response(b) for b in batches), _response()])
    model_calls = iter(range(len(batches) + 1))

    def respond(*_args, **_kwargs):
        if next(model_calls) in dedup_reset_before:
            reset_skill_view_dedup(TASK)
        return next(responses)

    agent.client.chat.completions.create.side_effect = respond
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = False  # no compaction or prune: either would release the dedup on its own
    agent.context_compressor.context_length = 1_000_000  # the 200K+ budget: 100K per result, 200K per turn
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("load the skills", task_id=TASK)
    assert result["completed"] is True
    return {m["tool_call_id"]: m["content"] for m in result["messages"] if m.get("role") == "tool"}


def test_views_after_a_per_result_spill_lead_back_to_the_skill(home, monkeypatch):
    _write_skill(home, "big-skill", 150_000)

    rows = _run(home, monkeypatch, *(_views(view, "big-skill") for view in ("first", "again", "third")))

    spill_path = extract_persisted_path(rows["first_big-skill"])
    assert spill_path
    for view in ("again_big-skill", "third_big-skill"):
        assert STUB_MARKER not in rows[view]
        assert PERSISTED_OUTPUT_TAG in rows[view] or spill_path in rows[view]


def test_view_after_a_turn_budget_spill_serves_only_the_spilled_skill_again(home, monkeypatch):
    for name, chars in (("skill-a", 70_000), ("skill-b", 70_000), ("skill-c", 80_000)):
        _write_skill(home, name, chars)

    rows = _run(home, monkeypatch, _views("first", "skill-a", "skill-b", "skill-c"),
                _views("again", "skill-a", "skill-c"))

    assert PERSISTED_OUTPUT_TAG not in rows["first_skill-a"]
    assert PERSISTED_OUTPUT_TAG in rows["first_skill-c"]
    assert "Step for skill-c" in rows["again_skill-c"] and STUB_MARKER not in rows["again_skill-c"]
    assert STUB_MARKER in rows["again_skill-a"]


def test_view_delivered_whole_with_a_guardrail_warning_keeps_its_entry(home, monkeypatch):
    # The reset before the second model call stands in for a mid-turn compaction: the dedup bucket is
    # cleared, the guardrail still counts skill-a's earlier identical result, so the next full view of
    # skill-a reaches the model whole with a "[Tool loop warning ...]" suffix appended.
    for name in ("skill-a", "skill-b"):
        _write_skill(home, name, 2_000)

    rows = _run(home, monkeypatch, _views("first", "skill-a", "skill-b"), _views("again", "skill-a"),
                _views("third", "skill-a"), dedup_reset_before=(1,))

    assert "Step for skill-a" in rows["again_skill-a"] and "Tool loop warning" in rows["again_skill-a"]
    assert STUB_MARKER in rows["third_skill-a"]


def test_dedup_stub_row_does_not_release_the_entry_it_points_at(home, monkeypatch):
    # The stub is never the recorded result, so a release keyed on the stub's own call would drop the
    # entry and the third view would serve the whole skill again.
    _write_skill(home, "small-skill", 900)

    rows = _run(home, monkeypatch, *(_views(view, "small-skill") for view in ("first", "again", "third")))

    assert "Step for small-skill" in rows["first_small-skill"]
    for view in ("again_small-skill", "third_small-skill"):
        assert STUB_MARKER in rows[view] and "Step for small-skill" not in rows[view]
