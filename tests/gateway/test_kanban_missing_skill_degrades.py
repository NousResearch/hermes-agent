"""An unresolvable preload skill must not kill a Kanban worker before its first model call.

The dispatcher force-loads ``sdlc-review`` on every review-lane card. When that skill is the only
requested one and the reviewer profile cannot resolve it, the turn used to raise
``kanban_skills_unavailable`` — the worker exited 1 with no verdict, and the board booked a crash
(6 of 26 qa runs). A missing skill body is a smaller failure than a dead reviewer: warn and run.
"""
from types import SimpleNamespace

from agent import skill_commands
from gateway import session_kanban


class _Agent:
    session_id = "local-test-worker"

    def __init__(self):
        self.prompts = []

    def run_conversation(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return {"completed": True, "final_response": "reviewed"}


def test_only_skill_missing_warns_and_still_runs_the_turn(monkeypatch, caplog):
    monkeypatch.setattr(skill_commands, "build_preloaded_skills_prompt",
                        lambda *_a, **_k: ("", [], ["sdlc-review"]))
    agent = _Agent()
    frame = {"text": "work kanban task t_x"}
    context = {"skills": ["sdlc-review"], "context": "card body", "task_id": "t_x", "goal_mode": False}

    result = session_kanban._run_task_turns(agent, frame, [], context)

    assert result == {"completed": True, "final_response": "reviewed"}
    assert agent.prompts == ["card body\n\nwork kanban task t_x"]
    assert any("continuing without: sdlc-review" in record.getMessage() for record in caplog.records)


def test_loaded_and_missing_mix_runs_with_the_loaded_block(monkeypatch):
    monkeypatch.setattr(skill_commands, "build_preloaded_skills_prompt",
                        lambda *_a, **_k: ("[skill: sdlc-review]", ["sdlc-review"], ["typo-skill"]))
    agent = _Agent()
    context = {"skills": ["sdlc-review", "typo-skill"], "context": "card body",
               "task_id": "t_y", "goal_mode": False}

    session_kanban._run_task_turns(agent, {"text": "request"}, [], context)

    assert agent.prompts == ["[skill: sdlc-review]\n\ncard body\n\nrequest"]


def test_full_resolution_is_unchanged(monkeypatch):
    monkeypatch.setattr(skill_commands, "build_preloaded_skills_prompt",
                        lambda *_a, **_k: ("[skill block]", ["sdlc-review"], []))
    agent = _Agent()
    context = {"skills": ["sdlc-review"], "context": "", "task_id": "t_z", "goal_mode": False}

    session_kanban._run_task_turns(agent, {"text": "request"}, [], context)

    assert agent.prompts == ["[skill block]\n\nrequest"]
