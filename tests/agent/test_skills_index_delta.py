"""Skills that change mid-conversation reach the model, behind the cached system prompt.

The ``<available_skills>`` index is frozen in the system prompt for the prefix cache, so the change
is delivered on the per-turn note channel (cloudflare/cloudflare-os#267 port).
"""
import subprocess
import sys
from types import SimpleNamespace

from agent import prompt_builder as pb
from agent.skills_index_delta import stage_skills_index_note
from agent.turn_context import consume_skills_index_note


def _write_skill(skills, name, description):
    path = skills / "devops" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\nBody.\n", encoding="utf-8")


def _agent(home):
    return SimpleNamespace(valid_tool_names={"skills_list", "skill_view", "skill_manage"}, platform="cli",
                           provider="openrouter", api_mode="chat_completions", session_id="s1",
                           _session_db=SimpleNamespace(db_path=str(home / "state.db")))


def _turn(agent, prompt, history):
    """One turn: stage, then the consume ``_merge_gateway_notes`` does; returns the note sent."""
    stage_skills_index_note(agent, prompt, history)
    note = consume_skills_index_note(agent)
    history.append({"role": "user", "content": "hi", **({"api_content": f"hi\n\n{note}"} if note else {})})
    history.append({"role": "assistant", "content": "ok"})
    return note


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    return tmp_path / "skills"


def test_skill_installed_by_another_process_reaches_a_running_process(tmp_path, monkeypatch):
    """The in-process index cache is keyed on skill files: a write this process never saw (a hub
    install in another terminal) is not answered from the pre-install cache entry (#92313)."""
    skills = _home(tmp_path, monkeypatch)
    _write_skill(skills, "docker-ops", "Run containers.")
    assert "docker-ops" in pb.build_skills_system_prompt()
    code = ("import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True); "
            "p.write_text('---\\nname: terraform-ops\\ndescription: Plan infra.\\n---\\nBody.\\n')")
    subprocess.run([sys.executable, "-c", code, str(skills / "devops" / "terraform-ops" / "SKILL.md")], check=True)
    assert "- terraform-ops: Plan infra." in pb.build_skills_system_prompt()


def test_long_lived_agent_is_told_each_skill_change_once(tmp_path, monkeypatch):
    """Every turn checks the index (a long-lived agent never rebuilds its prompt). The note names
    only what changed, is sent once per change (a fresh agent reading the transcript included,
    however far back the note is), and retires itself when the change is undone."""
    skills = _home(tmp_path, monkeypatch)
    _write_skill(skills, "docker-ops", "Run containers.")
    agent, history = _agent(tmp_path), []
    prompt = "Identity.\n\n" + pb.build_skills_system_prompt()
    assert _turn(agent, prompt, history) == ""

    _write_skill(skills, "terraform-ops", "Plan infra.")
    note = _turn(agent, prompt, history)
    assert "    - terraform-ops: Plan infra." in note and "docker-ops" not in note
    assert _turn(agent, prompt, history) == ""
    history += [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}] * 150
    assert _turn(_agent(tmp_path), prompt, history) == ""

    (skills / "devops" / "docker-ops" / "SKILL.md").unlink()
    note = _turn(agent, prompt, history)
    assert "    - terraform-ops: Plan infra." in note and note.endswith("No longer available: docker-ops]")

    (skills / "devops" / "terraform-ops" / "SKILL.md").unlink()
    _write_skill(skills, "docker-ops", "Run containers.")
    assert "accurate again" in _turn(agent, prompt, history)
    assert _turn(_agent(tmp_path), prompt, history) == ""

    # A compaction rebuild that now lists the announced skill needs no new note.
    _write_skill(skills, "terraform-ops", "Plan infra.")
    assert "terraform-ops" in _turn(agent, prompt, history)
    rebuilt = "Identity.\n\n" + pb.build_skills_system_prompt()
    assert _turn(agent, rebuilt, history) == "" and _turn(_agent(tmp_path), rebuilt, history) == ""
