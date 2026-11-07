"""Per-turn skill shortlist: suggestions ride the current user message, never the cached prompt."""
from types import SimpleNamespace

from agent import prompt_builder as pb
from agent.skill_selection import stage_skill_shortlist_note
from agent.turn_context import _pop_turn_note


def _write_skill(skills, name, description, situations=()):
    path = skills / "devops" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = "".join(f"      - {s}\n" for s in situations)
    meta = f"metadata:\n  hermes:\n    situations:\n{extra}" if situations else ""
    path.write_text(f"---\nname: {name}\ndescription: {description}\n{meta}---\nBody.\n", encoding="utf-8")


def _setup(tmp_path, monkeypatch, selection):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    if selection:
        (tmp_path / "config.yaml").write_text(f"skills:\n  selection: {selection}\n", encoding="utf-8")
    skills = tmp_path / "skills"
    _write_skill(skills, "merge-repair", "Untangle a broken branch history.",
                 situations=["my pull request has merge conflicts"])
    _write_skill(skills, "docker-ops", "Run and inspect containers.")
    return "Identity.\n\n" + pb.build_skills_system_prompt()


def _agent(home):
    return SimpleNamespace(valid_tool_names={"skills_list", "skill_view", "skill_manage"}, platform="cli",
                           provider="openrouter", api_mode="chat_completions", session_id="s1",
                           _hermes_home=str(home), _session_db=SimpleNamespace(db_path=str(home / "state.db")))


def _turn(agent, prompt, history, text):
    """One turn as build_turn_context runs it: stage, consume on the note channel, record the row
    with the note in its api_content sidecar only. Returns the note sent."""
    history.append({"role": "user", "content": text})
    stage_skill_shortlist_note(agent, prompt, history, text)
    note = _pop_turn_note(agent, "_skill_shortlist_note")
    if note:
        history[-1]["api_content"] = f"{text}\n\n{note}"
    history.append({"role": "assistant", "content": "ok"})
    return note


def test_shortlist_suggests_once_per_session_from_situations_never_rendered_in_prompt(tmp_path, monkeypatch):
    """``situations`` feed the ranking but never the cached index; the suggestion is a note on the
    current turn (sidecar only, durable content clean), shown once per session — including to a
    fresh per-turn gateway agent reading the transcript — and small talk gets none."""
    prompt = _setup(tmp_path, monkeypatch, "shortlist")
    assert "merge conflicts" not in prompt and "- merge-repair:" in prompt
    agent, history = _agent(tmp_path), []

    note = _turn(agent, prompt, history, "my pull request has merge conflicts again, can you sort it out")
    assert note.startswith("[System: Skills that may fit this request")
    assert "    - merge-repair: Untangle a broken branch history.]" in note and "docker-ops" not in note
    assert history[0]["content"] == "my pull request has merge conflicts again, can you sort it out"

    assert _turn(agent, prompt, history, "the pull request still has merge conflicts") == ""
    assert _turn(_agent(tmp_path), prompt, history, "pull request merge conflicts, please fix") == ""
    assert _turn(_agent(tmp_path), prompt, [], "thanks, that's perfect!") == ""


def test_legacy_default_sends_no_suggestion(tmp_path, monkeypatch):
    prompt = _setup(tmp_path, monkeypatch, None)
    assert _turn(_agent(tmp_path), prompt, [], "my pull request has merge conflicts again") == ""


def test_malformed_situations_in_one_skill_do_not_silence_the_others(tmp_path, monkeypatch):
    prompt = _setup(tmp_path, monkeypatch, "shortlist")
    bad = tmp_path / "skills" / "devops" / "docker-ops" / "SKILL.md"
    bad.write_text("---\nname: docker-ops\ndescription: Run and inspect containers.\ntriggers: true\n---\nBody.\n",
                   encoding="utf-8")
    note = _turn(_agent(tmp_path), prompt, [], "my pull request has merge conflicts again, can you sort it out")
    assert "    - merge-repair:" in note


def test_ranks_the_resolved_catalog_the_session_can_load_now(tmp_path, monkeypatch):
    """A same-tier name collision is listed by exact path and still ranks on its situations; a skill
    deleted after the prompt was built is never suggested from the prompt's stale row."""
    _setup(tmp_path, monkeypatch, "shortlist")
    skills = tmp_path / "skills"
    for category in ("ops", "infra"):
        path = skills / category / "rollout" / "SKILL.md"
        path.parent.mkdir(parents=True)
        situation = "my kubernetes canary is stuck halfway" if category == "infra" else "ship the release"
        path.write_text(f"---\nname: rollout\ndescription: {category} rollouts.\nmetadata:\n  hermes:\n"
                        f"    situations:\n      - {situation}\n---\nBody.\n", encoding="utf-8")
    pb.clear_skills_system_prompt_cache()
    prompt = "Identity.\n\n" + pb.build_skills_system_prompt()
    assert "- infra/rollout:" in prompt
    assert "    - infra/rollout:" in _turn(_agent(tmp_path), prompt, [], "the kubernetes canary is stuck halfway again")

    (skills / "devops" / "merge-repair" / "SKILL.md").unlink()
    pb.clear_skills_system_prompt_cache()
    assert _turn(_agent(tmp_path), prompt, [], "my pull request has merge conflicts again, can you sort it out") == ""


def test_skills_loaded_without_skill_view_are_not_suggested_again(tmp_path, monkeypatch):
    """A gateway auto-load / slash turn and ``skills.auto_load`` load a skill with no skill_view call."""
    prompt = _setup(tmp_path, monkeypatch, "shortlist")
    ask = "my pull request has merge conflicts again, can you sort it out"
    history = [{"role": "user", "content": '[IMPORTANT: The "merge-repair" skill is auto-loaded. '
                                           "Follow its instructions for this session.]\n\nBody."}]
    assert _turn(_agent(tmp_path), prompt, history, ask) == ""

    (tmp_path / "config.yaml").write_text("skills:\n  selection: shortlist\n  auto_load: [merge-repair]\n",
                                          encoding="utf-8")
    assert _turn(_agent(tmp_path), prompt, [], ask) == ""
