"""The skills delta reads the prompt's real ``<available_skills>`` index, not a mention of it.

Context files (AGENTS.md, SOUL.md, …) are injected ahead of the volatile band, and the skills index
leads that band, so the index is the LAST block in the prompt and it starts its own line. A prose
mention of the tag — or a fenced example of the block — used to be parsed as the index, which made
every ``- `` bullet after the mention a fake "skill" and told the model its own rules were
"no longer available" (#151915).

The helpers below are local copies of ``tests/agent/test_skills_index_delta.py``'s fixtures (a new
test module cannot import another module's fixture helpers without tripping the health gate's F811).
"""
from types import SimpleNamespace

from agent import prompt_builder as pb
from agent.skills_index_delta import index_entries, stage_skills_index_note
from agent.turn_context import consume_skills_index_note

CONTEXT_FILE_MENTION = (
    "# Project Context\n"
    "The list appears in `<available_skills>` twice.\n"
    "- rule one: text\n"
    "- rule two\n"
)

CONTEXT_FILE_EXAMPLE = (
    "# Project Context\n"
    "Example of the block:\n"
    "<available_skills>\n"
    "- fake-one: x\n"
    "</available_skills>\n"
)


def _write_skill(skills, name, description):
    path = skills / "devops" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\nBody.\n", encoding="utf-8")


def _agent(home):
    return SimpleNamespace(valid_tool_names={"skills_list", "skill_view", "skill_manage"}, platform="cli",
                           provider="openrouter", api_mode="chat_completions", session_id="s1",
                           _session_db=SimpleNamespace(db_path=str(home / "state.db")))


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    return tmp_path / "skills"


def test_context_file_mention_contributes_no_index_rows():
    """Bullets that follow a prose mention belong to the context file, not to the index."""
    prompt = CONTEXT_FILE_MENTION + "\n## Skills\n<available_skills>\n  cat:\n    - real-skill: desc\n</available_skills>\n"
    assert set(index_entries(prompt)) == {"real-skill"}


def test_prose_example_block_does_not_shadow_the_real_index():
    """A context file that quotes the block verbatim must not replace the index it precedes."""
    prompt = CONTEXT_FILE_EXAMPLE + "\n## Skills\n<available_skills>\n    - real-skill: desc\n</available_skills>\n"
    assert set(index_entries(prompt)) == {"real-skill"}


def test_inline_mention_alone_is_not_a_baseline():
    """Without a real block there is no baseline to diff against (the module's documented contract)."""
    assert index_entries(CONTEXT_FILE_MENTION) == {}


def test_no_skills_note_when_a_context_file_mentions_the_block(tmp_path, monkeypatch):
    """The user-visible symptom: no "No longer available: rule one, rule two" note on every turn."""
    skills = _home(tmp_path, monkeypatch)
    _write_skill(skills, "docker-ops", "Run containers.")
    prompt = CONTEXT_FILE_MENTION + "\n" + pb.build_skills_system_prompt()
    agent = _agent(tmp_path)
    assert stage_skills_index_note(agent, prompt, []) is False
    assert consume_skills_index_note(agent) == ""
