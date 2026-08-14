"""Regression: a bot profile's system prompt must reflect ITS OWN skills/home,
never the launch (default) profile's — even when the agent build runs on a
thread that did not bind the HERMES_HOME ContextVar.

Root cause this guards (confirmed empirically): ContextVars do not propagate
into ``threading.Thread``. ``build_skills_system_prompt`` and the
active-profile line resolved the home via the ambient ``get_hermes_home()``,
so an unbound build thread fell back to ``~/.hermes`` (default) and leaked
default's full skills index + "Active Hermes profile: default" into a bot's
prompt, while the live ``skills_list()`` (re-bound per turn) correctly
showed the bot's real, empty set. The agent now resolves its own home from
its ``_session_db.db_path`` and passes it explicitly.
"""

import re
import threading

import pytest


def _skills_body(prompt: str) -> str:
    m = re.search(r"<available_skills>(.*?)</available_skills>", prompt, re.DOTALL)
    return (m.group(1).strip() if m else "")


def test_skills_prompt_scoped_to_override_not_ambient_home(tmp_path, monkeypatch):
    """An explicit skills_dir_override wins over ambient HERMES_HOME, on a
    bare thread with no override bound."""
    from agent import prompt_builder

    # A "default" home WITH skills (the thing that must NOT leak).
    default_home = tmp_path / "default"
    default_skills = default_home / "skills" / "general" / "leaky-skill"
    default_skills.mkdir(parents=True)
    (default_skills / "SKILL.md").write_text(
        "---\nname: leaky-skill\ndescription: should never appear in a bot prompt\n---\nbody\n",
        encoding="utf-8",
    )

    # An empty bot profile (no skills dir at all).
    bot_skills = tmp_path / "profiles" / "emptybot" / "skills"

    # Bind ambient home to default (mimics a build thread that lost the
    # bot's override and fell back to launch).
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=False)

    result = {}

    def build():
        # No set_hermes_home_override on THIS thread — ambient resolves to
        # default. The override arg must still scope to the empty bot.
        result["bot"] = _skills_body(
            prompt_builder.build_skills_system_prompt(skills_dir_override=bot_skills)
        )

    t = threading.Thread(target=build)
    t.start()
    t.join()

    assert result["bot"] == "", (
        "empty bot profile leaked skills from the ambient (default) home: "
        + result["bot"][:200]
    )


def test_agent_home_resolves_from_session_db_path(tmp_path):
    """The agent's own home is read from its session_db, independent of any
    ContextVar."""
    from agent import system_prompt

    bot_home = tmp_path / "profiles" / "mybot"
    bot_home.mkdir(parents=True)

    class _DB:
        db_path = bot_home / "state.db"

    class _Agent:
        _session_db = _DB()

    assert system_prompt._agent_home(_Agent()) == bot_home
    assert system_prompt._agent_skills_dir(_Agent()) == bot_home / "skills"


def test_agent_home_none_without_session_db():
    from agent import system_prompt

    class _Agent:
        _session_db = None

    assert system_prompt._agent_home(_Agent()) is None
    assert system_prompt._agent_skills_dir(_Agent()) is None
