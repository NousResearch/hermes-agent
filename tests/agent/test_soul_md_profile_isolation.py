"""Regression (#50233, SOUL.md half): a profile agent's SOUL.md must load from
ITS OWN home, never the ambient/launch home — even on a thread that did not
bind the HERMES_HOME ContextVar. Same bug class as the skills-index leak
fixed in #86313; load_soul_md now accepts home_override.
"""

import threading


def test_soul_md_scoped_to_home_override_not_ambient(tmp_path, monkeypatch):
    from agent import prompt_builder

    default_home = tmp_path / "default"
    default_home.mkdir()
    (default_home / "SOUL.md").write_text(
        "DEFAULT SOUL — must never leak into a profile prompt", encoding="utf-8"
    )

    bot_home = tmp_path / "profiles" / "mybot"
    bot_home.mkdir(parents=True)
    (bot_home / "SOUL.md").write_text("BOT SOUL", encoding="utf-8")

    # Ambient home points at default (mimics a build thread that lost the
    # profile's ContextVar override and fell back to launch).
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    result = {}

    def build():
        result["soul"] = prompt_builder.load_soul_md(home_override=bot_home)

    t = threading.Thread(target=build)
    t.start()
    t.join()

    assert result["soul"] == "BOT SOUL"


def test_soul_md_override_missing_file_returns_none(tmp_path, monkeypatch):
    from agent import prompt_builder

    default_home = tmp_path / "default"
    default_home.mkdir()
    (default_home / "SOUL.md").write_text("DEFAULT SOUL", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    empty_bot = tmp_path / "profiles" / "emptybot"
    empty_bot.mkdir(parents=True)

    # No SOUL.md in the bot home -> None, NOT default's soul.
    assert prompt_builder.load_soul_md(home_override=empty_bot) is None


def test_soul_md_ambient_unchanged_without_override(tmp_path, monkeypatch):
    from agent import prompt_builder

    home = tmp_path / "home"
    home.mkdir()
    (home / "SOUL.md").write_text("AMBIENT SOUL", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert prompt_builder.load_soul_md() == "AMBIENT SOUL"
