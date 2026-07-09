import sys
from pathlib import Path

from hermes_state import SessionDB


def _run_hermes(monkeypatch, capsys, home: Path, *args: str) -> str:
    from hermes_cli import main as main_mod

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["hermes", *args])
    monkeypatch.setattr(main_mod, "_recover_from_interrupted_install", lambda: None)
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: None)
    main_mod.main()
    return capsys.readouterr().out


def test_sessions_copy_to_profile_copies_transcript_without_memories(tmp_path, monkeypatch, capsys):
    home = tmp_path / "hermes"
    target_home = home / "profiles" / "coder"
    target_home.mkdir(parents=True)
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("source-only memory\n", encoding="utf-8")

    source_db = SessionDB(db_path=home / "state.db")
    source_db.create_session(
        "sess-alpha",
        source="cli",
        model="test-model",
        model_config={"temperature": 0.2},
        system_prompt="source prompt",
        cwd="/tmp/project",
    )
    source_db.append_message("sess-alpha", role="user", content="hello", timestamp=10)
    source_db.append_message("sess-alpha", role="assistant", content="hi", timestamp=11)
    source_db.close()

    output = _run_hermes(
        monkeypatch,
        capsys,
        home,
        "sessions",
        "copy",
        "sess-alpha",
        "--from-profile",
        "default",
        "--to-profile",
        "coder",
    )

    assert "Copied session 'sess-alpha' from profile 'default' to profile 'coder'" in output

    target_db = SessionDB(db_path=target_home / "state.db")
    copied = target_db.get_session("sess-alpha")
    assert copied is not None
    assert copied["source"] == "cli"
    assert copied["model"] == "test-model"
    assert copied["system_prompt"] == "source prompt"
    assert copied["cwd"] == "/tmp/project"
    messages = target_db.get_messages("sess-alpha")
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]
    target_db.close()

    assert not (target_home / "memories" / "MEMORY.md").exists()


def test_sessions_copy_to_profile_generates_new_id_on_collision(tmp_path, monkeypatch, capsys):
    home = tmp_path / "hermes"
    target_home = home / "profiles" / "coder"
    target_home.mkdir(parents=True)

    source_db = SessionDB(db_path=home / "state.db")
    source_db.create_session("sess-alpha", source="cli")
    source_db.append_message("sess-alpha", role="user", content="source", timestamp=10)
    source_db.close()

    target_db = SessionDB(db_path=target_home / "state.db")
    target_db.create_session("sess-alpha", source="cli")
    target_db.append_message("sess-alpha", role="user", content="existing", timestamp=20)
    target_db.close()

    output = _run_hermes(
        monkeypatch,
        capsys,
        home,
        "sessions",
        "copy",
        "sess-alpha",
        "--to-profile",
        "coder",
    )

    assert "as 'sess-alpha-copy'" in output

    target_db = SessionDB(db_path=target_home / "state.db")
    assert target_db.get_session("sess-alpha") is not None
    assert target_db.get_session("sess-alpha-copy") is not None
    copied_messages = target_db.get_messages("sess-alpha-copy")
    assert [(m["role"], m["content"]) for m in copied_messages] == [("user", "source")]
    target_db.close()
