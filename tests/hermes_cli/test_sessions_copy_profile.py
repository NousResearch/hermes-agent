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


def test_sessions_copy_to_profile_clears_missing_parent_lineage(tmp_path, monkeypatch, capsys):
    home = tmp_path / "hermes"
    target_home = home / "profiles" / "coder"
    target_home.mkdir(parents=True)

    source_db = SessionDB(db_path=home / "state.db")
    source_db.create_session("parent-source", source="cli")
    source_db.create_session("child-source", source="cli", parent_session_id="parent-source")
    source_db.append_message("child-source", role="user", content="child transcript", timestamp=10)
    source_db.close()

    _run_hermes(
        monkeypatch,
        capsys,
        home,
        "sessions",
        "copy",
        "child-source",
        "--to-profile",
        "coder",
    )

    target_db = SessionDB(db_path=target_home / "state.db")
    copied = target_db.get_session("child-source")
    assert copied is not None
    assert copied["parent_session_id"] is None
    assert [(m["role"], m["content"]) for m in target_db.get_messages("child-source")] == [
        ("user", "child transcript"),
    ]
    target_db.close()


def test_sessions_copy_to_profile_does_not_link_to_colliding_target_parent(tmp_path, monkeypatch, capsys):
    home = tmp_path / "hermes"
    target_home = home / "profiles" / "coder"
    target_home.mkdir(parents=True)

    source_db = SessionDB(db_path=home / "state.db")
    source_db.create_session("shared-parent-id", source="cli")
    source_db.create_session("child-source", source="cli", parent_session_id="shared-parent-id")
    source_db.close()

    target_db = SessionDB(db_path=target_home / "state.db")
    target_db.create_session("shared-parent-id", source="cli")
    target_db.close()

    _run_hermes(
        monkeypatch,
        capsys,
        home,
        "sessions",
        "copy",
        "child-source",
        "--to-profile",
        "coder",
    )

    target_db = SessionDB(db_path=target_home / "state.db")
    copied = target_db.get_session("child-source")
    assert copied is not None
    assert copied["parent_session_id"] is None
    target_db.close()


def test_sessions_copy_to_profile_preserves_insertion_order_not_timestamp_order(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "hermes"
    target_home = home / "profiles" / "coder"
    target_home.mkdir(parents=True)

    source_db = SessionDB(db_path=home / "state.db")
    source_db.create_session("sess-alpha", source="cli")
    source_db.append_message("sess-alpha", role="user", content="call the tool", timestamp=30)
    source_db.append_message(
        "sess-alpha",
        role="assistant",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
        timestamp=20,
    )
    source_db.append_message(
        "sess-alpha",
        role="tool",
        content='{"ok": true}',
        tool_name="terminal",
        tool_call_id="call-1",
        timestamp=10,
    )
    source_db.append_message("sess-alpha", role="assistant", content="done", timestamp=40)
    source_db.close()

    _run_hermes(
        monkeypatch,
        capsys,
        home,
        "sessions",
        "copy",
        "sess-alpha",
        "--to-profile",
        "coder",
    )

    target_db = SessionDB(db_path=target_home / "state.db")
    messages = target_db.get_messages("sess-alpha")
    assert [(m["role"], m["content"], m.get("tool_call_id")) for m in messages] == [
        ("user", "call the tool", None),
        ("assistant", None, None),
        ("tool", '{"ok": true}', "call-1"),
        ("assistant", "done", None),
    ]
    target_db.close()
