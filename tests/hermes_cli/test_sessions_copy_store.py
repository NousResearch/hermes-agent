"""Single-session copy contracts: content-addressed payloads and atomic store isolation."""

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target_home = home / "profiles" / "coder"
    target_home.mkdir(parents=True)
    (target_home / "config.yaml").write_text("model: {}\n")
    source = SessionDB(db_path=home / "state.db")
    target = SessionDB(db_path=target_home / "state.db")
    try:
        yield source, target
    finally:
        source.close()
        target.close()


def test_copy_resume_does_not_restore_source_approval_bypass(stores):
    from types import SimpleNamespace
    from hermes_cli.cli_session_mixin import CLISessionMixin
    from tools import approval

    source, target = stores
    source.create_session("original", source="cli", model_config={"temperature": 0.2})
    source.set_session_yolo("original", True)
    copied_id = source.copy_session_to("original", target, new_session_id="safe-copy")
    # Exercise the same persisted metadata and restore consumer used by CLI resume.
    resumed = SimpleNamespace(session_id=copied_id)
    try:
        CLISessionMixin._restore_session_yolo(resumed, target.get_session(copied_id), quiet=True)
        assert not approval.is_session_yolo_enabled(copied_id)
        assert not target.session_yolo_enabled(target.get_session(copied_id))
        assert source.session_yolo_enabled(source.get_session("original"))
        assert json.loads(target.get_session(copied_id)["model_config"]) == {"temperature": 0.2}
    finally:
        approval.disable_session_yolo(copied_id)


@pytest.mark.parametrize("runtime_key", [
    "codex_thread_id", "_proactive_prune_rearm_tokens", "room_plumbing",
    "follow_profile_config", "composer_override_profile",
])
def test_copy_clears_source_runtime_bindings_but_keeps_model_settings(stores, runtime_key):
    from types import SimpleNamespace
    from agent.codex_runtime import _stored_codex_thread_id

    source, target = stores
    settings = {"model": "example/model", "provider": "custom:example", "base_url": "https://example.invalid/v1",
                "api_mode": "chat_completions", "temperature": 0.2, "service_tier": "normal",
                "reasoning_config": {"effort": "high"}, "gateway_runtime": {"provider": "custom:example"},
                "browser_model_lock": {"model": "example/model", "provider": "custom:example", "confirmed": True}}
    runtime_values = {"codex_thread_id": "source-thread", "_proactive_prune_rearm_tokens": 12345,
                      "room_plumbing": True, "follow_profile_config": True,
                      "composer_override_profile": {"model": "source/default", "provider": "nous"}}
    source.create_session("original", source="cli", model_config={**settings, runtime_key: runtime_values[runtime_key]})
    copied_id = source.copy_session_to("original", target)
    if runtime_key == "codex_thread_id":
        assert _stored_codex_thread_id(SimpleNamespace(_session_db=target, session_id=copied_id)) is None
    assert target.get_session_model_config_value(copied_id, runtime_key) is None
    assert json.loads(target.get_session(copied_id)["model_config"]) == settings
    assert source.get_session_model_config_value("original", runtime_key) == runtime_values[runtime_key]


@pytest.mark.parametrize("title,hidden,markers,expected_hidden", [
    ("Group: legacy room", True, {}, False),
    ("  Group: legacy room  ", True, {}, False),
    ("Group: marked room", True, {"room_plumbing": True}, False),
    ("My hidden scratchpad", True, {}, True),
    (None, True, {}, True),
    ("group: ordinary chat", True, {}, True),
    ("Group: visible ordinary chat", False, {}, False),
    ("Bot Chat", True, {"follow_profile_config": True}, True),
])
def test_copy_restores_runtime_without_legacy_room_identity(stores, title, hidden, markers, expected_hidden):
    from tui_gateway.server import _row_follows_profile, _stored_session_runtime_overrides

    source, target = stores
    settings = {"provider": "ollama-cloud", "api_mode": "chat_completions",
                "reasoning_config": {"effort": "high"}, "service_tier": "normal", "temperature": 0.2}
    source.create_session("original", source="cli", model="glm-5.1", model_config={**settings, **markers})
    if title is not None:
        source.set_session_title("original", title)
        # Older stored rows can predate the title setter's whitespace normalization.
        source._write_sql("UPDATE sessions SET title = ? WHERE id = ?", (title, "original"))
    source.set_session_hidden("original", hidden)
    before = source.get_session("original")
    copied_id = source.copy_session_to("original", target)
    copied = target.get_session(copied_id)
    # Actual resume consumer: stripping the marker alone must not discard model/provider settings.
    assert _stored_session_runtime_overrides(copied) == {
        "model_override": {"model": "glm-5.1", "provider": "ollama-cloud",
                           "base_url": None, "api_mode": "chat_completions"},
        "provider_override": "ollama-cloud", "reasoning_config_override": {"effort": "high"},
        "service_tier_override": "",
    }
    assert not _row_follows_profile(copied)
    assert copied["hidden"] == expected_hidden
    assert copied["title"] == (f"{title} (copy)" if title else None)
    assert json.loads(copied["model_config"]) == settings
    assert source.get_session("original") == before


@pytest.mark.parametrize("destination", ["empty", "populated", "same-store"])
@pytest.mark.parametrize("include_inactive", [True, False])
def test_copy_then_append_keeps_local_display_order_and_logical_groups(stores, destination, include_inactive):
    source, target = stores
    source.create_session("original", source="cli")
    # Real autoincrement IDs far above the destination's, not invented display values.
    source._write_sql("INSERT INTO sqlite_sequence(name, seq) VALUES ('messages', 1000)")
    source.append_message("original", "user", "question", timestamp=40)
    calls = [{"id": "call", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]
    source.append_message("original", "assistant", "checking", tool_calls=calls, timestamp=30)
    source.append_message("original", "tool", "result", tool_call_id="call", tool_name="terminal", timestamp=20)
    source.append_message("original", "assistant", "answer", timestamp=10)
    live = source.get_messages_as_conversation("original")
    source.archive_and_compact("original", [
        {"role": "user", "content": "summary", "display_metadata": {"model_only": True}},
    ] + live[-2:], tail_count=2)
    # Another representation of the carried tool/answer remains display-visible.
    source._write_sql("UPDATE messages SET compacted=1 WHERE session_id='original' AND active=0")
    rewind = source.append_message("original", "user", "rewound", timestamp=50)
    source.deactivate_message("original", rewind)
    expected = [("user", "question"), ("assistant", "checking"), ("tool", "result"), ("assistant", "answer")]
    before = source.get_messages("original", include_inactive=True)
    assert [(m["role"], m["content"]) for m in source.get_messages("original", include_compacted=True)] == expected
    if destination == "same-store":
        target = source
    elif destination == "populated":
        target.create_session("existing", source="cli")
        for i in range(5):
            target.append_message("existing", "user", f"existing {i}")
    copied_id = source.copy_session_to("original", target, include_inactive=include_inactive)
    copied = target.get_messages(copied_id, include_inactive=True)
    assert [m["message_uid"] for m in copied] == [
        m["message_uid"] for m in before if include_inactive or m["active"]]
    assert [{k: v for k, v in m.items() if k not in ("id", "session_id")} for m in copied] == [
        {k: v for k, v in m.items() if k not in ("id", "session_id")}
        for m in before if include_inactive or m["active"]]
    visible_orders = target._read_all(
        "SELECT display_order, MIN(id) AS first_id FROM messages WHERE session_id=? "
        "AND (active=1 OR compacted=1) GROUP BY display_order", (copied_id,))
    assert all(row["display_order"] == row["first_id"] for row in visible_orders)
    target.append_message(copied_id, "user", "next question", timestamp=5)
    target.append_message(copied_id, "assistant", "next answer", timestamp=4)
    retained = expected if include_inactive else expected[-2:]
    expected_display = retained + [("user", "next question"), ("assistant", "next answer")]
    display = target.get_messages(copied_id, include_compacted=True)
    assert [(m["role"], m["content"]) for m in display] == expected_display
    _, resumed_display = target.get_resume_conversations(copied_id)
    assert [(m["role"], m["content"]) for m in resumed_display] == expected_display
    assert target.display_message_count(copied_id) == len(expected_display)
    assert [m["content"] for m in target.get_messages(copied_id, include_compacted=True, latest=True, limit=2)] == [
        "next question", "next answer"]
    assert source.get_messages("original", include_inactive=True) == before
    if destination == "populated":
        assert [m["content"] for m in target.get_messages("existing")] == [f"existing {i}" for i in range(5)]


def test_copy_imports_only_referenced_prompt_and_tool_payloads(stores):
    source, target = stores
    source.create_session("original", source="cli", system_prompt="assembled prompt")
    pin = {"version": 1, "tools": [{"type": "function", "function": {"name": "terminal"}}]}
    source.update_session_tool_names("original", pin)
    source.create_session("unrelated", source="cli", system_prompt="unrelated prompt")
    source.update_session_tool_names("unrelated", ["unrelated-tool"])
    source.copy_session_to("original", target)
    assert target.get_session("original")["system_prompt"] == "assembled prompt"
    assert json.loads(target.get_session("original")["tool_names"]) == pin
    assert {r[0] for r in target._conn.execute("SELECT prompt FROM system_prompts")} == {
        "assembled prompt", json.dumps(pin),
    }
    assert source.get_session("unrelated") is not None
    assert target.get_session("unrelated") is None
    assert target._conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_copy_allocates_unique_non_destructive_titles(stores):
    source, target = stores
    source.create_session("original", source="cli")
    source.set_session_title("original", "Title")
    target.create_session("occupied", source="cli")
    target.set_session_title("occupied", "Title (copy)")
    for expected_id, title in (("original", "Title (copy 2)"), ("original-copy", "Title (copy 3)"),
                               ("original-copy-2", "Title (copy 4)")):
        assert source.copy_session_to("original", target) == expected_id
        assert target.get_session(expected_id)["title"] == title
    assert source.get_session("original")["title"] == "Title"
    assert target.get_session("occupied")["title"] == "Title (copy)"


def test_copy_rebinds_owner_without_importing_source_routing(stores):
    source, target = stores
    source.create_session("original", source="telegram", profile_name="default",
                          model_config={"temperature": 0.2, "_delegate_from": "parent"})
    source._write_sql("UPDATE sessions SET transport_profile='default', session_key='agent:default:chat', "
                      "user_id='source-user', chat_id='source-chat', chat_type='dm', thread_id='source-thread', "
                      "display_name='source display', origin_json='{}', handoff_state='pending', "
                      "handoff_platform='telegram', handoff_error='old error' WHERE id='original'")
    source.copy_session_to("original", target)
    row = target.get_session("original")
    assert row["profile_name"] == "coder"
    for key in ("transport_profile", "session_key", "user_id", "chat_id", "chat_type", "thread_id",
                "display_name", "origin_json", "handoff_state", "handoff_platform", "handoff_error"):
        assert row[key] is None, key
    assert json.loads(row["model_config"]) == {"temperature": 0.2}
    assert source.get_session("original")["session_key"] == "agent:default:chat"


@pytest.mark.parametrize("column", ["system_prompt_hash", "tool_names"])
def test_copy_refuses_missing_hashed_payload_without_partial_import(stores, column):
    source, target = stores
    source.create_session("original", source="cli")
    # Deliberately corrupt a synthetic source, never a real store.
    source._conn.execute("PRAGMA foreign_keys=OFF")
    source._write_sql(f"UPDATE sessions SET {column}=? WHERE id='original'", ("a" * 64,))
    with pytest.raises(ValueError, match="Missing source.*payload"):
        source.copy_session_to("original", target)
    assert target.session_count() == 0
    assert target._conn.execute("SELECT * FROM system_prompts").fetchall() == []


def test_copy_recovers_dangling_prompt_from_inline_fallback(stores):
    source, target = stores
    source.create_session("original", source="cli")
    source._conn.execute("PRAGMA foreign_keys=OFF")
    source._write_sql("UPDATE sessions SET system_prompt_hash=?, system_prompt=? WHERE id='original'",
                      ("b" * 64, "intact inline fallback"))
    source.copy_session_to("original", target)
    assert target.get_session("original")["system_prompt"] == "intact inline fallback"
    assert target._conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_copy_reads_one_source_snapshot_during_concurrent_update(stores):
    source, target = stores
    source.create_session("original", source="cli", system_prompt="before update")
    source.append_message("original", role="user", content="before update")
    peer = SessionDB(db_path=source.db_path)
    changed = []
    errors = []
    def concurrent_write(sql):
        if sql.startswith("SELECT * FROM messages") and not changed:
            changed.append(True)
            try:
                peer.update_system_prompt("original", "after update")
                peer.append_message("original", role="assistant", content="after update")
            except Exception as exc:
                errors.append(exc)
    source._conn.set_trace_callback(concurrent_write)
    try:
        source.copy_session_to("original", target)
    finally:
        source._conn.set_trace_callback(None)
        peer.close()
    assert changed and not errors
    assert target.get_session("original")["system_prompt"] == "before update"
    assert [m["content"] for m in target.get_messages("original")] == ["before update"]
    assert source.get_session("original")["system_prompt"] == "after update"


@pytest.mark.parametrize("new_id", ["", " "])
def test_copy_rejects_blank_explicit_id(stores, new_id):
    source, target = stores
    source.create_session("original", source="cli")
    with pytest.raises(ValueError, match="non-empty"):
        source.copy_session_to("original", target, new_session_id=new_id)
    assert target.session_count() == 0


def test_cli_reports_sqlite_copy_failure_without_partial_rows(stores, capsys):
    from argparse import Namespace
    from hermes_cli.sessions_cmd import cmd_sessions
    source, target = stores
    source.create_session("original", source="cli", system_prompt="new prompt")
    source.append_message("original", role="user", content="fail insertion")
    target._write_sql("CREATE TRIGGER reject_copy BEFORE INSERT ON messages "
                      "BEGIN SELECT RAISE(ABORT, 'synthetic copy failure'); END")
    args = Namespace(sessions_action="copy", session_id="original", from_profile="default",
                     to_profile="coder", new_session_id=None)
    assert cmd_sessions(args) == 1
    assert "synthetic copy failure" in capsys.readouterr().out
    assert target.get_session("original") is None
    assert target._conn.execute("SELECT * FROM system_prompts").fetchall() == []


def test_copy_deduplicates_payloads_across_repeated_copies(stores):
    source, target = stores
    source.create_session("original", source="cli", system_prompt="shared payload")
    source.update_session_tool_names("original", ["terminal"])
    target.create_session("existing", source="cli", system_prompt="shared payload")
    target.update_session_tool_names("existing", ["terminal"])
    before = [tuple(r) for r in target._conn.execute("SELECT * FROM system_prompts ORDER BY hash")]
    for _ in range(3):
        source.copy_session_to("original", target)
    assert [tuple(r) for r in target._conn.execute("SELECT * FROM system_prompts ORDER BY hash")] == before
    assert target.session_count() == 4


def test_copy_explicit_collision_is_non_destructive(stores):
    source, target = stores
    source.create_session("original", source="cli", system_prompt="source only payload")
    target.create_session("occupied", source="cli", system_prompt="target only payload")
    target.append_message("occupied", role="user", content="keep target")
    before = target.get_session("occupied")
    with pytest.raises(ValueError, match="already exists"):
        source.copy_session_to("original", target, new_session_id="occupied")
    assert target.get_session("occupied") == before
    assert target.get_messages("occupied")[0]["content"] == "keep target"
    assert [r[0] for r in target._conn.execute("SELECT prompt FROM system_prompts")] == ["target only payload"]
    assert target.session_count() == 1


def test_copy_preserves_inactive_and_raw_tool_fields(stores):
    source, target = stores
    source.create_session("original", source="cli")
    source.append_message("original", role="assistant", tool_calls=[{"id": "call", "type": "function",
                          "function": {"name": "terminal", "arguments": "{}"}}], timestamp=30)
    source.append_message("original", role="tool", content="result", tool_call_id="call",
                          tool_name="terminal", timestamp=10)
    source._write_sql("UPDATE messages SET active=0, reasoning_content='historical reasoning', "
                      "api_content='historical api payload' WHERE role='assistant'")
    before = [dict(row) for row in source._conn.execute("SELECT * FROM messages ORDER BY id")]
    source.copy_session_to("original", target)
    after = [dict(row) for row in target._conn.execute("SELECT * FROM messages ORDER BY id")]
    assert [{k: v for k, v in row.items() if k != "id"} for row in after] == [
        {k: v for k, v in row.items() if k != "id"} for row in before]
    assert target._conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_copy_legacy_inline_tools_and_prompt(stores):
    source, target = stores
    source.create_session("original", source="cli")
    source._write_sql("UPDATE sessions SET system_prompt='legacy inline', tool_names=? WHERE id='original'",
                      (json.dumps(["terminal"]),))
    source.copy_session_to("original", target)
    assert target.get_session("original")["system_prompt"] == "legacy inline"
    assert json.loads(target.get_session("original")["tool_names"]) == ["terminal"]


def test_copy_rolls_back_all_messages_and_new_payloads(stores):
    source, target = stores
    source.create_session("original", source="cli", system_prompt="new prompt")
    source.update_session_tool_names("original", ["new tool"])
    source.append_message("original", role="user", content="first")
    source.append_message("original", role="assistant", content="second")
    target.create_session("existing", source="cli", system_prompt="existing payload")
    before = [tuple(r) for r in target._conn.execute("SELECT * FROM system_prompts")]
    target._write_sql("CREATE TRIGGER reject_second BEFORE INSERT ON messages WHEN NEW.content='second' "
                      "BEGIN SELECT RAISE(ABORT, 'synthetic second message failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="second message failure"):
        source.copy_session_to("original", target)
    assert target.get_session("original") is None
    assert target.message_count() == 0
    assert [tuple(r) for r in target._conn.execute("SELECT * FROM system_prompts")] == before
    assert source.message_count() == 2
    assert target.get_session("existing") is not None


def test_copy_same_store_keeps_original_and_detaches_lineage(stores):
    source, _ = stores
    source.create_session("parent", source="cli")
    source.create_session("original", source="cli", parent_session_id="parent", system_prompt="payload")
    original = source.get_session("original")
    copied_id = source.copy_session_to("original", source)
    assert copied_id == "original-copy"
    assert source.get_session("original") == original
    assert source.get_session(copied_id)["parent_session_id"] is None
    assert source.get_session(copied_id)["profile_name"] == "default"
    assert source._conn.execute("PRAGMA foreign_key_check").fetchall() == []


def _copy_args(session_id, **kwargs):
    from argparse import ArgumentParser
    from hermes_cli.subcommands.sessions import build_sessions_parser
    from hermes_cli.sessions_cmd import cmd_sessions
    parser = ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=cmd_sessions)
    argv = ["sessions", "copy", session_id, "--to-profile", kwargs.pop("to_profile", "coder")]
    for key, value in kwargs.items():
        argv.extend(["--" + key.replace("_", "-"), value])
    return parser.parse_args(argv)


@pytest.mark.parametrize("active_source", [True, False])
def test_cli_copies_named_source_without_creating_active_store(tmp_path, monkeypatch, active_source):
    from hermes_cli.sessions_cmd import cmd_sessions
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    source_home = root / "profiles" / "writer"
    target_home = root / "profiles" / "coder"
    for home in (source_home, target_home):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("model: {}\n")
    monkeypatch.setenv("HERMES_HOME", str(source_home if active_source else root))
    source = SessionDB(db_path=source_home / "state.db")
    source.create_session("named-original", source="cli", system_prompt="named prompt")
    source.close()
    source_bytes = (source_home / "state.db").read_bytes()
    kwargs = {} if active_source else {"from_profile": "writer"}
    assert cmd_sessions(_copy_args("named-orig", **kwargs)) == 0
    assert (source_home / "state.db").read_bytes() == source_bytes
    assert not (root / "state.db").exists()
    target = SessionDB(db_path=target_home / "state.db", read_only=True)
    try:
        assert target.get_session("named-original")["profile_name"] == "coder"
        assert target.get_session("named-original")["system_prompt"] == "named prompt"
    finally:
        target.close()


@pytest.mark.parametrize("failure", ["missing-target", "shell-target", "missing-source", "unknown-session", "ambiguous-prefix"])
def test_cli_refuses_unresolvable_copy_without_creating_target_db(tmp_path, monkeypatch, capsys, failure):
    from hermes_cli.sessions_cmd import cmd_sessions
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    target_home = root / "profiles" / "coder"
    if failure != "missing-target":
        target_home.mkdir(parents=True)
        if failure != "shell-target":
            (target_home / "config.yaml").write_text("model: {}\n")
    if failure != "missing-source":
        source = SessionDB(db_path=root / "state.db")
        source.create_session("original-a", source="cli")
        source.create_session("original-b", source="cli")
        source.close()
    session_id = "original" if failure == "ambiguous-prefix" else "missing" if failure == "unknown-session" else "original-a"
    assert cmd_sessions(_copy_args(session_id)) == 1
    assert "Error:" in capsys.readouterr().out
    assert not (target_home / "state.db").exists()
    if failure == "missing-source":
        assert not (root / "state.db").exists()


def test_cli_copies_only_database_payloads_not_profile_artifacts(stores):
    from hermes_cli.sessions_cmd import cmd_sessions
    source, target = stores
    source_home, target_home = source.db_path.parent, target.db_path.parent
    artifacts = {"memories/MEMORY.md": "source-only memory", "config.yaml": "model: {}\n",
                 "skills/private/SKILL.md": "source skill", ".env": "SYNTHETIC=value\n",
                 "auth.json": "{}", "cron/jobs.json": "{}", "sessions/sessions.json": "{}",
                 "gateway.pid": "synthetic pid"}
    for path, content in artifacts.items():
        p = source_home / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    target_config = (target_home / "config.yaml").read_bytes()
    source.create_session("original", source="cli")
    assert cmd_sessions(_copy_args("original")) == 0
    assert (target_home / "config.yaml").read_bytes() == target_config
    for path, content in artifacts.items():
        assert (source_home / path).read_text() == content
        if path != "config.yaml":
            assert not (target_home / path).exists(), path
