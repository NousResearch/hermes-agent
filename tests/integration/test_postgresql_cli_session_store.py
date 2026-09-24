"""Executable isolated PostgreSQL CLI session lifecycle contract."""
from __future__ import annotations


import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

import agent.skill_commands as skill_commands
import tools.skills_tool as skills_tool
from cli_session_store import PostgreSQLCLISessionCapabilityError, open_cli_session_store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from state_store_runtime_readiness import trap_state_db_opens

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_CONFIG = {"state_store": {"backend": "postgresql", "postgresql": {
    "dsn_env": "HERMES_STATE_STORE_TEST_DSN", "connect_timeout_seconds": 5, "pool_max_size": 2,
}}}

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_cli_home(tmp_path, monkeypatch, postgresql_test_target):
    home = tmp_path / ".hermes-dev-postgresql-state-store"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n    connect_timeout_seconds: 5\n    pool_max_size: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    import state_store
    monkeypatch.setattr(state_store, "postgresql_tenant_schema", lambda *_args, **_kwargs: postgresql_test_target.schema)
    token = set_hermes_home_override(str(home))
    stores = []
    try:
        yield home, stores
    finally:
        for store in stores:
            try:
                store.close()
            except Exception:
                pass
        reset_hermes_home_override(token)


def _open(stores):
    store = open_cli_session_store(_CONFIG)
    stores.append(store)
    return store


def _install_skill_scaffold(tmp_path, monkeypatch):
    """Use the same canonical /skill message builder as SQLite's retitle tests."""
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "work"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: work\ndescription: Test work skill\n---\n\n# work\n\nRepair session title evidence.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    skill_commands.scan_skill_commands()
    return skill_commands.build_skill_invocation_message("/work", user_instruction="repair the title")


def test_fresh_end_resume_prompt_messages_and_search_never_open_sqlite(pg_cli_home, monkeypatch):
    home, stores = pg_cli_home
    session_id = "20260914_010203_pgcli"
    with trap_state_db_opens(home) as opens:
        # Exercise HermesCLI's real early acquisition seam, without a provider or TUI.
        import cli
        shell = cli.HermesCLI.__new__(cli.HermesCLI)
        monkeypatch.setattr(cli, "CLI_CONFIG", _CONFIG)
        shell._init_session_store()
        fresh = shell._session_db
        assert fresh is not None
        stores.append(fresh)
        fresh.create_session(session_id, "cli", model="test-model", model_config={"provider": "local"},
                             system_prompt="stable system prompt", cwd="/tmp", profile_name="pg-cli-test")
        single_id = fresh.append_message(
            session_id, "user", {"text": "remember postgresql resume"},
            platform_message_id="single-identity", timestamp=1_789_000_000,
            display_metadata={"origin": "single"},
        )
        assert single_id > 0
        assert fresh.append_messages_batch(session_id, [
            {"role": "assistant", "content": "persisted answer", "finish_reason": "stop"},
            {"role": "tool", "content": None, "tool_call_id": "null-content"},
        ]) == 2
        records = fresh._store.get_message_records(session_id)
        assert [(row["content"], row["platform_message_id"], row["tool_call_id"])
                for row in records] == [
            ({"text": "remember postgresql resume"}, "single-identity", None),
            ("persisted answer", None, None),
            (None, None, "null-content"),
        ]
        with pytest.raises(PostgreSQLCLISessionCapabilityError, match="single-message controls"):
            fresh.append_message(session_id, "user", "must not write", compression_lock_holder="sqlite-only")
        assert len(fresh._store.get_message_records(session_id)) == 3
        fresh.end_session(session_id, "cli_close")
        assert fresh.get_session(session_id)["ended_at"] is not None
        fresh.close()

        resumed = _open(stores)
        assert resumed.get_session(session_id)["system_prompt"] == "stable system prompt"
        restored, display = resumed.get_resume_conversations(session_id)
        assert [row["content"] for row in restored] == [{"text": "remember postgresql resume"}, "persisted answer", None]
        assert [row["content"] for row in display] == [{"text": "remember postgresql resume"}, "persisted answer", None]
        resumed.reopen_session(session_id)
        assert resumed.get_session(session_id)["ended_at"] is None
        assert resumed.search_sessions(source="cli")[0]["id"] == session_id
        resumed.set_session_title(session_id, "PG resume title")
        assert resumed.resolve_session_by_title("PG resume title") == session_id
        assert resumed.search_sessions(source="cli", workspace_key="/tmp")[0]["id"] == session_id
        before_rotation = {
            "session": dict(resumed.get_session(session_id)),
            "messages": list(resumed._store.get_message_records(session_id)),
        }
        with pytest.raises(PostgreSQLCLISessionCapabilityError, match="no SQLite fallback"):
            resumed.archive_and_compact(session_id)
        assert resumed.get_session(session_id) == before_rotation["session"]
        assert resumed._store.get_message_records(session_id) == before_rotation["messages"]
    assert opens == []
    assert not (home / "state.db").exists()


def test_postgresql_cli_history_listing_and_bare_resume_never_open_sqlite(pg_cli_home, monkeypatch, capsys):
    """The CLI list and bare ``/resume`` enumerate the selected PG profile history."""
    from cli import HermesCLI
    from hermes_cli.sessions_cmd import cmd_sessions

    home, stores = pg_cli_home
    with trap_state_db_opens(home) as opens:
        store = _open(stores)
        store.create_session("pg-cli-history", "cli", cwd="/tmp/pg-history")
        store.set_session_title("pg-cli-history", "PostgreSQL history")
        store.append_message("pg-cli-history", "user", "persisted PostgreSQL preview", timestamp=100)
        store.create_session("pg-oneshot-history", "oneshot", cwd="/tmp/pg-oneshot")
        store.set_session_title("pg-oneshot-history", "PostgreSQL one-shot history")
        store.append_message("pg-oneshot-history", "user", "persisted one-shot preview", timestamp=101)
        store.create_session("pg-gateway-history", "telegram", cwd="/tmp/pg-gateway")
        store.set_session_title("pg-gateway-history", "Gateway history must stay hidden")
        store.append_message("pg-gateway-history", "user", "gateway preview", timestamp=102)
        rows = store.list_sessions_rich(source="cli", limit=10, order_by_last_active=True)
        assert rows[0]["id"] == "pg-cli-history"
        assert {"id", "source", "title", "preview", "last_active", "message_count", "cwd"} <= rows[0].keys()
        assert rows[0]["preview"] == "persisted PostgreSQL preview"
        assert rows[0]["message_count"] == 1

        # The exact listing facade drives both the interactive renderer and the
        # command-line list action; no SessionDB() is permitted for this profile.
        shell = HermesCLI.__new__(HermesCLI)
        shell.session_id = "current-session"
        shell._session_db = store
        shell._pending_resume_sessions = None
        shell.conversation_history = []
        shell.agent = None
        shell._handle_resume_command("/resume")
        pending_ids = [row["id"] for row in shell._pending_resume_sessions]
        assert {"pg-cli-history", "pg-oneshot-history"} <= set(pending_ids)
        assert "pg-gateway-history" not in pending_ids
        assert shell._resolve_resume_target("2")[0] == pending_ids[1]
        resume_output = capsys.readouterr().out
        assert "PostgreSQL history" in resume_output
        assert "PostgreSQL one-shot history" in resume_output
        assert "Gateway history must stay hidden" not in resume_output

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CONFIG)
        result = cmd_sessions(SimpleNamespace(
            sessions_action="list", limit=10, source=None, workspace=None,
        ))
        assert result is None
        listed = capsys.readouterr().out
        assert "PostgreSQL history" in listed
        assert "pg-cli-history" in listed

        assert cmd_sessions(SimpleNamespace(sessions_action="stats")) is None
        stats = capsys.readouterr().out
        assert "Total sessions: 3" in stats
        assert "Total messages: 3" in stats
        assert "cli: 1 sessions" in stats
        assert "Database size:" not in stats
    assert opens == []
    assert not (home / "state.db").exists()


def test_postgresql_sessions_admin_and_browse_match_selected_store_contract(pg_cli_home, monkeypatch, capsys, tmp_path):
    """Admin actions use PG title/pin/status/scaffold contracts without opening SQLite."""
    from hermes_cli.sessions_cmd import cmd_sessions

    home, stores = pg_cli_home
    skill_message = _install_skill_scaffold(tmp_path, monkeypatch)
    with trap_state_db_opens(home) as opens:
        store = _open(stores)
        store.create_session("admin-pg", "cli")
        store.append_message("admin-pg", "user", "unfinished request")
        store.create_session("skill-pg", "cli")
        store.append_message("skill-pg", "user", skill_message)
        store.append_message("skill-pg", "assistant", "handled", finish_reason="stop")
        store.set_session_title("skill-pg", "Skill body title")

        assert store.session_lifecycle_statuses(["admin-pg", "skill-pg", "unknown-pg"]) == {
            "admin-pg": "interrupted", "skill-pg": "complete", "unknown-pg": "empty",
        }
        assert [row["id"] for row in store.list_skill_scaffolded_sessions()] == ["skill-pg"]

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CONFIG)
        assert cmd_sessions(SimpleNamespace(sessions_action="rename", session_id="admin-pg", title=["Admin", "title"])) is None
        assert store.get_session_title("admin-pg") == "Admin title"
        capsys.readouterr()
        assert cmd_sessions(SimpleNamespace(sessions_action="pin", session_ids=["admin-pg"])) is None
        assert store.get_session("admin-pg")["pinned"] is True
        assert {row["id"] for row in store.list_sessions_rich(limit=1, include_pinned=True)} >= {"admin-pg"}
        capsys.readouterr()
        assert cmd_sessions(SimpleNamespace(sessions_action="pinned", json=True)) is None
        assert {row["id"] for row in json.loads(capsys.readouterr().out)} == {"admin-pg"}
        assert cmd_sessions(SimpleNamespace(sessions_action="unpin", session_ids=["admin-pg"])) is None
        assert store.get_session("admin-pg")["pinned"] is False

        monkeypatch.setattr("agent.title_generator.generate_title", lambda _typed: "Repaired skill title")
        assert cmd_sessions(SimpleNamespace(sessions_action="retitle-skills", limit=10, apply=True)) is None
        assert store.get_session_title("skill-pg") == "Repaired skill title"

        observed = {}
        def picker(rows, session_db):
            observed["rows"] = rows
            observed["statuses"] = session_db.session_lifecycle_statuses([row["id"] for row in rows])
            return None
        monkeypatch.setattr("hermes_cli.sessions_cmd._session_browse_picker", picker)
        assert cmd_sessions(SimpleNamespace(sessions_action="browse", source=None, limit=10)) is None
        assert {row["id"] for row in observed["rows"]} >= {"admin-pg", "skill-pg"}
        assert observed["statuses"]["admin-pg"] == "interrupted"
    assert opens == []
    assert not (home / "state.db").exists()


def _export_args(**overrides):
    """Complete parser-shaped export arguments for direct cmd_sessions coverage."""
    values = {
        "sessions_action": "export", "output": None, "format": "jsonl", "session_id": "export-pg",
        "older_than": None, "newer_than": None, "before": None, "after": None, "source": None,
        "title": None, "end_reason": None, "cwd": None, "min_messages": None, "max_messages": None,
        "model": None, "provider": None, "user": None, "chat_id": None, "chat_type": None,
        "branch": None, "min_tokens": None, "max_tokens": None, "min_cost": None, "max_cost": None,
        "min_tool_calls": None, "max_tool_calls": None, "dry_run": False, "redact": False,
        "only": None, "lineage": "single", "delete_after_verified": False, "yes": False,
        "force": False, "upload": False, "public": False, "no_redact": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_postgresql_cli_export_matches_sqlite_oracle_and_stays_local(pg_cli_home, monkeypatch, tmp_path):
    """Selected PG exports canonical segments/lineages without opening state.db."""
    from hermes_cli.sessions_cmd import cmd_sessions
    from hermes_state import SessionDB

    home, stores = pg_cli_home
    oracle_home = tmp_path / ".hermes-oracle"
    oracle_home.mkdir()
    oracle_token = set_hermes_home_override(str(oracle_home))
    try:
        sqlite = SessionDB(db_path=tmp_path / "oracle-state.db")
    finally:
        reset_hermes_home_override(oracle_token)
    with trap_state_db_opens(home) as opens:
        pg = _open(stores)
        for store, prefix in ((pg, "export-pg"), (sqlite, "export-sqlite")):
            parent, child = f"{prefix}-parent", prefix
            store.create_session(parent, "cli", model="oracle/model")
            store.append_message(parent, "user", "parent evidence", timestamp=100)
            store.end_session(parent, "compression")
            store.create_session(child, "cli", model="oracle/model", parent_session_id=parent)
            store.append_message(child, "assistant", "child answer", timestamp=101, finish_reason="stop")

        pg_single, sqlite_single = pg.export_session("export-pg"), sqlite.export_session("export-sqlite")
        assert pg_single is not None and sqlite_single is not None
        for key in ("source", "model", "message_count"):
            assert pg_single[key] == sqlite_single[key]
        fields = ("role", "content", "timestamp", "finish_reason")
        assert [{key: message.get(key) for key in fields} for message in pg_single["messages"]] == [
            {key: message.get(key) for key in fields} for message in sqlite_single["messages"]
        ]
        pg_lineage, sqlite_lineage = pg.export_session_lineage("export-pg"), sqlite.export_session_lineage("export-sqlite")
        assert pg_lineage is not None and sqlite_lineage is not None
        assert [segment["id"].replace("export-pg", "export") for segment in pg_lineage["segments"]] == [
            segment["id"].replace("export-sqlite", "export") for segment in sqlite_lineage["segments"]
        ]
        assert [message["content"] for message in pg_lineage["messages"]] == [message["content"] for message in sqlite_lineage["messages"]]
        assert pg.session_count() == sqlite.session_count()
        assert pg.message_count() == sqlite.message_count()

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CONFIG)
        jsonl = tmp_path / "session.jsonl"
        assert cmd_sessions(_export_args(output=str(jsonl))) is None
        assert json.loads(jsonl.read_text(encoding="utf-8"))["id"] == "export-pg"
        markdown_dir = tmp_path / "markdown"
        assert cmd_sessions(_export_args(output=str(markdown_dir), format="md", lineage="logical")) is None
        assert "parent evidence" in next(markdown_dir.glob("*.md")).read_text(encoding="utf-8")
        qmd_dir = tmp_path / "qmd"
        assert cmd_sessions(_export_args(output=str(qmd_dir), format="qmd")) is None
        assert "child answer" in next(qmd_dir.glob("*.qmd")).read_text(encoding="utf-8")
        html = tmp_path / "session.html"
        assert cmd_sessions(_export_args(output=str(html), format="html")) is None
        assert "child answer" in html.read_text(encoding="utf-8")
        prompt_file = tmp_path / "prompts.jsonl"
        assert cmd_sessions(_export_args(output=str(prompt_file), session_id="export-pg-parent", only="user-prompts")) is None
        assert json.loads(prompt_file.read_text(encoding="utf-8"))["text"] == "parent evidence"
    sqlite.close()
    assert opens == []
    assert not (home / "state.db").exists()


@pytest.mark.parametrize("overrides, expected", [
    ({"session_id": None}, "requires --session-id"),
    ({"source": "cli"}, "bulk, filter, or dry-run"),
    ({"format": "trace"}, "trace export"),
    ({"delete_after_verified": True}, "--delete-after-verified"),
    ({"upload": True}, "--upload"),
])
def test_postgresql_cli_export_rejects_unsupported_controls_before_sqlite(pg_cli_home, monkeypatch, capsys, overrides, expected):
    from hermes_cli.sessions_cmd import cmd_sessions

    home, _stores = pg_cli_home
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CONFIG)
    with trap_state_db_opens(home) as opens:
        assert cmd_sessions(_export_args(**overrides)) == 2
    output = capsys.readouterr().out
    assert expected in output
    assert "no SQLite fallback" in output
    assert opens == []
    assert not (home / "state.db").exists()


def test_postgresql_delete_matches_sqlite_delegate_contract_and_keeps_generation(pg_cli_home):
    """Real PG delete cascades only delegate rows, inside one transaction."""
    home, stores = pg_cli_home
    root, delegate, nested, branch = "delete-root", "delete-delegate", "delete-nested", "delete-branch"
    with trap_state_db_opens(home) as opens:
        store = _open(stores)
        store.create_session(root, "cli", system_prompt="exclusive root prompt")
        store.create_session(delegate, "cli", model_config={"_delegate_from": root}, system_prompt="delegate prompt")
        store.create_session(nested, "cli", parent_session_id=delegate, model_config={"_delegate_from": delegate})
        store.create_session(branch, "cli", parent_session_id=root, system_prompt="shared branch prompt")
        for session_id in (root, delegate, nested, branch):
            store.append_message(session_id, "user", f"message {session_id}")
        root_hash = store.get_session(root)["system_prompt_hash"]
        branch_hash = store.get_session(branch)["system_prompt_hash"]
        assert root_hash and branch_hash and root_hash != branch_hash
        with store._store._connection() as connection, connection.cursor() as cursor:
            schema = store._store._schema
            cursor.execute(
                f"INSERT INTO {schema}.session_model_usage (session_id, model) VALUES (%s, %s)",
                (delegate, "delete-test-model"),
            )
            cursor.execute(
                f"INSERT INTO {schema}.compression_rotation_receipts "
                "(request_id, parent_session_id, child_session_id, holder, fence, committed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                ("delete-receipt", root, delegate, "delete-test", 1, 1.0),
            )
            cursor.execute(
                f"INSERT INTO {schema}.conversation_generations (source, session_key, generation) "
                "VALUES (%s, %s, %s)",
                ("cli", root, 9),
            )
        assert store.get_session_delete_targets(root) == [root, delegate, nested]
        assert not store.delete_session(root, expected_delete_ids=[root, delegate])
        assert store.get_session(root) is not None
        assert store.delete_session(root, sessions_dir=home / "sessions", expected_delete_ids=[root, delegate, nested])
        assert all(store.get_session(session_id) is None for session_id in (root, delegate, nested))
        assert store.get_session(branch)["parent_session_id"] is None
        assert [row["content"] for row in store._store.get_message_records(branch)] == [f"message {branch}"]
        with store._store._connection() as connection, connection.cursor() as cursor:
            schema = store._store._schema
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.session_model_usage WHERE session_id=%s", (delegate,))
            assert cursor.fetchone()[0] == 0
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.compression_rotation_receipts WHERE request_id=%s", ("delete-receipt",))
            assert cursor.fetchone()[0] == 0
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.system_prompts WHERE hash=%s", (root_hash,))
            assert cursor.fetchone()[0] == 0
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.system_prompts WHERE hash=%s", (branch_hash,))
            assert cursor.fetchone()[0] == 1
            cursor.execute(f"SELECT generation FROM {schema}.conversation_generations WHERE source=%s AND session_key=%s", ("cli", root))
            assert cursor.fetchone()[0] == 9
    assert opens == []
    assert not (home / "state.db").exists()


def test_postgresql_delete_observationally_matches_sqlite_oracle(pg_cli_home, tmp_path):
    """The selected PG store has the same single-session delete observations as SQLite."""
    from hermes_state import SessionDB

    _home, stores = pg_cli_home
    pg = _open(stores)
    oracle_home = tmp_path / ".hermes-sqlite-delete-oracle"
    oracle_home.mkdir()
    token = set_hermes_home_override(str(oracle_home))
    try:
        sqlite = SessionDB(db_path=tmp_path / "sqlite-delete-oracle.db")
    finally:
        reset_hermes_home_override(token)
    root, delegate, branch = "oracle-root", "oracle-delegate", "oracle-branch"
    try:
        for store in (sqlite, pg):
            store.create_session(root, "cli")
            store.create_session(delegate, "cli", model_config={"_delegate_from": root})
            store.create_session(branch, "cli", parent_session_id=root)
            for session_id in (root, delegate, branch):
                store.append_message(session_id, "user", f"message {session_id}")
        assert pg.get_session_delete_targets(root) == sqlite.get_session_delete_targets(root)
        targets = sqlite.get_session_delete_targets(root)
        assert sqlite.delete_session(root, expected_delete_ids=targets)
        assert pg.delete_session(root, expected_delete_ids=targets)
        for store in (sqlite, pg):
            assert store.get_session(root) is None
            assert store.get_session(delegate) is None
            branch_session = store.get_session(branch)
            assert branch_session is not None and branch_session["parent_session_id"] is None
            assert [row["content"] for row in store.get_messages(branch)] == [f"message {branch}"]
    finally:
        sqlite.close()


def test_postgresql_delete_rolls_back_on_dependent_cleanup_failure(pg_cli_home):
    """A failed final cleanup restores messages and the session atomically."""
    _home, stores = pg_cli_home
    session_id = "delete-rollback"
    store = _open(stores)
    store.create_session(session_id, "cli", system_prompt="rollback prompt")
    store.append_message(session_id, "user", "must survive rollback")
    with store._store._connection() as connection, connection.cursor() as cursor:
        schema = store._store._schema
        cursor.execute(f"CREATE FUNCTION {schema}.fail_delete_prompt() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'forced cleanup failure'; END; $$")
        cursor.execute(f"CREATE TRIGGER fail_delete_prompt BEFORE DELETE ON {schema}.system_prompts FOR EACH STATEMENT EXECUTE FUNCTION {schema}.fail_delete_prompt()")
    with pytest.raises(Exception, match="forced cleanup failure"):
        store.delete_session(session_id)
    assert store.get_session(session_id) is not None
    assert [row["content"] for row in store._store.get_message_records(session_id)] == ["must survive rollback"]


def test_cli_delete_contract_removes_postgresql_session_without_sqlite(pg_cli_home, monkeypatch, capsys):
    from hermes_cli.sessions_cmd import cmd_sessions

    home, stores = pg_cli_home
    with trap_state_db_opens(home) as opens:
        store = _open(stores)
        store.create_session("delete-pg-cli", "cli")
        store.append_messages_batch("delete-pg-cli", [{"role": "user", "content": "remove me"}])
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CONFIG)
        assert cmd_sessions(SimpleNamespace(sessions_action="delete", session_id="delete-pg-cli", yes=True)) is None
        assert "Deleted session 'delete-pg-cli'." in capsys.readouterr().out
        assert store.get_session("delete-pg-cli") is None
        assert cmd_sessions(SimpleNamespace(sessions_action="delete", session_id="missing-pg-cli", yes=True)) == 1
        assert "No session 'missing-pg-cli'." in capsys.readouterr().out
        store.create_session("cancel-pg-cli", "cli")
        monkeypatch.setattr("hermes_cli.sessions_cmd._confirm_prompt", lambda _prompt: False)
        assert cmd_sessions(SimpleNamespace(sessions_action="delete", session_id="cancel-pg-cli", yes=False)) is None
        assert store.get_session("cancel-pg-cli") is not None
        assert "Cancelled." in capsys.readouterr().out
    assert opens == []
    assert not (home / "state.db").exists()


def test_agent_lazy_recall_acquisition_persists_and_resumes_without_sqlite(pg_cli_home):
    """The real AIAgent lifecycle selects the CLI PG facade, not the SQLite registry."""
    from run_agent import AIAgent

    home, stores = pg_cli_home
    session_id = "20260914_010203_pg_agent"
    tool_defs = [{"type": "function", "function": {
        "name": "local_test", "description": "offline test tool",
        "parameters": {"type": "object", "properties": {}},
    }}]
    with (
        trap_state_db_opens(home) as opens,
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="http://127.0.0.1/offline",
            quiet_mode=True, skip_context_files=True, skip_memory=True, session_id=session_id,
        )
        agent.client = MagicMock()
        with patch("hermes_cli.config.load_config", return_value=_CONFIG):
            store = agent._get_session_db_for_recall()
        assert store.__class__.__name__ == "PostgreSQLCLISessionStore"
        stores.append(store)
        agent._ensure_db_session()
        assert agent._session_db_created
        agent._touch_activity("selected-pg fake-agent preflight", force_persist=True)
        assert store.get_session(session_id)["last_activity_description"] == "selected-pg fake-agent preflight"
        agent._reset_activity_labels_after_turn()
        assert store.get_session(session_id)["last_activity_description"] == ""
        prompt = store.get_session(session_id)["system_prompt"]
        assert prompt == agent._cached_system_prompt

        messages = [
            {"role": "user", "content": "fresh agent turn"},
            {"role": "assistant", "content": "persisted agent answer", "finish_reason": "stop"},
        ]
        assert agent._flush_messages_to_session_db(messages, []) is True
        assert [row["content"] for row in store.get_messages_as_conversation(session_id)] == [
            "fresh agent turn", "persisted agent answer",
        ]
        assert store.try_acquire_session_turn_lease(session_id, "active-lease")
        assert store.append_messages_batch(
            session_id, [{"role": "user", "content": "lease-owned write"}], turn_lease_holder="active-lease") == 1
        store.release_session_turn_lease(session_id, "active-lease")
        assert len(store._store.get_message_records(session_id)) == 3
        store.end_session(session_id, "agent_close")
        store.close()

        resumed = _open(stores)
        assert resumed.get_session(session_id)["system_prompt"] == prompt
        restored, _display = resumed.get_resume_conversations(session_id)
        assert [row["content"] for row in restored] == ["fresh agent turn", "persisted agent answer", "lease-owned write"]
        resumed.reopen_session(session_id)
        assert resumed.get_session(session_id)["ended_at"] is None
    assert opens == []
    assert not (home / "state.db").exists()


def test_inline_session_search_uses_selected_postgresql_contextual_contract_without_sqlite(pg_cli_home):
    """The agent's public inline tool keeps the selected tenant and full response shapes."""
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    from run_agent import AIAgent

    home, stores = pg_cli_home
    session_id, history_id = "20260914_010203_pg_search_live", "20260914_010203_pg_search_history"
    with (
        trap_state_db_opens(home) as opens,
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=_CONFIG),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="http://127.0.0.1/offline",
                       quiet_mode=True, skip_context_files=True, skip_memory=True, session_id=session_id)
        store = cast(Any, agent._get_session_db_for_recall())
        stores.append(store)
        store.create_session(history_id, "cli")
        store.set_session_title(history_id, "Selected PostgreSQL contextual history")
        store.append_message(history_id, "user", "opening contextual evidence", timestamp=10)
        anchor = store.append_message(history_id, "assistant", "中文记忆 selected PostgreSQL anchor", timestamp=20)
        store.append_message(history_id, "tool", "tool boundary", timestamp=30)
        store.append_message(history_id, "assistant", "closing contextual evidence", timestamp=40)
        assert store.search_index_status()["backend"] == "postgresql"
        from state_store import contextual_session_search_store
        from tools.session_search_tool import session_search
        assert contextual_session_search_store(store) is store
        direct = json.loads(session_search(query="中文记忆", db=store))
        assert direct["success"] is True, direct
        assert agent._get_session_db_for_recall() is store

        def invoke(args):
            return json.loads(INLINE_TOOL_EXECUTORS["session_search"](
                agent, args, InlineToolContext(effective_task_id="selected-pg-contextual")))

        discovered = invoke({"query": "中文记忆", "detail": "full", "limit": 3})
        assert discovered["success"] is True, discovered
        assert discovered["mode"] == "discover"
        assert discovered["search_index"]["backend"] == "postgresql"
        assert discovered["search_index"]["available"] is True
        hit = next(result for result in discovered["results"] if result["session_id"] == history_id)
        assert hit["match_message_id"] == anchor
        assert [message["content"] for message in hit["messages"]] == [
            "opening contextual evidence", "中文记忆 selected PostgreSQL anchor", "closing contextual evidence",
        ]
        history_started = int(store.get_session(history_id)["started_at"])
        started_bound = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(history_started))
        next_bound = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(history_started + 1))
        bounded = invoke({"query": "selected PostgreSQL", "after": started_bound, "before": next_bound})
        assert bounded["success"] is True, bounded
        assert [result["session_id"] for result in bounded["results"]] == [history_id]
        before_window = invoke({"query": "中文记忆", "before": started_bound})
        assert before_window["success"] is True, before_window
        assert before_window["results"] == []

        scrolled = invoke({"session_id": history_id, "around_message_id": anchor, "window": 1})
        assert scrolled["success"] is True and scrolled["mode"] == "scroll"
        assert [message["id"] for message in scrolled["messages"]] == [anchor - 1, anchor, anchor + 1]
        read = invoke({"session_id": history_id})
        assert read["success"] is True and read["mode"] == "read" and read["message_count"] == 4
        browsed = invoke({})
        assert browsed["success"] is True and browsed["mode"] == "browse"
        assert history_id in [result["session_id"] for result in browsed["results"]]
        grammar_error = invoke({"query": "中文 AND memory"})
        assert grammar_error["success"] is False
        assert "unsupported or ambiguous" in grammar_error["error"]
        rebuilt = store.rebuild_search_index()
        assert rebuilt["backend"] == "postgresql" and rebuilt["available"] is True
        assert invoke({"query": "中文记忆"})["success"] is True
    assert opens == []
    assert not (home / "state.db").exists()


def test_public_agent_turn_rotates_selected_postgresql_with_real_compressor(pg_cli_home):
    """A real public turn reaches ContextCompressor then the PG child publisher."""
    from run_agent import AIAgent

    home, stores = pg_cli_home
    parent = "20260914_010203_pg_rotation_parent"
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"durable history {index}: " + ("evidence " * 500)}
        for index in range(60)
    ]
    main_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="final deterministic answer", reasoning_content=None, reasoning=None, tool_calls=None,
        ), finish_reason="stop")], model="oracle/model", usage=None,
    )
    summary_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="deterministic provider summary"), finish_reason="stop")]
    )
    with (
        trap_state_db_opens(home) as opens,
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.call_llm", return_value=summary_response) as fake_provider,
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="http://127.0.0.1/offline", model="oracle/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, session_id=parent,
        )
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = main_response
        with patch("hermes_cli.config.load_config", return_value=_CONFIG):
            store = agent._get_session_db_for_recall()
        stores.append(store)
        agent._ensure_db_session()
        store.set_session_title(parent, "PostgreSQL rotation title")
        store.set_session_title_source(parent, "user")
        store.patch_session_model_config(parent, {"provider": "deterministic", "oracle": True})
        agent.compression_in_place = False
        agent.context_compressor.threshold_tokens = 1
        agent.context_compressor.note_usage_less_response()
        agent.max_compression_attempts = 1
        result = agent.run_conversation("live public input", conversation_history=history)

        child = agent.session_id
        assert result["completed"] is True
        assert result["final_response"] == "final deterministic answer"
        assert fake_provider.call_count >= 1
        assert child != parent
        parent_row, child_row = store.get_session(parent), store.get_session(child)
        assert parent_row["end_reason"] == "compression" and parent_row["ended_at"] is not None
        assert child_row["parent_session_id"] == parent
        assert child_row["title"] == "PostgreSQL rotation title"
        assert child_row["title_source"] == "user"
        assert child_row["model"] == "oracle/model"
        assert child_row["system_prompt"] == agent._cached_system_prompt
        assert store.get_compression_tip(parent) == child
        assert store.get_conversation_root(child) == parent
        assert store.get_compression_lineage(child) == [parent, child]
        assert store.get_active_message_watermark(child) > 0
        assert store.get_compression_fallback_streak(child) == 0
        assert store.get_compression_ineffective_count(child) == 0
        assert store.get_compression_recovery_deadline(child) == 0.0
        child_messages = store.get_messages_as_conversation(child)
        assert any(row["content"] == "final deterministic answer" for row in child_messages)
        assert any("deterministic provider summary" in str(row["content"]) for row in child_messages)
    assert opens == []
    assert not (home / "state.db").exists()


def test_public_agent_turn_recovers_lost_publish_ack_from_durable_receipt(pg_cli_home):
    """A public retry-classification path adopts the one committed PG child."""
    from run_agent import AIAgent

    home, stores = pg_cli_home
    parent = "20260914_010203_pg_lost_ack_parent"
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"history {index}: " + ("evidence " * 500)}
        for index in range(60)
    ]
    main_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="lost acknowledgement recovered", reasoning_content=None, reasoning=None, tool_calls=None,
    ), finish_reason="stop")], model="oracle/model", usage=None)
    summary_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="durable receipt summary"), finish_reason="stop")])
    with (
        trap_state_db_opens(home) as opens,
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.call_llm", return_value=summary_response),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="http://127.0.0.1/offline", model="oracle/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, session_id=parent)
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = main_response
        with patch("hermes_cli.config.load_config", return_value=_CONFIG):
            store = agent._get_session_db_for_recall()
        stores.append(store)
        agent._ensure_db_session()
        original_publish = store.publish_compression_child
        request_ids = []
        def commit_then_lose_ack(**kwargs):
            request_ids.append(kwargs["request_id"])
            original_publish(**kwargs)
            raise ConnectionError("simulated lost acknowledgement after commit")
        store.publish_compression_child = commit_then_lose_ack
        agent.compression_in_place = False
        agent.context_compressor.threshold_tokens = 1
        agent.context_compressor.note_usage_less_response()
        agent.max_compression_attempts = 1
        result = agent.run_conversation("lost ack public input", conversation_history=history)
        assert result["completed"] is True and len(request_ids) == 1
        receipt = store.get_compression_publication_receipt(request_ids[0])
        assert receipt is not None
        assert receipt["parent_session_id"] == parent == store.get_session(receipt["child_session_id"])["parent_session_id"]
        assert agent.session_id == receipt["child_session_id"]
        with store._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {store._store._schema}.sessions WHERE parent_session_id=%s", (parent,))
            assert cursor.fetchone()[0] == 1
    assert opens == []
    assert not (home / "state.db").exists()


def test_public_provider_abort_reopens_selected_postgresql_without_receipt_replay(pg_cli_home):
    """A public provider cancellation leaves the PG parent writable and retryable."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from run_agent import AIAgent

    home, stores = pg_cli_home
    parent = "20260914_010203_pg_provider_abort_parent"
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"abort history {index}: " + ("evidence " * 500)}
        for index in range(60)
    ]
    main_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="provider abort retry answer", reasoning_content=None, reasoning=None, tool_calls=None,
    ), finish_reason="stop")], model="oracle/model", usage=None)
    summary_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="provider abort retry summary"), finish_reason="stop")])
    with (
        trap_state_db_opens(home) as opens,
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.call_llm", side_effect=[AuxiliaryExplicitCancellation(), summary_response]) as provider,
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="http://127.0.0.1/offline", model="oracle/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, session_id=parent)
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = main_response
        with patch("hermes_cli.config.load_config", return_value=_CONFIG):
            store = agent._get_session_db_for_recall()
        stores.append(store)
        agent._ensure_db_session()
        agent.compression_in_place = False
        agent.context_compressor.threshold_tokens = 1
        agent.context_compressor.note_usage_less_response()
        agent.max_compression_attempts = 1
        first = agent.run_conversation("abort public input", conversation_history=history)
        assert first["completed"] is True
        assert agent.session_id == parent
        assert store.get_session(parent)["ended_at"] is None
        assert store.get_compression_tip(parent) == parent
        assert store.get_compression_publication_receipt("not-a-real-request") is None
        reopened = _open(stores)
        assert reopened.get_session(parent)["ended_at"] is None
        agent._session_db = reopened
        retry = agent.run_conversation("retry public input", conversation_history=history)
        assert retry["completed"] is True
        assert agent.session_id != parent
        assert store.get_session(parent)["end_reason"] == "compression"
        assert provider.call_count == 2
    assert opens == []
    assert not (home / "state.db").exists()


def test_public_stale_owner_successor_rotates_selected_postgresql(pg_cli_home):
    """A stale public owner cannot strand the parent; the successor publishes once."""
    from run_agent import AIAgent

    home, stores = pg_cli_home
    parent = "20260914_010203_pg_stale_owner_parent"
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"stale history {index}: " + ("evidence " * 500)}
        for index in range(60)
    ]
    main_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="successor answer", reasoning_content=None, reasoning=None, tool_calls=None,
    ), finish_reason="stop")], model="oracle/model", usage=None)
    summary_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="successor server-clock summary"), finish_reason="stop")])
    with (
        trap_state_db_opens(home) as opens,
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.call_llm", return_value=summary_response),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="http://127.0.0.1/offline", model="oracle/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, session_id=parent)
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = main_response
        with patch("hermes_cli.config.load_config", return_value=_CONFIG):
            store = agent._get_session_db_for_recall()
        stores.append(store)
        agent._ensure_db_session()
        assert store.try_acquire_compression_lock(parent, "stale-public-owner", ttl_seconds=0.1)
        time.sleep(0.2)
        agent.compression_in_place = False
        agent.context_compressor.threshold_tokens = 1
        agent.context_compressor.note_usage_less_response()
        agent.max_compression_attempts = 1
        result = agent.run_conversation("successor public input", conversation_history=history)
        assert result["completed"] is True and agent.session_id != parent
        child = agent.session_id
        assert store.get_session(parent)["end_reason"] == "compression"
        assert store.get_compression_tip(parent) == child
        assert store.get_conversation_root(child) == parent
        with store._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT holder, fence FROM {store._store._schema}.compression_rotation_receipts WHERE parent_session_id=%s", (parent,))
            holder, fence = cursor.fetchone()
        assert holder != "stale-public-owner" and fence > 1
    assert opens == []
    assert not (home / "state.db").exists()


def test_postgresql_empty_session_cleanup_accepts_cli_filesystem_compatibility_arg(pg_cli_home):
    """The /new and /clear caller always passes SQLite's sessions_dir argument."""
    home, stores = pg_cli_home
    with trap_state_db_opens(home) as opens:
        store = _open(stores)
        store.create_session("empty-predecessor", "cli")
        store.end_session("empty-predecessor", "new_session")
        assert store.delete_session_if_empty("empty-predecessor", sessions_dir=home / "sessions")
        assert store.get_session("empty-predecessor") is None

        store.create_session("persisted-predecessor", "cli")
        store.append_message("persisted-predecessor", "user", "must survive boundary")
        store.end_session("persisted-predecessor", "new_session")
        assert not store.delete_session_if_empty("persisted-predecessor", sessions_dir=home / "sessions")
        retained = store.get_session("persisted-predecessor")
        assert retained["end_reason"] == "new_session" and retained["ended_at"] is not None
        assert [row["content"] for row in store._store.get_message_records("persisted-predecessor")] == [
            "must survive boundary"
        ]

        with pytest.raises(PostgreSQLCLISessionCapabilityError, match="does not support"):
            store.delete_session_if_empty("persisted-predecessor", unexpected_control=True)
    assert opens == []
    assert not (home / "state.db").exists()


def test_default_sqlite_factory_path_remains_legacy(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    store = open_cli_session_store({})
    try:
        assert store.db_path == home / "state.db"
        assert (home / "state.db").exists()
    finally:
        store.close()


@pytest.mark.parametrize("command", ["/branch PG branch", "/fork PG fork"])
def test_interactive_postgresql_branch_and_fork_are_atomic_and_resume_visible(
    pg_cli_home, monkeypatch, command,
):
    """Both slash spellings use the one-transaction PG publisher, never SQLite."""
    from cli import HermesCLI

    home, stores = pg_cli_home
    parent = f"pg-{command.split()[0][1:]}-parent"
    with trap_state_db_opens(home) as opens:
        store = _open(stores)
        store.create_session(parent, "cli", model="oracle/model")
        store.set_session_title(parent, "PG parent title")
        store.append_messages_batch(parent, [
            {"role": "user", "content": {"text": "canonical input"}, "timestamp": 100,
             "api_content": "wire-user", "display_kind": "chat", "display_metadata": {"ordinal": 1}},
            {"role": "assistant", "content": "canonical output", "timestamp": 101,
             "tool_calls": [{"id": "call-1", "type": "function"}], "reasoning": "because",
             "reasoning_details": [{"type": "reasoning.text", "text": "detail"}],
             "codex_reasoning_items": [{"id": "rs-1", "type": "reasoning"}],
             "codex_message_items": [{"id": "msg-1", "type": "message"}],
             "api_content": "wire-assistant", "display_kind": "assistant",
             "display_metadata": {"ordinal": 2}},
            {"role": "tool", "content": None, "tool_name": "local", "tool_call_id": "call-1",
             "effect_disposition": "success", "timestamp": 102, "api_content": "wire-tool"},
        ])
        expected = store._store.get_message_records(parent)
        shell = HermesCLI.__new__(HermesCLI)
        shell._agent_running = False
        shell._session_db = store
        shell.session_id = parent
        shell.model = "oracle/model"
        shell.max_turns = 17
        shell.reasoning_config = {"effort": "medium"}
        shell.session_start = None
        shell._pending_title = None
        shell._resumed = False
        shell.agent = None
        shell.conversation_history = [{"role": "user", "content": "stale in-memory copy"}]
        shell._transfer_session_yolo = MagicMock()
        monkeypatch.setattr("cli._sync_process_session_id", lambda _session_id: None)

        HermesCLI._handle_branch_command(shell, command)
        child = shell.session_id
        assert child != parent
        parent_row, child_row = store.get_session(parent), store.get_session(child)
        assert parent_row["end_reason"] == "branched" and parent_row["ended_at"] is not None
        assert child_row["parent_session_id"] == parent
        assert child_row["model"] == "oracle/model"
        assert child_row["model_config"] == {
            "max_iterations": 17, "reasoning_config": {"effort": "medium"}, "_branched_from": parent,
        }
        assert child_row["title"] == command.removeprefix("/branch ").removeprefix("/fork ")
        fields = ("role", "content", "tool_name", "tool_calls", "tool_call_id", "effect_disposition", "timestamp",
                  "reasoning", "reasoning_details", "codex_reasoning_items", "codex_message_items", "api_content",
                  "display_kind", "display_metadata", "active", "compacted")
        actual = store._store.get_message_records(child)
        assert [{key: row[key] for key in fields} for row in actual] == [
            {key: row[key] for key in fields} for row in expected
        ]
        assert [row["content"] for row in store.get_resume_conversations(child)[0]] == [
            row["content"] for row in expected
        ]
        assert child in [row["id"] for row in store.search_sessions(source="cli")]
    assert opens == []
    assert not (home / "state.db").exists()


def test_postgresql_branch_normalized_observation_matches_sqlite_oracle(pg_cli_home, tmp_path, monkeypatch):
    """The new atomic path preserves the legacy slash command's visible contract."""
    from cli import HermesCLI
    from hermes_state import SessionDB

    home, stores = pg_cli_home
    pg = _open(stores)
    oracle_home = tmp_path / ".hermes-branch-sqlite-oracle"
    oracle_home.mkdir()
    oracle_token = set_hermes_home_override(str(oracle_home))
    try:
        sqlite = SessionDB(db_path=tmp_path / "branch-sqlite-oracle.db")
    finally:
        reset_hermes_home_override(oracle_token)
    parent_ids = {"pg": "branch-oracle-pg-parent", "sqlite": "branch-oracle-sqlite-parent"}
    history = [
        {"role": "user", "content": "compare input", "timestamp": 100},
        {"role": "assistant", "content": "compare output", "tool_calls": [{"id": "call-1"}],
         "reasoning": "comparison", "reasoning_details": [{"type": "reasoning.text", "text": "detail"}],
         "codex_reasoning_items": [{"id": "rs-1"}], "codex_message_items": [{"id": "msg-1"}], "timestamp": 101},
        {"role": "tool", "content": "compare tool", "tool_name": "local", "tool_call_id": "call-1", "timestamp": 102},
    ]
    with trap_state_db_opens(home) as opens:
        for store, parent in ((pg, parent_ids["pg"]), (sqlite, parent_ids["sqlite"])):
            store.create_session(parent, "cli", model="oracle/model")
            store.set_session_title(parent, "Oracle branch title")
        pg.append_messages_batch(parent_ids["pg"], history)
        shells = {}
        monkeypatch.setattr("cli._sync_process_session_id", lambda _session_id: None)
        for name, store in (("pg", pg), ("sqlite", sqlite)):
            shell = HermesCLI.__new__(HermesCLI)
            shell._agent_running = False
            shell._session_db = store
            shell.session_id = parent_ids[name]
            shell.model = "oracle/model"
            shell.max_turns = 17
            shell.reasoning_config = {"effort": "medium"}
            shell.session_start = None
            shell._pending_title = None
            shell._resumed = False
            shell.agent = None
            shell.conversation_history = history
            shell._transfer_session_yolo = MagicMock()
            HermesCLI._handle_branch_command(shell, "/branch Oracle child")
            shells[name] = shell

        fields = ("role", "content", "tool_name", "tool_calls", "tool_call_id", "reasoning",
                  "reasoning_details", "codex_reasoning_items", "codex_message_items")
        def normalize(value):
            if isinstance(value, str) and value[:1] in "[{":
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return value
        def normalize_rows(rows):
            return [{key: normalize(row.get(key)) for key in fields} for row in rows]
        def observe(store, parent, child):
            session = store.get_session(child)
            model_config = session["model_config"]
            if isinstance(model_config, str):
                model_config = json.loads(model_config)
            model_config["_branched_from"] = "PARENT"
            messages = store.get_messages_as_conversation(child)
            return {
                "parent_end_reason": store.get_session(parent)["end_reason"],
                "child": {"parent_session_id": "PARENT", "model": session["model"],
                          "model_config": model_config, "title": session["title"]},
                "records": normalize_rows(messages), "counter": len(messages),
                "search_visible": child in [row["id"] for row in store.search_sessions(source="cli")],
                "resume": normalize_rows(store.get_resume_conversations(child)[0]),
            }

        observed = {
            name: observe(store, parent_ids[name], shells[name].session_id)
            for name, store in (("pg", pg), ("sqlite", sqlite))
        }
        assert observed["pg"] == observed["sqlite"]
    sqlite.close()
    assert opens == []
    assert not (home / "state.db").exists()


def test_postgresql_branch_rolls_back_parent_and_child_on_message_copy_failure(pg_cli_home):
    """A failed canonical copy cannot strand a closed parent or partial branch."""
    _home, stores = pg_cli_home
    store = _open(stores)
    parent, child = "pg-branch-rollback-parent", "pg-branch-rollback-child"
    store.create_session(parent, "cli", model="oracle/model")
    store.append_message(parent, "user", "must survive")
    with store._store._connection() as connection, connection.cursor() as cursor:
        schema = store._store._schema
        cursor.execute(
            f"CREATE FUNCTION {schema}.fail_branch_copy() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'forced branch copy failure'; END; $$"
        )
        cursor.execute(
            f"CREATE TRIGGER fail_branch_copy BEFORE INSERT ON {schema}.messages "
            f"FOR EACH STATEMENT EXECUTE FUNCTION {schema}.fail_branch_copy()"
        )
    with pytest.raises(Exception, match="forced branch copy failure"):
        store.branch_session(
            parent_session_id=parent, child_session_id=child, source="cli", model="oracle/model",
            model_config={"max_iterations": 1}, title="never published",
        )
    assert store.get_session(child) is None
    parent_row = store.get_session(parent)
    assert parent_row["ended_at"] is None and parent_row["end_reason"] is None
    assert [row["content"] for row in store._store.get_message_records(parent)] == ["must survive"]


def _maintenance_args(action: str, **overrides):
    values = {
        "sessions_action": action, "older_than": None, "newer_than": None, "before": None, "after": None,
        "source": None, "title": None, "end_reason": None, "cwd": None, "min_messages": None,
        "max_messages": None, "model": None, "provider": None, "user": None, "chat_id": None,
        "chat_type": None, "branch": None, "min_tokens": None, "max_tokens": None, "min_cost": None,
        "max_cost": None, "min_tool_calls": None, "max_tool_calls": None, "dry_run": False, "yes": False,
        "include_archived": False, "include_pinned": False, "never_active": False, "no_backup": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_postgresql_maintenance_archive_prune_markers_and_title_repair_match_sqlite_oracle(
    pg_cli_home, monkeypatch, capsys, tmp_path,
):
    """Real isolated PG maintenance preserves the SQLite candidate/pin/refusal contract."""
    from hermes_cli.sessions_cmd import cmd_sessions
    from hermes_state import SessionDB

    home, stores = pg_cli_home
    oracle_home = tmp_path / ".hermes-maintenance-oracle"
    oracle_home.mkdir()
    oracle_token = set_hermes_home_override(str(oracle_home))
    try:
        oracle = SessionDB(db_path=tmp_path / "maintenance-oracle.db")
    finally:
        reset_hermes_home_override(oracle_token)
    try:
        with trap_state_db_opens(home) as opens:
            pg = _open(stores)
            for store, suffix in ((pg, "pg"), (oracle, "sqlite")):
                for session_id, pinned in ((f"old-{suffix}", False), (f"pin-{suffix}", True), (f"open-{suffix}", False)):
                    store.create_session(session_id, "maintenance")
                    store.append_message(session_id, "user", session_id, timestamp=10)
                    if session_id.startswith("open"):
                        continue
                    store.end_session(session_id, "done")
                    store.set_session_pinned(session_id, pinned)
                store.create_session(f"marker-{suffix}", "maintenance")
                store.append_message(f"marker-{suffix}", "assistant", "[memory]", tool_calls=[{"id": "call"}])
                store.end_session(f"marker-{suffix}", "done")
            filters = {"source": "maintenance", "archived": False, "include_pinned": False}
            assert [row["id"].removesuffix("-pg") for row in pg.list_prune_candidates(**filters)] == [
                row["id"].removesuffix("-sqlite") for row in oracle.list_prune_candidates(**filters)
            ]
            assert pg.count_prune_matches(**filters) == oracle.count_prune_matches(**filters) == 2
            assert pg.count_open_prune_matches(**filters) == oracle.count_open_prune_matches(**filters) == 1
            assert pg.purge_stale_tool_call_markers(dry_run=True)["rows_affected"] == 1
            assert pg.get_messages("marker-pg")[0]["content"] == "[memory]"
            assert pg.purge_stale_tool_call_markers()["rows_affected"] == 1
            assert pg.get_messages("marker-pg")[0]["content"] == ""

            monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CONFIG)
            assert cmd_sessions(_maintenance_args("archive", source="maintenance", dry_run=True)) is None
            assert "Dry run" in capsys.readouterr().out
            assert not pg.get_session("old-pg")["archived"]
            monkeypatch.setattr("hermes_cli.sessions_cmd._confirm_prompt", lambda _prompt: False)
            assert cmd_sessions(_maintenance_args("archive", source="maintenance", yes=False)) is None
            assert "Cancelled." in capsys.readouterr().out
            assert not pg.get_session("old-pg")["archived"]
            assert cmd_sessions(_maintenance_args("archive", source="maintenance", yes=True)) is None
            assert pg.get_session("old-pg")["archived"] is True
            assert pg.get_session("pin-pg")["archived"] is False
            assert cmd_sessions(_maintenance_args("prune", source="maintenance", include_archived=True, yes=True)) is None
            assert pg.get_session("old-pg") is None
            assert pg.get_session("pin-pg") is not None and pg.get_session("open-pg") is not None
            assert cmd_sessions(_maintenance_args("prune", source="maintenance", include_archived=True, include_pinned=True, yes=True)) is None
            assert pg.get_session("pin-pg") is None
        assert opens == []
        assert not (home / "state.db").exists()
    finally:
        oracle.close()
