"""Tests for temporary sessions: /temp and --no-session.

Naming note: the user-facing feature is "temporary chat". The internal
plumbing still uses ``ephemeral`` (SessionEntry.ephemeral, the gateway
``ephemeral`` param, check_ephemeral_tool_block) -- those identifiers are
referenced verbatim below and must not be renamed here.

These assert BEHAVIOUR, not configuration literals: what the tool guard
blocks, what the session entry round-trips, what the parser produces. Per
AGENTS.md, snapshotting config values would make these tests restate the
implementation instead of constraining it.

The invariant under test throughout: a temporary session must leave no
durable trace, and must never silently downgrade to a persistent one.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Tool write-guard
# ---------------------------------------------------------------------------

class TestTemporaryToolGuard:
    """Write-side tools are blocked in a temporary chat; read-side still work.

    Blocking reads too (as an earlier upstream attempt did) breaks the agent
    for no privacy gain: reading memory leaves no trace.
    """

    @pytest.fixture
    def guard(self):
        from agent.agent_runtime_helpers import check_ephemeral_tool_block
        return check_ephemeral_tool_block

    @pytest.mark.parametrize("tool,args", [
        ("memory", {"action": "add", "content": "x"}),
        ("memory", {"action": "replace", "old_text": "a", "content": "b"}),
        ("memory", {"action": "remove", "old_text": "a"}),
        ("skill_manage", {"action": "create", "name": "s"}),
        ("skill_manage", {"action": "delete", "name": "s"}),
        ("cronjob", {"action": "create", "schedule": "1h", "prompt": "p"}),
        ("cronjob", {"action": "remove", "job_id": "j"}),
    ])
    def test_write_actions_blocked(self, guard, tool, args):
        assert guard(tool, args) is not None

    @pytest.mark.parametrize("tool,args", [
        ("memory", {"action": "read"}),
        ("skill_view", {"name": "s"}),
        ("skills_list", {}),
        ("cronjob", {"action": "list"}),
        ("terminal", {"command": "ls"}),
        ("read_file", {"path": "x"}),
    ])
    def test_reads_and_unrelated_tools_allowed(self, guard, tool, args):
        assert guard(tool, args) is None

    def test_batch_operations_form_is_blocked(self, guard):
        """memory(operations=[...]) is a write even with no top-level action.

        This shape has no ``action`` key, so an action-only blocklist misses
        it and memory writes leak out of a temporary chat.
        """
        reason = guard("memory", {"operations": [{"action": "add", "content": "x"}]})
        assert reason is not None

    def test_batch_with_only_reads_is_not_blocked(self, guard):
        assert guard("memory", {"operations": [{"action": "read"}]}) is None

    def test_block_reason_is_user_facing(self, guard):
        """The agent surfaces this string, so it must explain itself."""
        reason = guard("memory", {"action": "add", "content": "x"})
        assert reason and len(reason) > 20
        assert "temporary" in reason.lower() or "ephemeral" in reason.lower()


# ---------------------------------------------------------------------------
# Gateway session entry
# ---------------------------------------------------------------------------

class TestSessionEntryTemporaryFlag:
    """The flag must survive persistence, or a restart downgrades a temp chat."""

    def _entry(self, **kw):
        from gateway.session import SessionEntry
        now = datetime.now()
        return SessionEntry(
            session_key="telegram:1:2", session_id="abc123",
            created_at=now, updated_at=now, **kw
        )

    def test_defaults_to_persistent(self):
        assert self._entry().ephemeral is False

    def test_round_trips_through_serialization(self):
        from gateway.session import SessionEntry
        e = self._entry(ephemeral=True)
        assert SessionEntry.from_dict(e.to_dict()).ephemeral is True

    def test_legacy_entry_without_key_loads_as_persistent(self):
        """Existing on-disk sessions predate the field; they must still load."""
        from gateway.session import SessionEntry
        data = self._entry().to_dict()
        data.pop("ephemeral", None)
        assert SessionEntry.from_dict(data).ephemeral is False

    def test_persistent_entry_round_trips_false(self):
        from gateway.session import SessionEntry
        e = self._entry(ephemeral=False)
        assert SessionEntry.from_dict(e.to_dict()).ephemeral is False


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestNoSessionFlag:
    @staticmethod
    @pytest.fixture(scope="class")
    def parser():
        from hermes_cli._parser import build_top_level_parser
        return build_top_level_parser()[0]

    def test_top_level_oneshot(self, parser):
        args = parser.parse_args(["-z", "hello", "--no-session"])
        assert args.no_session is True
        assert args.oneshot == "hello"

    def test_chat_subcommand(self, parser):
        assert parser.parse_args(["chat", "--no-session"]).no_session is True

    def test_defaults_off(self, parser):
        assert parser.parse_args(["-z", "hi"]).no_session is False

    def test_chat_bare_defaults_off(self, parser):
        """SUPPRESS on the subparser must not shadow the top-level default."""
        assert getattr(parser.parse_args(["chat"]), "no_session", False) is False


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

class TestTempCommandRegistration:
    def test_registered_and_cross_surface(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        d = next(c for c in COMMAND_REGISTRY if c.name == "temp")
        # Neither cli_only nor gateway_only: /temp must exist everywhere.
        assert not getattr(d, "cli_only", False)
        assert not getattr(d, "gateway_only", False)

    def test_alias_resolves(self):
        from hermes_cli.commands import resolve_command
        assert resolve_command("temporary").name == "temp"

    def test_busy_policy_dispatches_rather_than_queues(self):
        """Queued as user text, /temp would be replayed INTO the agent.

        The user believes they went private; instead the toggle never happens
        and their message is fed to a persistent session.
        """
        from hermes_cli.commands import COMMAND_REGISTRY
        d = next(c for c in COMMAND_REGISTRY if c.name == "temp")
        assert d.busy_policy == "interrupt_then_dispatch"
        assert d.busy_handler == "temp"

    def test_busy_handler_is_wired(self):
        """A busy_handler key with no handler silently falls back to reject."""
        from gateway.run import GatewayRunner
        assert hasattr(GatewayRunner, "_busy_temp_command")

    def test_handlers_exist_on_both_surfaces(self):
        from hermes_cli.cli_commands_mixin import CLICommandsMixin
        from gateway.slash_commands import GatewaySlashCommandsMixin
        assert hasattr(CLICommandsMixin, "_handle_temp_command")
        assert hasattr(GatewaySlashCommandsMixin, "_handle_temp_command")


# ---------------------------------------------------------------------------
# No row on disk
# ---------------------------------------------------------------------------

class TestNoSessionRowIsPersisted:
    """A temporary chat must not create a `sessions` row AT ALL.

    Asserting only `title is None` and `message_count == 0` (as an earlier
    version of this file did) passes while an empty row still sits in the
    database. That row is not harmless: it records *when* a private chat was
    opened, plus model, billing provider, cwd and profile. The promise is
    "nothing on disk", so the assertion has to be a row count of zero.

    The leak these cover is indirect: token/cost accounting calls
    _insert_session_row(session_id, "unknown") purely to satisfy a foreign
    key, so a chat the user never typed in still materialises a row as soon
    as the first API call is billed.
    """

    @pytest.fixture()
    def db(self, tmp_path):
        from hermes_state import SessionDB, unmark_session_ephemeral
        d = SessionDB(db_path=tmp_path / "state.db")
        yield d
        unmark_session_ephemeral("temp-sid")

    def _count(self, db, sid):
        with db._read_ctx() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?", (sid,)
            ).fetchone()[0]

    def test_create_session_writes_no_row(self, db):
        from hermes_state import mark_session_ephemeral
        mark_session_ephemeral("temp-sid")
        db.create_session("temp-sid", "desktop")
        assert self._count(db, "temp-sid") == 0

    def test_token_accounting_writes_no_row(self, db):
        """The actual regression: billing resurrected the row."""
        from hermes_state import mark_session_ephemeral
        mark_session_ephemeral("temp-sid")
        db.update_token_counts(
            "temp-sid", model="claude-opus-5", input_tokens=6, output_tokens=1685
        )
        assert self._count(db, "temp-sid") == 0

    def test_normal_session_still_persists(self, db):
        """The guard must not suppress ordinary sessions."""
        db.create_session("normal-sid", "desktop")
        assert self._count(db, "normal-sid") == 1

    def test_session_create_response_echoes_the_ephemeral_flag(self):
        """The create response must carry `ephemeral`.

        The desktop builds its optimistic sidebar row from this response alone,
        before any session.info arrives. When the flag was missing the row
        defaulted to "not temporary" and a temporary chat sat in Recents for
        its whole life -- with the gateway registry AND the sidebar filter both
        already correct. Nothing downstream can recover it, so assert on the
        payload itself.
        """
        import re
        from pathlib import Path

        src = Path("tui_gateway/methods_session.py").read_text(encoding="utf-8")
        create = src[src.index('@method("session.create")'):]
        create = create[: create.index("\n@method(")]
        info = create[create.index('"info": {'):]
        assert re.search(r'"ephemeral":\s*ephemeral', info), (
            "session.create no longer reports `ephemeral` in its info payload; "
            "the desktop sidebar row will silently default to non-temporary"
        )

    def test_cli_prompt_uses_incognito_glyph_not_a_padlock(self):
        """The CLI prompt must not promise encryption.

        A padlock reads as "encrypted/secure", which is a different and
        misleading claim: a temporary chat is not safer in transit, it is
        simply not written down. The desktop badge uses the spy glyph for
        exactly this reason and the two surfaces must not disagree.
        """
        import inspect
        import re

        import cli

        src = inspect.getsource(cli.HermesCLI._get_tui_prompt_fragments)
        # Strip comments -- the rationale comment mentions the padlock.
        code = "\n".join(
            re.sub(r"#.*$", "", line) for line in src.splitlines()
        )
        assert "\U0001F512" not in code, "CLI temp prompt uses a padlock glyph"
        assert "\U0001F575" in code, "CLI temp prompt lost the incognito glyph"

    def test_no_trace_in_any_table_including_lazy_paths(self, db, tmp_path):
        """End-to-end: nothing lands in ANY table, via the paths that lack
        their own guard.

        run_agent._get_session_db_for_recall and _ensure_db_session both guard
        `_persist_disabled` but NOT `ephemeral`, so they will happily open the
        store and call create_session() for a temporary chat. That is only safe
        because _insert_session_row is the chokepoint underneath them. This
        test exercises that exact call sequence and then sweeps every table
        with a session-id column, so a future refactor that adds a second
        INSERT path fails here rather than in production.
        """
        from hermes_state import mark_session_ephemeral, unmark_session_ephemeral

        sid = "20990101_000000_probe1"
        mark_session_ephemeral(sid)
        try:
            db.create_session(session_id=sid, source="cli", cwd=str(tmp_path))
            db.update_token_counts(sid, input_tokens=100, output_tokens=50, model="claude-x")

            with db._read_ctx() as conn:
                tables = [
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                ]
                leaks = []
                for t in tables:
                    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t})")]
                    for c in cols:
                        if "session" in c.lower() and "id" in c.lower():
                            n = conn.execute(
                                f"SELECT COUNT(*) FROM {t} WHERE {c}=?", (sid,)
                            ).fetchone()[0]
                            if n:
                                leaks.append(f"{t}.{c}={n}")

            assert not leaks, f"temporary session left rows behind: {leaks}"
        finally:
            unmark_session_ephemeral(sid)

    def test_registry_is_released_on_teardown(self):
        """The id set must not grow forever, and must not poison id reuse.

        A leaked registration is worse than a memory leak: a later session
        that reused the id would be silently treated as temporary and never
        persisted -- a data-loss bug wearing a privacy feature's clothes.
        """
        import inspect

        from tui_gateway import server

        src = inspect.getsource(server._pop_session_by_id)
        assert "unmark_session_ephemeral" in src, (
            "session teardown must release the temporary-session registration"
        )

    def test_cli_reads_the_underscore_attribute(self):
        """cli.py stores the flag as `_ephemeral`, not `ephemeral`.

        `getattr(self, "ephemeral", False)` silently returns False, so the
        auto-title guard never fires and a temporary CLI chat gets titled --
        the precise leak the flag exists to stop. A default-False getattr on a
        misspelled attribute fails open and cannot be caught by type checking.
        """
        import re

        src = (Path(__file__).resolve().parents[2] / "cli.py").read_text(encoding="utf-8")
        bad = re.findall(r'getattr\(\s*self\s*,\s*"ephemeral"', src)
        assert not bad, (
            f"cli.py reads self.ephemeral ({len(bad)}x) but the attribute is _ephemeral"
        )

    def test_unmark_restores_persistence(self, db):
        from hermes_state import mark_session_ephemeral, unmark_session_ephemeral
        mark_session_ephemeral("temp-sid")
        unmark_session_ephemeral("temp-sid")
        db.create_session("temp-sid", "desktop")
        assert self._count(db, "temp-sid") == 1
