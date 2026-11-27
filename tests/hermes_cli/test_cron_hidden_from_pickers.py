"""Cron sessions are excluded from human session pickers, yet retrievable via --source cron."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.cli_session_mixin import CLISessionMixin
from hermes_cli.gateway_session_store import SurfaceLabelReader
from hermes_cli.sessions_cmd import _listing_scope
from persistence.conversation.sessions import INTERNAL_LISTING_SOURCES
from persistence.sqlite.conversation._store.sessions import (
    INTERNAL_LISTING_SOURCES as SQLITE_INTERNAL_LISTING_SOURCES,
)
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session


def test_cron_in_internal_listing_sources():
    assert "cron" in INTERNAL_LISTING_SOURCES
    assert INTERNAL_LISTING_SOURCES == SQLITE_INTERNAL_LISTING_SOURCES


def test_cron_sessions_hidden_from_pickers_and_default_list_but_retrievable_by_source(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, "human-1", "cli")
        seed_session(store, "cron-1", "cron")
        raw_holder(store)._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET message_count = 2 WHERE id IN ('human-1', 'cron-1')"
            )
        )

    with open_persistence_reader(tmp_path / "state.db") as store:
        reader = SurfaceLabelReader(store)

        # In-chat /sessions & /resume picker uses INTERNAL_LISTING_SOURCES
        picker_rows = CLISessionMixin._list_recent_sessions(
            SimpleNamespace(_session_db=reader, session_id="other"),
            limit=20,
        )
        picker_ids = [r["id"] for r in picker_rows]
        assert "human-1" in picker_ids
        assert "cron-1" not in picker_ids

        # Default `hermes sessions list` excludes cron rows
        default_scope = _listing_scope(SimpleNamespace(source=None, all=False))
        default_ids = [r["id"] for r in reader.list_sessions_rich(limit=20, **default_scope)]
        assert "human-1" in default_ids
        assert "cron-1" not in default_ids

        # Explicit `--source cron` retrieves cron rows
        cron_scope = _listing_scope(SimpleNamespace(source="cron", all=False))
        cron_ids = [r["id"] for r in reader.list_sessions_rich(limit=20, **cron_scope)]
        assert cron_ids == ["cron-1"]
