"""Caller-appended wildcards must not erase a star inside quoted CJK search text."""

import pytest

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.mark.parametrize("suffix", ["", "*"])
def test_quoted_cjk_search_keeps_literal_star(tmp_path, suffix):
    database = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(database, "quoted-star", "cli")
        row_ids = seed_messages(
            database, "quoted-star",
            {"role": "user", "content": "项目甲* 备份档"},
            {"role": "assistant", "content": "项目甲 备份档"},
        )

        hits = database.search.search_messages(f'"项目甲* 备份档"{suffix}', fields=("id",))

        assert [hit.message_id for hit in hits] == [row_ids[0]]
        # Ordinary unquoted prefix searches still find both variants.
        assert {hit.message_id for hit in database.search.search_messages("项目甲*", fields=("id",))} == set(row_ids)
    finally:
        database.close()
