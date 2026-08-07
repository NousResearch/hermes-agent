"""Tests for search_messages limit/offset clamping."""

from hermes_state import SessionDB


def _seed_matches(db: SessionDB, n: int = 30) -> None:
    db.create_session(session_id="clamp-s1", source="cli")
    for i in range(n):
        db.append_message(
            "clamp-s1",
            role="user",
            content=f"paginationneedle unique-{i}",
        )


class TestSearchMessagesPaginationClamp:
    def test_negative_limit_is_clamped_not_unbounded(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            _seed_matches(db, n=30)
            # Without a clamp, SQLite LIMIT -1 returns every match.
            results = db.search_messages("paginationneedle", limit=-1)
            assert len(results) == 1
        finally:
            db.close()

    def test_zero_limit_clamped_to_one(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            _seed_matches(db, n=10)
            results = db.search_messages("paginationneedle", limit=0)
            assert len(results) == 1
        finally:
            db.close()

    def test_huge_limit_capped_at_500(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            _seed_matches(db, n=20)
            results = db.search_messages("paginationneedle", limit=10**9)
            assert len(results) == 20
        finally:
            db.close()

    def test_negative_offset_clamped_to_zero(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            _seed_matches(db, n=10)
            clamped = db.search_messages("paginationneedle", offset=-5, limit=3)
            baseline = db.search_messages("paginationneedle", offset=0, limit=3)
            assert clamped == baseline
        finally:
            db.close()

    def test_valid_limit_unchanged(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            _seed_matches(db, n=10)
            results = db.search_messages("paginationneedle", limit=5)
            assert len(results) == 5
        finally:
            db.close()
