"""Root and ordinary directory filters share exact directory-boundary semantics."""

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("prefix, descendants, outside", [
    ("/", ["/", "/repo", "/repo/src"], ["relative", "C:\\repo"]),
    ("\\", ["\\", "\\repo", "\\repo\\src"], ["relative", "/repo"]),
    ("C:\\", ["C:\\", "C:\\repo", "C:\\repo\\src"], ["D:\\repo", "relative"]),
    ("/work/a_%/", ["/work/a_%", "/work/a_%/src"], ["/work/aXX/src", "/work/a_%other"]),
])
@pytest.mark.parametrize("order_by_last_active", [False, True])
def test_directory_filter_agrees_across_list_count_and_resume(
    tmp_path, prefix, descendants, outside, order_by_last_active,
):
    # These are stored path strings, not a simulated host OS or real directories.
    db = SessionDB(tmp_path / "state.db")
    try:
        for index, cwd in enumerate(descendants + outside):
            db.create_session(f"s{index}", source="cli", cwd=cwd)
        expected = {f"s{i}" for i in range(len(descendants))}
        rows = db.list_sessions_rich(cwd_prefix=prefix, order_by_last_active=order_by_last_active)
        assert {row["id"] for row in rows} == expected
        assert db.session_count(cwd_prefix=prefix) == len(expected)
        assert {row["id"] for row in db.search_sessions(workspace_key=prefix)} == expected
    finally:
        db.close()


def test_root_prune_matches_its_preview_and_preserves_outside_rows(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        for sid, cwd in [("root", "/"), ("child", "/repo/src"), ("outside", "relative")]:
            db.create_session(sid, source="cli", cwd=cwd)
            db.append_message(sid, role="user", content=sid)
            db.end_session(sid, end_reason="completed")
        candidates = db.list_prune_candidates(cwd_prefix="/")
        assert {row["id"] for row in candidates} == {"root", "child"}
        assert db.count_prune_matches(cwd_prefix="/") == len(candidates)
        assert db.prune_sessions(older_than_days=None, cwd_prefix="/") == len(candidates)
        assert {row["id"] for row in db.search_sessions()} == {"outside"}
        assert db.get_messages("outside")[0]["content"] == "outside"
    finally:
        db.close()
