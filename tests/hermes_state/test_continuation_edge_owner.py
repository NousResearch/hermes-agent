"""Cross-surface contract for compression continuation edge ownership.

All tests use the real SessionDB and a real temporary SQLite database.  The
same parent-bound edge definition must drive routing, resume, listing and
lineage-wide flags.
"""

from contextlib import closing
import json

import pytest

from hermes_cli.web_server_sessions import _session_latest_descendant
from hermes_state import SessionDB


def _foreign_marker_lineage(db: SessionDB, marker: str) -> tuple[str, str, str]:
    """origin -> explicit fork root -(compression)-> continuation.

    The continuation inherits the root's model_config, so *marker* still names
    origin.  It is therefore a real continuation of root, not a new fork of
    root.
    """
    suffix = marker.removeprefix("_").replace("_from", "")
    origin = f"{suffix}-origin"
    root = f"{suffix}-root"
    tip = f"{suffix}-tip"

    db.create_session(origin, source="cli")
    db.create_session(
        root,
        source="cli",
        parent_session_id=origin,
        model_config={marker: origin},
    )
    db.append_message(root, "user", "before compression")
    db.end_session(root, "compression")
    db.create_session(
        tip,
        source="cli",
        parent_session_id=root,
        model_config={marker: origin},
    )
    db.append_message(tip, "assistant", "after compression")
    return origin, root, tip


@pytest.mark.parametrize(
    "marker",
    ["_branched_from", "_delegate_from", "_reset_from"],
)
def test_foreign_fork_marker_does_not_sever_compression_continuation(tmp_path, marker):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        origin, root, tip = _foreign_marker_lineage(db, marker)

        assert db._is_compression_child_row(db.get_session(tip)) is True
        assert db.get_compression_tip(root) == tip
        assert db.resolve_resume_session_id(root) == tip

        routing_key = f"route:{marker}"
        db.record_gateway_session_peer(
            tip,
            source="cli",
            session_key=routing_key,
            include_compression_ancestors=True,
        )
        assert db.get_session(root)["session_key"] == routing_key
        assert db.get_session(tip)["session_key"] == routing_key
        assert db.get_session(origin)["session_key"] != routing_key

        leaf, path = _session_latest_descendant(root, db)
        assert leaf == tip
        assert path == [root, tip]


@pytest.mark.parametrize("marker", ["_branched_from", "_reset_from"])
def test_user_visible_foreign_marker_lineage_lists_once_at_its_tip(tmp_path, marker):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        _origin, root, tip = _foreign_marker_lineage(db, marker)

        bounded = db.list_recent_sessions_bounded(limit=20)
        bounded_ids = [row["id"] for row in bounded]
        assert root not in bounded_ids
        assert bounded_ids.count(tip) == 1

        rich = db.list_sessions_rich(
            limit=20,
            order_by_last_active=True,
        )
        rich_ids = [row["id"] for row in rich]
        assert root not in rich_ids
        assert rich_ids.count(tip) == 1


def _compression_parent_with_independent_children(db: SessionDB) -> tuple[str, str, list[str]]:
    root = "flags-root"
    tip = "flags-tip"
    db.create_session(root, source="cli")
    db.append_message(root, "user", "root")
    db.end_session(root, "compression")
    db.create_session(tip, source="cli", parent_session_id=root)

    children = []
    for name, source, model_config in (
        ("branch", "cli", {"_branched_from": root}),
        ("delegate", "cli", {"_delegate_from": root}),
        ("reset", "cli", {"_reset_from": root}),
        ("tool", "tool", None),
    ):
        sid = f"flags-{name}"
        db.create_session(
            sid,
            source=source,
            parent_session_id=root,
            model_config=model_config,
        )
        children.append(sid)
    return root, tip, children


def test_lineage_flag_from_root_stops_at_independent_children(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        root, tip, independent = _compression_parent_with_independent_children(db)

        assert db.set_session_read(root, read=False) is True

        assert db.get_session(root)["last_read_at"] == 0
        assert db.get_session(tip)["last_read_at"] == 0
        for sid in independent:
            assert db.get_session(sid)["last_read_at"] is None


def test_lineage_flag_from_fork_does_not_walk_back_into_parent_conversation(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        root, tip, independent = _compression_parent_with_independent_children(db)
        branch = "flags-branch"

        assert db.set_session_pinned(branch, True) is True

        assert db.get_session(branch)["pinned"] == 1
        assert db.get_session(root)["pinned"] == 0
        assert db.get_session(tip)["pinned"] == 0
        for sid in independent:
            if sid != branch:
                assert db.get_session(sid)["pinned"] == 0



def test_legacy_reset_repair_ignores_foreign_fork_marker(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        db.create_session("older-parent", source="cli")
        db.create_session(
            "reset-parent",
            source="cli",
            session_key="cli:peer",
        )
        db.end_session("reset-parent", "session_reset")
        db.create_session(
            "legacy-reset-child",
            source="cli",
            session_key="cli:peer",
            parent_session_id="reset-parent",
            model_config={"_branched_from": "older-parent"},
        )

        db.reopen_session("reset-parent")

        child = db.get_session("legacy-reset-child")
        model_config = json.loads(child["model_config"])
        assert model_config["_branched_from"] == "older-parent"
        assert model_config["_reset_from"] == "reset-parent"
