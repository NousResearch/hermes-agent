from contextlib import closing

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("method,column,value", [
    ("set_session_archived", "archived", True),
    ("set_session_hidden", "hidden", True),
    ("set_session_pinned", "pinned", True),
    ("set_session_read", "last_read_at", False),
])
@pytest.mark.parametrize("child_kind", ["branch", "delegate", "tool"])
@pytest.mark.parametrize("target", ["root", "separate", "separate-tip"])
def test_lineage_flags_do_not_cross_independent_child_edges(tmp_path, method, column, value, child_kind, target):
    marker = {"branch": {"_branched_from": "root"}, "delegate": {"_delegate_from": "root"}, "tool": {}}[child_kind]
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        assert db.import_sessions([
            {"id": "root", "source": "cli", "end_reason": "compression"},
            {"id": "tip", "source": "cli", "parent_session_id": "root"},
            {"id": "separate", "source": "tool" if child_kind == "tool" else "cli",
             "parent_session_id": "root", "model_config": marker, "end_reason": "compression"},
            {"id": "separate-tip", "source": "cli", "parent_session_id": "separate", "model_config": marker},
        ])["ok"]
        before = {sid: db.get_session(sid)[column] for sid in ("root", "tip", "separate", "separate-tip")}
        getattr(db, method)(target, value)
        affected = {"root", "tip"} if target == "root" else {"separate", "separate-tip"}
        for sid, previous in before.items():
            actual = db.get_session(sid)[column]
            assert actual == (int(value) if sid in affected else previous), sid
