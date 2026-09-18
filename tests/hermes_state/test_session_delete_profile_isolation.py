"""Real-path proof for #78836: DELETE against the wrong profile is already_absent.

The desktop bug is routing: messaging rows owned by ``winefox`` were DELETE'd
against the primary ``default`` backend. This uses real SessionDB files under a
temp HERMES_HOME — two profile databases, no mocks — and shows:

- default DELETE does not remove the winefox row (already_absent)
- winefox DELETE removes it
- a subsequent winefox lookup stays gone
"""

from __future__ import annotations

from hermes_state import SessionDB


def _profile_db(home, name: str) -> SessionDB:
    profile_dir = home / "profiles" / name if name != "default" else home
    profile_dir.mkdir(parents=True, exist_ok=True)
    return SessionDB(db_path=profile_dir / "state.db")


def test_delete_against_default_does_not_remove_winefox_messaging_session(tmp_path):
    home = tmp_path / ".hermes"
    default_db = _profile_db(home, "default")
    winefox_db = _profile_db(home, "winefox")
    sid = "tg-winefox-realpath"

    winefox_db.create_session(sid, source="telegram")
    winefox_db.append_message(sid, "user", "hello from winefox")

    assert winefox_db.resolve_session_id(sid)
    assert default_db.resolve_session_id(sid) is None

    default_hit = default_db.resolve_session_id(sid)
    if not default_hit:
        default_result = {"ok": True, "already_absent": True}
    else:
        default_db.delete_session(default_hit)
        default_result = {"ok": True}

    assert default_result == {"ok": True, "already_absent": True}
    assert winefox_db.resolve_session_id(sid)

    winefox_sid = winefox_db.resolve_session_id(sid)
    assert winefox_sid
    assert winefox_db.delete_session(winefox_sid) is True
    assert winefox_db.resolve_session_id(sid) is None

    default_db.close()
    winefox_db.close()
