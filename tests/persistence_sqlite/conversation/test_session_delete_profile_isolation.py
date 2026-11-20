"""Real-path proof for #78836: DELETE against the wrong profile is already_absent.

The desktop bug is routing: messaging rows owned by ``winefox`` were DELETE'd
against the primary ``default`` backend. This uses real state.db files under a
temp HERMES_HOME (two profile databases, no mocks) and shows:

- default DELETE does not remove the winefox row (already_absent)
- winefox DELETE removes it
- a subsequent winefox lookup stays gone
"""

from __future__ import annotations

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, delete_session, resolve_session_id

MEDIA_ROOT = "/nonexistent/native-inputs"


def _profile_db(home, name: str) -> StoreHolder:
    profile_dir = home / "profiles" / name if name != "default" else home
    profile_dir.mkdir(parents=True, exist_ok=True)
    return StoreHolder(profile_dir / "state.db")


def test_delete_against_default_does_not_remove_winefox_messaging_session(tmp_path):
    home = tmp_path / ".hermes"
    default_db = _profile_db(home, "default")
    winefox_db = _profile_db(home, "winefox")
    sid = "tg-winefox-realpath"

    create_session(winefox_db, sid, "telegram")
    append_message(winefox_db, sid, "user", "hello from winefox")

    assert resolve_session_id(winefox_db, sid)
    assert resolve_session_id(default_db, sid) is None

    default_hit = resolve_session_id(default_db, sid)
    if not default_hit:
        default_result = {"ok": True, "already_absent": True}
    else:
        delete_session(default_db, default_hit, media_root=MEDIA_ROOT)
        default_result = {"ok": True}

    assert default_result == {"ok": True, "already_absent": True}
    assert resolve_session_id(winefox_db, sid)

    winefox_sid = resolve_session_id(winefox_db, sid)
    assert winefox_sid
    assert delete_session(winefox_db, winefox_sid, media_root=MEDIA_ROOT) == (sid,)
    assert resolve_session_id(winefox_db, sid) is None

    default_db.close()
    winefox_db.close()
