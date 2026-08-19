"""Regression tests: multiplexed peer-fallback must not cross profiles (#74285).

Incident shape (Aug 2026, local install, two Telegram bots on one multiplexed
gateway): a Telegram private chat reports the *user's own id* as ``chat.id``, so
for every bot in the multiplexer the peer tuple
``(source, user_id, chat_id, chat_type, thread_id)`` is byte-identical. The
conservative fallback inside ``find_latest_gateway_session_for_peer`` matched on
that tuple alone and therefore returned a SIBLING PROFILE's row — executing the
wrong persona, credentials and filesystem scope.

Two fixes under test:
  1. ``find_latest_gateway_session_for_peer`` fences the fallback by the
     ``agent:<ns>:`` namespace carried in ``session_key`` (keyless candidates
     must agree on ``profile_name`` instead), failing closed to ``None``.
  2. ``create_session`` inherits ``profile_name`` from a parent only when the
     two rows share that namespace, so a reset lineage that had already
     borrowed a sibling row cannot mislabel a default session as the sibling
     profile's forever after.
"""

import uuid

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


UID = "8693894969"
# Telegram DM: chat_id == user_id, thread_id NULL — identical for every bot.
PEER_TUPLE = dict(source="telegram", user_id=UID, chat_id=UID, chat_type="dm", thread_id=None)
KEY_DEFAULT = f"agent:main:telegram:dm:{UID}"
KEY_MEDICINA = f"agent:medicina:telegram:dm:{UID}"


def _mk(db, sid, *, key=None, profile=None, msgs=1, parent=None):
    kwargs = dict(user_id=UID, chat_id=UID, chat_type="dm")
    if key:
        kwargs["session_key"] = key
    if profile:
        kwargs["profile_name"] = profile
    if parent:
        kwargs["parent_session_id"] = parent
    db.create_session(sid, "telegram", **kwargs)
    for i in range(msgs):
        db.append_message(sid, "user" if i % 2 == 0 else "assistant", f"m{i}")
    return sid


class TestPeerFallbackProfileFence:
    def test_fallback_ignores_sibling_profile_row(self, db):
        """The bug: only the sibling's row exists, keyed for another profile.

        Its peer tuple matches exactly, so the unfenced fallback returned it.
        The fence must return None (mint a fresh session) instead of handing
        the default profile a medicina session.
        """
        _mk(db, "sib_medicina", key=KEY_MEDICINA, profile="medicina")
        found = db.find_latest_gateway_session_for_peer(
            session_key=KEY_DEFAULT, **PEER_TUPLE
        )
        assert found is None, (
            "peer fallback adopted a sibling profile's session: "
            f"{found and found.get('id')}"
        )

    def test_fallback_ignores_sibling_keyless_row_by_profile_name(self, db):
        """Keyless candidate (identity write lost the key) must still be
        fenced — by profile_name, the only signal left."""
        _mk(db, "keyless_medicina", key=None, profile="medicina")
        found = db.find_latest_gateway_session_for_peer(
            session_key=KEY_DEFAULT, **PEER_TUPLE
        )
        assert found is None

    def test_fallback_still_recovers_own_keyless_row(self, db):
        """The fence must not break the case the fallback exists for: a row
        for THIS profile whose session_key write was lost."""
        _mk(db, "mine_keyless", key=None, profile=None)
        found = db.find_latest_gateway_session_for_peer(
            session_key=KEY_DEFAULT, **PEER_TUPLE
        )
        assert found is not None and found["id"] == "mine_keyless"

    def test_fallback_recovers_own_profile_keyless_row(self, db):
        """Same, from the non-default side: medicina recovers its own row."""
        _mk(db, "medicina_keyless", key=None, profile="medicina")
        found = db.find_latest_gateway_session_for_peer(
            session_key=KEY_MEDICINA, **PEER_TUPLE
        )
        assert found is not None and found["id"] == "medicina_keyless"

    def test_exact_key_match_unaffected(self, db):
        """Primary (exact session_key) path keeps working with both rows
        present — each profile resolves to its own session."""
        _mk(db, "row_default", key=KEY_DEFAULT, profile=None)
        _mk(db, "row_medicina", key=KEY_MEDICINA, profile="medicina")
        got_default = db.find_latest_gateway_session_for_peer(
            session_key=KEY_DEFAULT, **PEER_TUPLE
        )
        got_medicina = db.find_latest_gateway_session_for_peer(
            session_key=KEY_MEDICINA, **PEER_TUPLE
        )
        assert got_default["id"] == "row_default"
        assert got_medicina["id"] == "row_medicina"

    def test_newest_sibling_does_not_outrank_own_row(self, db):
        """Ordering is by recency; a NEWER sibling row must not win over this
        profile's older-but-correct row."""
        _mk(db, "mine_old", key=KEY_DEFAULT, profile=None)
        _mk(db, "sibling_new", key=KEY_MEDICINA, profile="medicina")
        with db._lock:
            db._conn.execute(
                "UPDATE sessions SET last_activity_at=? WHERE id=?", (1000.0, "mine_old")
            )
            db._conn.execute(
                "UPDATE sessions SET last_activity_at=?, session_key=NULL WHERE id=?",
                (9999.0, "sibling_new"),
            )
            db._conn.commit()
        found = db.find_latest_gateway_session_for_peer(
            session_key=KEY_DEFAULT, **PEER_TUPLE
        )
        assert found is not None and found["id"] == "mine_old"


class TestProfileInheritanceFence:
    def test_child_does_not_inherit_sibling_profile_across_namespaces(self, db):
        """A reset lineage that crossed profiles must not stamp the child.

        Real row observed locally: child key ``agent:main:...`` (default)
        pointing at a parent keyed ``agent:medicina:...``. The blind COALESCE
        stamped the default child ``profile_name='medicina'``.
        """
        _mk(db, "parent_medicina", key=KEY_MEDICINA, profile="medicina")
        _mk(db, "child_default", key=KEY_DEFAULT, profile=None, parent="parent_medicina")
        row = db.get_session("child_default")
        assert row["profile_name"] is None, (
            f"default child inherited sibling profile: {row['profile_name']!r}"
        )

    def test_child_inherits_within_same_namespace(self, db):
        """Legitimate case must keep working: same-profile rotation."""
        _mk(db, "parent_med2", key=KEY_MEDICINA, profile="medicina")
        _mk(db, "child_med2", key=KEY_MEDICINA, profile=None, parent="parent_med2")
        assert db.get_session("child_med2")["profile_name"] == "medicina"

    def test_keyless_child_still_inherits(self, db):
        """CLI/subagent children have no session_key — the parent's profile is
        the only signal, so inheritance must remain unconditional there."""
        _mk(db, "parent_med3", key=KEY_MEDICINA, profile="medicina")
        sid = "sub_" + uuid.uuid4().hex[:6]
        db.create_session(sid, "subagent", parent_session_id="parent_med3")
        assert db.get_session(sid)["profile_name"] == "medicina"

    def test_explicit_child_profile_never_overwritten(self, db):
        _mk(db, "parent_med4", key=KEY_MEDICINA, profile="medicina")
        _mk(db, "child_expl", key=KEY_MEDICINA, profile="other", parent="parent_med4")
        assert db.get_session("child_expl")["profile_name"] == "other"
