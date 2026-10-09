"""A delegate subagent transcript can never own a gateway routing key (#92859).

Reported failure: a ``delegate_task`` batch dispatched from a Discord DM ended
with the DM's routing key bound to one of the *spawned children*. That bind
went through ``SessionStore.switch_session()``, which promotes the outgoing
session to the un-resurrectable ``session_switch`` boundary — so the human
parent was killed mid-batch, every in-flight delegation from that parent was
then classified ``terminal`` and dropped, and the user's next DM was routed
into a leaf subagent with no knowledge of the conversation.

Two invariants are pinned here, both against real ``SessionDB`` rows:

1. ``switch_session()`` — the single sink every route rebind funnels through
   (async-completion pinning, /resume, CLI handoff, Telegram topic rebinding)
   — refuses a delegate/subagent target and leaves the incumbent route and
   its session row untouched.
2. ``find_latest_gateway_session_for_peer()`` — the restart-recovery path —
   never returns a subagent row, so a route that was hijacked before this fix
   (or by any other means) is not re-adopted after a gateway restart.
"""

import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, is_internal_subagent_row


@pytest.fixture
def store_and_db(tmp_path):
    from unittest.mock import patch

    from hermes_state import SessionDB

    config = GatewayConfig()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    store._loaded = True
    try:
        yield store, db
    finally:
        db.close()


def _dm_source():
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="1540910309580214372",
        chat_type="dm",
        user_id="user-1",
        user_name="karan",
    )


class TestSwitchSessionRefusesSubagentTargets:
    def test_delegate_child_cannot_take_over_its_parents_route(self, store_and_db):
        """The exact reported sequence: parent DM, live child, rebind attempt."""
        store, db = store_and_db
        parent_entry = store.get_or_create_session(_dm_source())
        parent_id = parent_entry.session_id

        # A subagent child exactly as delegate_tool creates it: platform
        # source 'subagent', durable _delegate_from marker, no routing key.
        db.create_session(
            "child_leaf",
            source="subagent",
            parent_session_id=parent_id,
            model_config={"_delegate_from": parent_id},
        )

        switched = store.switch_session(parent_entry.session_key, "child_leaf")

        assert switched is None, "a subagent row must never become a route owner"
        # The route still points at the human conversation...
        assert (
            store.get_or_create_session(_dm_source()).session_id == parent_id
        )
        # ...and, decisively, the parent was NOT ended at the hard
        # session_switch boundary, which is what made the batch's own
        # completion undeliverable in the incident.
        parent_row = db.get_session(parent_id)
        assert parent_row["ended_at"] is None
        assert parent_row["end_reason"] is None
        # The child never acquired the routing key.
        assert not db.get_session("child_leaf")["session_key"]

    def test_refusal_holds_for_nested_and_already_hijacked_rows(self, store_and_db):
        """Provenance, not the parent edge, decides — two harder shapes.

        ``parent_session_id == current route`` is not the invariant: a
        grandchild's parent edge points at another child, and a row hijacked
        before this fix has had ``source`` overwritten with the platform name
        by ``record_gateway_session_peer``. Both must still be refused.
        """
        store, db = store_and_db
        parent_entry = store.get_or_create_session(_dm_source())
        parent_id = parent_entry.session_id

        db.create_session(
            "child_leaf", source="subagent", parent_session_id=parent_id,
            model_config={"_delegate_from": parent_id},
        )
        # Grandchild: parent edge is the CHILD, not the routed session.
        db.create_session(
            "grandchild", source="subagent", parent_session_id="child_leaf",
            model_config={"_delegate_from": "child_leaf"},
        )
        # A row that was already hijacked once: source is now the platform,
        # only the _delegate_from marker still betrays what it is (this is
        # the literal shape of session 20260823_041917_519cdb on the
        # reporter's machine).
        db.create_session(
            "rehijack", source="discord", parent_session_id=parent_id,
            model_config={"_delegate_from": parent_id},
        )

        for target in ("grandchild", "rehijack"):
            assert store.switch_session(parent_entry.session_key, target) is None
            assert db.get_session(parent_id)["ended_at"] is None

    def test_real_gateway_session_still_switches(self, store_and_db):
        """The guard must not break /resume — the feature switch_session exists for."""
        store, db = store_and_db
        parent_entry = store.get_or_create_session(_dm_source())
        parent_id = parent_entry.session_id

        db.create_session("older_convo", source="discord", user_id="user-1")
        db.end_session("older_convo", end_reason="user_exit")

        switched = store.switch_session(parent_entry.session_key, "older_convo")

        assert switched is not None
        assert switched.session_id == "older_convo"
        assert db.get_session("older_convo")["ended_at"] is None
        assert db.get_session(parent_id)["end_reason"] == "session_switch"


class TestPeerRecoveryExcludesSubagentRows:
    def test_hijacked_child_is_not_re_adopted_after_restart(self, store_and_db):
        """Restart recovery must prefer the real conversation, not the child.

        The incident left a subagent row carrying the DM's routing key and
        still open, ranked *above* the real parent by recency. Recovery would
        hand the chat straight back to the subagent on the next restart.
        """
        store, db = store_and_db
        key = "agent:main:discord:dm:1540910309580214372"

        db.create_session(
            "real_convo", source="discord", session_key=key,
            user_id="user-1", chat_id="1540910309580214372", chat_type="dm",
        )
        db.append_message(session_id="real_convo", role="user", content="hi")
        # Hijacked child: same key, newer, holds messages too.
        db.create_session(
            "hijacked_child", source="discord", session_key=key,
            user_id="user-1", chat_id="1540910309580214372", chat_type="dm",
            parent_session_id="real_convo",
            model_config={"_delegate_from": "real_convo"},
        )
        db.append_message(session_id="hijacked_child", role="user", content="task")

        recovered = db.find_latest_gateway_session_for_peer(
            source="discord", session_key=key, user_id="user-1",
            chat_id="1540910309580214372", chat_type="dm",
        )

        assert recovered is not None
        assert recovered["id"] == "real_convo"

    def test_peer_tuple_fallback_also_excludes_subagent_rows(self, store_and_db):
        """The keyless fallback query needs the same fence, not just the keyed one."""
        store, db = store_and_db
        db.create_session(
            "subagent_only", source="discord", user_id="user-1",
            chat_id="1540910309580214372", chat_type="dm",
            model_config={"_delegate_from": "some_parent"},
        )
        db.append_message(session_id="subagent_only", role="user", content="task")

        recovered = db.find_latest_gateway_session_for_peer(
            source="discord", session_key="agent:main:discord:dm:no-such-key",
            user_id="user-1", chat_id="1540910309580214372", chat_type="dm",
        )

        assert recovered is None


class TestIsInternalSubagentRow:
    @pytest.mark.parametrize(
        "row",
        [
            {"source": "subagent"},
            {"source": "discord", "model_config": {"_delegate_from": "p"}},
            {"source": "discord", "model_config": json.dumps({"_delegate_from": "p"})},
        ],
    )
    def test_detects_subagent_rows(self, row):
        assert is_internal_subagent_row(row) is True

    @pytest.mark.parametrize(
        "row",
        [
            None,
            {},
            {"source": "discord"},
            {"source": "discord", "model_config": "not json"},
            {"source": "discord", "model_config": json.dumps(["not", "a", "dict"])},
            {"source": "discord", "model_config": {"_delegate_from": ""}},
            # A /branch child is a real, user-addressable conversation.
            {"source": "discord", "model_config": {"_branched_from": "p"}},
        ],
    )
    def test_leaves_real_sessions_alone(self, row):
        assert is_internal_subagent_row(row) is False


class TestDetectorHalvesAgree:
    """The Python guard and the SQL fence must answer identically (#93208 review).

    Subagent-ness has two homes: ``is_internal_subagent_row`` gates
    ``switch_session``, and ``_NOT_SUBAGENT_ROW_SQL`` fences restart recovery.
    The prior revision disagreed on a blank ``_delegate_from``: Python treated
    ``""`` as absent (``.strip()``) while the SQL used a bare ``IS NULL``,
    under which an empty string is NOT null. The same row was therefore
    refused a route bind but still returned by recovery — one invariant with
    two answers, and the residual case the per-half tests cannot catch because
    each only pins its own side.

    These cases drive BOTH halves over the same rows and assert they agree.
    """

    @pytest.mark.parametrize(
        "marker,is_subagent",
        [
            ("real_parent", True),   # a genuine delegate marker
            ("", False),             # blank -> not a delegate, both halves
            ("   ", False),          # whitespace-only -> .strip() / TRIM
        ],
    )
    def test_blank_marker_treated_identically_by_both_halves(
        self, store_and_db, marker, is_subagent
    ):
        store, db = store_and_db
        key = "agent:main:discord:dm:1540910309580214372"

        db.create_session(
            "probe", source="discord", session_key=key,
            user_id="user-1", chat_id="1540910309580214372", chat_type="dm",
            model_config={"_delegate_from": marker},
        )
        db.append_message(session_id="probe", role="user", content="hi")

        # Half 1 — the Python detector used by switch_session().
        python_says = is_internal_subagent_row(db.get_session("probe"))

        # Half 2 — the SQL fence used by restart recovery. A row it considers
        # a subagent is excluded, so "not recovered" == "SQL says subagent".
        recovered = db.find_latest_gateway_session_for_peer(
            source="discord", session_key=key, user_id="user-1",
            chat_id="1540910309580214372", chat_type="dm",
        )
        sql_says = recovered is None

        assert python_says is is_subagent
        assert sql_says is is_subagent
        assert python_says == sql_says, (
            "the Python guard and the SQL fence disagree on "
            f"_delegate_from={marker!r}: switch_session and restart recovery "
            "would classify the same row differently"
        )


class TestResumeRefusalIsExplained:
    """A refused bind must not read as a transient failure (#93208 review).

    ``switch_session`` returning ``None`` is surfaced by /resume as the generic
    "Failed to switch session." For a user who pastes a subagent id explicitly
    that is misleading — nothing failed, the target is simply not a
    conversation. All four other callers tolerate ``None`` correctly (CLI
    handoff raises, async-completion pinning logs and drops the injection,
    Telegram topic rebinding keeps the incumbent entry), so /resume is the one
    path that needed a specific message.
    """

    def test_locale_string_exists_and_names_the_reason(self):
        from pathlib import Path

        from ruamel.yaml import YAML

        locales = Path(__file__).resolve().parents[2] / "locales" / "en.yaml"
        data = YAML(typ="safe").load(locales.read_text(encoding="utf-8"))
        msg = data["gateway"]["resume"]["blocked_subagent"]

        assert "{name}" in msg, "must interpolate the requested target"
        # It has to tell the user WHY, not just that something failed.
        assert "subagent" in msg.lower()
        assert msg != data["gateway"]["resume"]["switch_failed"]


class TestGuardSurvivesSessionGenerationChanges:
    """The #92872 review's case 1: a compression rotation must not reopen this.

    Merged #69312 makes compression a logical continuation that can advance the
    gateway route from parent ``P`` to live tip ``P2`` while a delegation
    spawned by ``P`` is still running. Any guard keyed on
    ``child.parent_session_id == <currently routed id>`` stops firing at that
    point, because the child still points at ``P`` while the route is on
    ``P2``. This guard reads the *target's* provenance and never the parent
    edge, so the route generation is irrelevant — pinned here so a future
    refactor cannot quietly reintroduce the edge comparison.
    """

    def test_child_cannot_take_the_route_after_its_parent_compressed(
        self, store_and_db
    ):
        store, db = store_and_db
        entry = store.get_or_create_session(_dm_source())
        p_id = entry.session_id

        # C is spawned by P while P is still the routed session.
        db.create_session(
            "child_leaf", source="subagent", parent_session_id=p_id,
            model_config={"_delegate_from": p_id},
        )
        # P then compresses; the route legitimately advances to tip P2.
        db.end_session(p_id, "compression")
        db.create_session("p2_tip", source="discord", parent_session_id=p_id)
        assert store.switch_session(entry.session_key, "p2_tip") is not None
        assert store.get_or_create_session(_dm_source()).session_id == "p2_tip"

        # Now the completion stamped C arrives. C.parent_session_id is P, and
        # the route is P2 — a parent-edge guard would not fire here.
        assert store.switch_session(entry.session_key, "child_leaf") is None

        assert store.get_or_create_session(_dm_source()).session_id == "p2_tip"
        assert db.get_session("p2_tip")["ended_at"] is None

    def test_grandchild_after_compression_is_also_refused(self, store_and_db):
        """Both review objections composed: nested provenance + rotated route."""
        store, db = store_and_db
        entry = store.get_or_create_session(_dm_source())
        p_id = entry.session_id

        db.create_session(
            "child_leaf", source="subagent", parent_session_id=p_id,
            model_config={"_delegate_from": p_id},
        )
        db.create_session(
            "grandchild", source="subagent", parent_session_id="child_leaf",
            model_config={"_delegate_from": "child_leaf"},
        )
        db.end_session(p_id, "compression")
        db.create_session("p2_tip", source="discord", parent_session_id=p_id)
        store.switch_session(entry.session_key, "p2_tip")

        assert store.switch_session(entry.session_key, "grandchild") is None
        assert store.get_or_create_session(_dm_source()).session_id == "p2_tip"
        assert db.get_session("p2_tip")["ended_at"] is None
