"""REST transcript endpoints and Bot Chat roster resolvers must follow a local policy receipt's
``current_target`` after a reset (matching ``authority.attach``), allow ``oneshot``-created
canonical Bot Chats, and skip empty 0-message cron rows in ``last_session``."""
import json
from persistence.conversation.sessions import NewSession
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence_assembly import open_persistence, profile_scope
from gateway.session_persistence import owner_rule
from hermes_cli.web_routers.sessions import _resolve_display_session_id


def test_resolve_display_session_id_and_roster_follow_reset_receipt(tmp_path):
    with open_persistence(profile_scope(home=tmp_path), eager=True) as store:
        root_id = "local-bot-root-0001"
        reset_id = "20261121_093957_reset1"
        store.sessions.create(NewSession(
            session_id=root_id, source="desktop", model="m",
            chat_id=root_id, user_id="uid:1001",
        ))
        store.sessions.set_hidden(root_id, True)
        store.sessions.set_title(root_id, "Bot Chat")
        store.sessions.create(NewSession(
            session_id=reset_id, source="desktop", model="m",
            parent_session_id=root_id, chat_id=root_id, user_id="uid:1001",
            model_config={"_reset_from": root_id},
        ))
        holder = store.sessions._store.holder()
        receipt = {
            "profile_id": str(tmp_path), "principal_id": "uid:1001", "request_id": "r1",
            "session_id": root_id, "route": f"agent:main:local:dm:{root_id}",
            "entry": {"session_id": reset_id}, "policy": {},
            "lineage": [root_id, reset_id],
        }
        def _seed(conn):
            conn.execute("UPDATE sessions SET end_reason='session_reset', ended_at=1000 WHERE id=?", (root_id,))
            conn.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                (POLICY_PREFIX + root_id, json.dumps(receipt)),
            )
        holder._execute_write(_seed)

        # resolve_resume_target intentionally stops before _reset_from, but display resolution follows the receipt
        assert store.lineage.resolve_resume_target(root_id) == root_id
        assert _resolve_display_session_id(store, root_id) == reset_id

        roster = store.sessions.roster_sessions(owner_user_id="uid:1001", owner=owner_rule())
        assert roster.canonical_session is not None
        assert roster.canonical_session.id == root_id
        assert roster.canonical_session.resolved_id == reset_id
