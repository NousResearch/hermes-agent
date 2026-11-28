"""Hidden canonical sessions (e.g. Bot Chat) must stay hidden across compression and reset
continuations so they never leak into the visible regular session list, while explicit branches
remain visible."""
from persistence.conversation.sessions import NewSession
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence_assembly import open_persistence, profile_scope


def test_hidden_inherited_on_compression_and_reset_not_branch(tmp_path):
    with open_persistence(profile_scope(home=tmp_path), eager=True) as store:
        store.sessions.create(NewSession(session_id="bot_root", source="desktop", model="m"))
        store.sessions.set_hidden("bot_root", True)
        store.sessions.set_title("bot_root", "Bot Chat")

        # Reset continuation via create_session (worker_lifecycle path) stays hidden
        store.sessions.create(NewSession(
            session_id="bot_reset", source="desktop", model="m",
            parent_session_id="bot_root", model_config={"_reset_from": "bot_root"},
        ))
        assert store.sessions.get("bot_reset").hidden is True

        # Branch child remains visible
        store.sessions.create(NewSession(
            session_id="bot_branch", source="desktop", model="m",
            parent_session_id="bot_root", model_config={"_branched_from": "bot_root"},
        ))
        assert store.sessions.get("bot_branch").hidden is False

        # Compression continuation via publish_compression_child stays hidden
        holder = store.sessions._store.holder()
        publish_compression_child(
            holder,
            parent_session_id="bot_root",
            child_session_id="bot_compressed",
            source="desktop",
            messages=[{"role": "user", "content": "compaction summary"}],
            model="m",
            require_compression_lease=False,
        )
        assert store.sessions.get("bot_compressed").hidden is True
