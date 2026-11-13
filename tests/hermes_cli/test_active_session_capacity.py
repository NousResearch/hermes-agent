"""Idle ownership is exclusive, but does not consume a running-work slot."""

from hermes_cli import active_sessions


def test_idle_desktop_owners_allow_telegram_but_running_turns_obey_cap():
    cfg = {"max_concurrent_sessions": 1}
    owners = []
    for sid in ("desktop-a", "desktop-b"):
        lease, refusal = active_sessions.try_acquire_active_session(
            session_id=sid, surface="desktop", config=cfg,
            metadata={"live_session_id": sid}, track_liveness=True)
        assert refusal is None
        assert lease.set_capacity_active(False, config=cfg) is None
        owners.append(lease)

    telegram, refusal = active_sessions.try_acquire_active_session(
        session_id="telegram-a", surface="gateway:telegram", config=cfg)
    assert refusal is None
    refusal = owners[0].set_capacity_active(True, config=cfg)
    assert refusal.reason == active_sessions.MAX_CONCURRENT_SESSIONS
    foreign, refusal = active_sessions.try_acquire_active_session(
        session_id="desktop-a", surface="desktop", config={},
        metadata={"live_session_id": "different-writer"})
    assert foreign is None
    assert refusal.reason == active_sessions.SESSION_NOT_OWNED

    telegram.release()
    assert owners[0].set_capacity_active(True, config=cfg) is None
    refusal = owners[1].set_capacity_active(True, config=cfg)
    assert refusal.reason == active_sessions.MAX_CONCURRENT_SESSIONS
    for lease in owners:
        lease.release()
