"""Runtime guard tests for oneshot's direct SessionDB writer path."""

from pathlib import Path


def test_oneshot_skips_direct_writer_when_gateway_is_active(monkeypatch, tmp_path: Path):
    import gateway.status
    import hermes_cli.config
    import hermes_cli.oneshot as oneshot
    import hermes_state

    monkeypatch.setattr(hermes_cli.config, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        gateway.status,
        "get_running_pid",
        lambda *, cleanup_stale=False: 4242,
    )

    opened = []
    real_session_db = hermes_state.SessionDB

    class SpySessionDB(real_session_db):
        def __init__(self, *args, **kwargs):
            opened.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(hermes_state, "SessionDB", SpySessionDB)

    assert oneshot._create_session_db_for_oneshot() is None
    assert opened == []
