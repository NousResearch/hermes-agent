from pathlib import Path


def test_cli_init_does_not_open_canonical_db_with_live_gateway(monkeypatch):
    import cli
    import gateway.status
    import hermes_state

    console = object.__new__(cli.HermesCLI)
    opened = []

    class ForbiddenSessionDB:
        def __init__(self, *args, **kwargs):
            opened.append((args, kwargs))
            raise AssertionError("CLI must not open SessionDB while Gateway is active")

    monkeypatch.setattr(gateway.status, "get_running_pid", lambda *, cleanup_stale=False: 69521)
    monkeypatch.setattr(cli, "_run_checkpoint_auto_maintenance", lambda: None)
    monkeypatch.setattr(hermes_state, "SessionDB", ForbiddenSessionDB)

    console._init_session_store()

    assert console._cli_persistence_disabled is True
    assert console._session_db is None
    assert opened == []


def test_titled_session_creation_refuses_live_gateway(monkeypatch):
    import gateway.status
    import hermes_state
    from hermes_cli import main

    opened = []

    class ForbiddenSessionDB:
        def __init__(self, *args, **kwargs):
            opened.append((args, kwargs))
            raise AssertionError("titled session creation must not open SessionDB")

    monkeypatch.setattr(gateway.status, "get_running_pid", lambda *, cleanup_stale=False: 69521)
    monkeypatch.setattr(hermes_state, "SessionDB", ForbiddenSessionDB)

    assert main._create_titled_session("blocked") is None
    assert opened == []


def test_cli_init_allows_session_db_when_gateway_is_stopped(monkeypatch):
    import cli
    import gateway.status
    import hermes_state

    console = object.__new__(cli.HermesCLI)
    opened = []

    class SpySessionDB:
        def __init__(self, *args, **kwargs):
            opened.append((args, kwargs))

    monkeypatch.setattr(gateway.status, "get_running_pid", lambda *, cleanup_stale=False: None)
    monkeypatch.setattr(hermes_state, "SessionDB", SpySessionDB)
    monkeypatch.setattr(cli, "_run_state_db_auto_maintenance", lambda db: None)
    monkeypatch.setattr(cli, "_run_checkpoint_auto_maintenance", lambda: None)

    console._init_session_store()

    assert console._cli_persistence_disabled is False
    assert console._session_db is not None
    assert len(opened) == 1
