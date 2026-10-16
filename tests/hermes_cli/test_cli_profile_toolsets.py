"""A stale ``platform_toolsets.cli`` entry (a toolset whose plugin was uninstalled) must not refuse every
`hermes chat` (#138670): the profile default is the gateway's to resolve, only a picked list rides
session.create. The CLI's launch params go through the real gateway policy build (scripted transport)."""
from contextlib import asynccontextmanager

import pytest

from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore
from tests.hermes_cli.test_gateway_session_store import FakeClient

CONFIG = {"agent": {}, "platform_toolsets": {"cli": ["terminal", "gone_plugin"]}}


@pytest.fixture
def store():
    client = FakeClient()

    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    yield GatewaySessionStore(conn), client
    conn.close()


def _launch(monkeypatch, db, toolsets=None, coding=None):
    """``_build_cli_from_args`` then the session's gateway create; HermesCLI is the real toolset init on a
    bare host. Returns the CLI and the lines its startup printed."""
    import cli as cli_mod
    from hermes_cli.cli_init_mixin import CLIInitMixin
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", CONFIG)
    monkeypatch.setattr("agent.coding_context.coding_selection", lambda **_: coding)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_toolset_keys_nowait", lambda: set())
    printed = []

    class Host(CLIInitMixin):
        def __init__(self, toolsets=None, **_):
            self.model, self.requested_provider, self._resumed = "m1", "auto", False
            self.session_id, self._session_db, self.ignore_rules = "20261012_000000_abcdef", db, False
            self._console_print = printed.append
            self._init_toolsets(toolsets)
    monkeypatch.setattr(cli_mod, "HermesCLI", Host)
    host = cli_mod._build_cli_from_args(None, toolsets, None, None, None, None, None, None, None, False, None,
                                        False, False, False, None)
    host._ensure_gateway_session()
    return host, printed


def _created(client):
    return dict([c for c in client.calls if c[0] == "session.create"][-1][1])


def test_stale_profile_toolset_warns_and_the_session_is_created(store, monkeypatch):
    from gateway.session_local_inventory import _policy_tools
    from gateway.session_policy import build_policy
    db, client = store
    host, printed = _launch(monkeypatch, db)
    assert len(printed) == 1 and "gone_plugin" in printed[0]  # main's "Unknown toolsets" startup warning
    params = _created(client)
    assert "toolsets" not in params
    policy = build_policy(params, CONFIG)
    assert "terminal" in policy.toolsets
    _enabled, tools = _policy_tools(policy, CONFIG)
    assert "terminal" in {tool["function"]["name"] for tool in tools}
    # /new is a fresh session.create on the same profile default.
    host._new_session_id(None)
    assert "toolsets" not in _created(client)


def test_explicit_unknown_toolset_is_still_refused(store, monkeypatch):
    from gateway.session_policy import build_policy
    from hermes_state_runtime import RuntimeStoreError
    db, client = store
    _launch(monkeypatch, db, toolsets="terminal,bogus")
    params = _created(client)
    assert params["toolsets"] == ["terminal", "bogus"]
    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        build_policy(params, CONFIG)


def test_coding_posture_is_left_to_the_gateway(store, monkeypatch):
    db, client = store
    host, _printed = _launch(monkeypatch, db, coding=["terminal"])
    assert host.enabled_toolsets == ["terminal"]  # the local display still shows the posture
    assert "toolsets" not in _created(client)
