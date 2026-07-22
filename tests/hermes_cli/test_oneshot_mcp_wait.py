"""Tests for hermes -z joining background MCP discovery before the tool snapshot.

Regression: one-shot started MCP discovery on a background thread but never
joined it, so a still-connecting server's tools were silently absent from the
turn. ``_wait_for_mcp_discovery_before_snapshot`` now bounds-joins that thread.
"""

from hermes_cli import mcp_startup
from hermes_cli import oneshot as oneshot_mod


def test_waits_with_bounded_timeout_when_mcp_configured(monkeypatch):
    monkeypatch.setattr(oneshot_mod, "_read_mcp_server_names", lambda: ({"demo"}, set()))
    calls: dict = {}
    monkeypatch.setattr(mcp_startup, "_resolve_discovery_timeout", lambda _explicit: 1.5)
    monkeypatch.setattr(
        mcp_startup,
        "wait_for_mcp_discovery",
        lambda timeout=None: calls.__setitem__("timeout", timeout),
    )
    monkeypatch.setattr(mcp_startup, "mcp_discovery_in_flight", lambda: False)

    oneshot_mod._wait_for_mcp_discovery_before_snapshot(
        ["demo"],
        use_config_toolsets=False,
    )

    assert calls["timeout"] == 1.5


def test_no_wait_when_no_mcp_configured(monkeypatch):
    monkeypatch.setattr(oneshot_mod, "_read_mcp_server_names", lambda: (set(), set()))
    called = {"n": 0}
    monkeypatch.setattr(
        mcp_startup,
        "wait_for_mcp_discovery",
        lambda timeout=None: called.__setitem__("n", called["n"] + 1),
    )

    oneshot_mod._wait_for_mcp_discovery_before_snapshot()

    # Zero cost for the common no-MCP path — discovery is never joined.
    assert called["n"] == 0


def test_timeout_warns_and_proceeds(monkeypatch, capsys):
    monkeypatch.setattr(
        oneshot_mod, "_read_mcp_server_names", lambda: ({"demo", "other"}, set())
    )
    monkeypatch.setattr(mcp_startup, "_resolve_discovery_timeout", lambda _explicit: 2.75)
    monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", lambda timeout=None: None)
    # Still in flight after the bound == discovery timed out.
    monkeypatch.setattr(mcp_startup, "mcp_discovery_in_flight", lambda: True)

    # Must not raise — the run proceeds with whatever tools did connect.
    oneshot_mod._wait_for_mcp_discovery_before_snapshot(
        ["demo", "other"],
        use_config_toolsets=False,
    )

    err = capsys.readouterr().err
    assert err == (
        "hermes -z: MCP discovery still pending after 2.75s; tools from these "
        "servers may be missing this turn: demo, other\n"
    )


def test_explicit_non_mcp_toolsets_skip_wait(monkeypatch, capsys):
    monkeypatch.setattr(oneshot_mod, "_read_mcp_server_names", lambda: ({"demo"}, set()))
    calls = {"wait": 0}
    monkeypatch.setattr(
        mcp_startup,
        "wait_for_mcp_discovery",
        lambda timeout=None: calls.__setitem__("wait", calls["wait"] + 1),
    )
    monkeypatch.setattr(oneshot_mod, "_run_agent", lambda *_args, **_kwargs: ("done", {}))

    assert oneshot_mod.run_oneshot("hello", toolsets="web,terminal") == 0

    assert calls["wait"] == 0
    assert capsys.readouterr().out == "done\n"


def test_no_mcp_sentinel_in_config_skips_wait(monkeypatch, capsys):
    from hermes_cli import config as config_mod

    config = {
        "mcp_servers": {"demo": {"command": "demo-server"}},
        "platform_toolsets": {"cli": ["web", "no_mcp"]},
    }
    monkeypatch.setattr(oneshot_mod, "_read_mcp_server_names", lambda: ({"demo"}, set()))
    monkeypatch.setattr(config_mod, "load_config", lambda: config)
    calls = {"wait": 0}
    monkeypatch.setattr(
        mcp_startup,
        "wait_for_mcp_discovery",
        lambda timeout=None: calls.__setitem__("wait", calls["wait"] + 1),
    )
    monkeypatch.setattr(oneshot_mod, "_run_agent", lambda *_args, **_kwargs: ("done", {}))

    assert oneshot_mod.run_oneshot("hello") == 0

    assert calls["wait"] == 0
    assert capsys.readouterr().out == "done\n"


def test_config_resolved_timeout_reaches_wait_call(monkeypatch):
    from hermes_cli import config as config_mod

    monkeypatch.setattr(oneshot_mod, "_read_mcp_server_names", lambda: ({"demo"}, set()))
    monkeypatch.setattr(config_mod, "load_config", lambda: {"mcp_discovery_timeout": 7.25})
    calls: dict = {}
    monkeypatch.setattr(
        mcp_startup,
        "wait_for_mcp_discovery",
        lambda timeout=None: calls.__setitem__("timeout", timeout),
    )
    monkeypatch.setattr(mcp_startup, "mcp_discovery_in_flight", lambda: False)

    oneshot_mod._wait_for_mcp_discovery_before_snapshot(
        ["demo"],
        use_config_toolsets=False,
    )

    assert calls["timeout"] == 7.25


def test_temp_hermes_home_waits_before_oneshot_snapshot(monkeypatch, tmp_path, capsys):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "mcp_servers:\n"
        "  demo:\n"
        "    command: demo-server\n"
        "platform_toolsets:\n"
        "  cli:\n"
        "    - web\n"
        "    - demo\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    events: list[str] = []
    monkeypatch.setattr(
        mcp_startup,
        "wait_for_mcp_discovery",
        lambda timeout=None: events.append("wait"),
    )
    monkeypatch.setattr(mcp_startup, "mcp_discovery_in_flight", lambda: False)

    def _snapshot(*_args, **_kwargs):
        events.append("snapshot")
        return "done", {}

    monkeypatch.setattr(oneshot_mod, "_run_agent", _snapshot)

    assert oneshot_mod.run_oneshot("hello") == 0
    assert events == ["wait", "snapshot"]
    assert capsys.readouterr().out == "done\n"
