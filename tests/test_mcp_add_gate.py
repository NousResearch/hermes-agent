from types import SimpleNamespace

from hermes_cli import mcp_config


def test_pre_mcp_add_blocks_before_probe(monkeypatch):
    # Regression: the gate must reject BEFORE `_probe_single_server`, which for a
    # stdio server launches the (untrusted) command during discovery. A
    # save-time-only gate would run after that execution.
    probed = {"called": False}

    def _fake_probe(name, cfg):
        probed["called"] = True
        return []

    saved = {"called": False}

    monkeypatch.setattr(mcp_config, "_get_mcp_servers", lambda: {})
    monkeypatch.setattr(mcp_config, "_probe_single_server", _fake_probe)
    monkeypatch.setattr(
        mcp_config,
        "invoke_hook",
        lambda hook, **kw: [["blocked by test"]] if hook == "pre_mcp_add" else [],
        raising=False,
    )
    monkeypatch.setattr(
        mcp_config,
        "_save_mcp_server",
        lambda *a, **k: saved.__setitem__("called", True) or True,
    )

    args = SimpleNamespace(
        name="evil",
        url=None,
        mcp_command="node",
        args=["/tmp/evil.js"],
        auth=None,
        preset=None,
        env=None,
        connect_timeout=None,
    )
    mcp_config.cmd_mcp_add(args)

    assert probed["called"] is False  # rejected before discovery/launch
    assert saved["called"] is False  # and never persisted
