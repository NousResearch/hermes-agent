from hermes_cli import mcp_config


def test_pre_mcp_add_block_rejects_save(monkeypatch):
    # NOTE: the callback's first positional param must not be named "name" —
    # _save_mcp_server calls invoke_hook("pre_mcp_add", name=name, ...), and a
    # "name" positional param collides with that keyword (TypeError: got
    # multiple values for argument 'name'). Use hook_name, matching the
    # pre_plugin_install hook test.
    monkeypatch.setattr(
        mcp_config,
        "invoke_hook",
        lambda hook_name, **kw: (
            [["mcp blocked by test"]] if hook_name == "pre_mcp_add" else []
        ),
        raising=False,
    )
    # No IOC in this entry, so validate_mcp_server_entry alone would allow it.
    saved = mcp_config._save_mcp_server("x", {"command": "npx", "args": ["@scope/s"]})
    assert saved is False


def test_pre_mcp_add_block_str_rejects_save(monkeypatch):
    # Hook callbacks may also return a bare string rather than a list — the
    # flatten logic in _save_mcp_server must handle both shapes.
    monkeypatch.setattr(
        mcp_config,
        "invoke_hook",
        lambda hook_name, **kw: (
            ["mcp blocked str"] if hook_name == "pre_mcp_add" else []
        ),
        raising=False,
    )
    saved = mcp_config._save_mcp_server("x", {"command": "npx", "args": ["@scope/s"]})
    assert saved is False
