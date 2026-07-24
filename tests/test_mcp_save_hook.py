import hermes_cli.plugins as plugins_mod
from hermes_cli import mcp_config


def _patch_gate(monkeypatch, returns):
    """Patch plugins-level discovery + hook invocation.

    The gate sites call ``collect_hook_block_reasons``, which resolves
    ``discover_plugins``/``invoke_hook`` as globals of hermes_cli.plugins —
    patch there, not on the consuming module.

    NOTE: the callback's first positional param must not be named "name" —
    the gate calls invoke_hook("pre_mcp_add", name=name, ...), and a "name"
    positional param collides with that keyword (TypeError: got multiple
    values for argument 'name'). Use hook_name.
    """
    monkeypatch.setattr(plugins_mod, "discover_plugins", lambda force=False: None)
    monkeypatch.setattr(
        plugins_mod,
        "invoke_hook",
        lambda hook_name, **kw: returns if hook_name == "pre_mcp_add" else [],
    )


def test_pre_mcp_add_block_rejects_save(monkeypatch):
    _patch_gate(monkeypatch, [["mcp blocked by test"]])
    # No IOC in this entry, so validate_mcp_server_entry alone would allow it.
    saved = mcp_config._save_mcp_server("x", {"command": "npx", "args": ["@scope/s"]})
    assert saved is False


def test_pre_mcp_add_block_str_rejects_save(monkeypatch):
    # Hook callbacks may also return a bare string rather than a list — the
    # flatten logic in collect_hook_block_reasons must handle both shapes.
    _patch_gate(monkeypatch, ["mcp blocked str"])
    saved = mcp_config._save_mcp_server("x", {"command": "npx", "args": ["@scope/s"]})
    assert saved is False


def test_pre_mcp_add_block_rejects_bulk_replace(monkeypatch):
    # Dashboard whole-map edits go through _replace_mcp_servers, NOT
    # _save_mcp_server — the gate must fire there too (PR review finding).
    _patch_gate(monkeypatch, [["mcp blocked by test"]])
    ok, issues = mcp_config._replace_mcp_servers(
        {"x": {"command": "npx", "args": ["@scope/s"]}}
    )
    assert ok is False
    assert any("mcp blocked by test" in i for i in issues)


def test_bulk_replace_proceeds_without_callbacks(monkeypatch):
    _patch_gate(monkeypatch, [])
    saved = {}
    monkeypatch.setattr(mcp_config, "load_config", lambda: {})
    monkeypatch.setattr(mcp_config, "save_config", lambda cfg: saved.update(cfg))
    ok, issues = mcp_config._replace_mcp_servers(
        {"x": {"command": "npx", "args": ["@scope/s"]}}
    )
    assert ok is True and issues == []
    assert saved["mcp_servers"]["x"]["command"] == "npx"
