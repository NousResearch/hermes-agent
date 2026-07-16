from hermes_cli.plugins import VALID_HOOKS


def test_install_gate_hooks_are_registered_events():
    assert "pre_plugin_install" in VALID_HOOKS
    assert "pre_mcp_add" in VALID_HOOKS
