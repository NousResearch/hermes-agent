from gateway.direct_control_router import DIRECT_CONTROL_ROUTER_METHODS
from gateway.direct_shortcuts import DIRECT_CONTROL_ROUTER_METHODS as SHORTCUT_ROUTER_METHODS


def test_gateway_runner_uses_router_direct_control_registry():
    # Router methods live on DirectControlRouter; direct_shortcuts reuses the
    # same registry instead of a private copy on gateway.run.
    assert SHORTCUT_ROUTER_METHODS is DIRECT_CONTROL_ROUTER_METHODS
    assert "_try_handle_admin_qq_send_shortcut" in DIRECT_CONTROL_ROUTER_METHODS
    assert "_try_handle_admin_qq_group_moderation" in DIRECT_CONTROL_ROUTER_METHODS
    assert "_try_handle_admin_weixin_group_moderation" in DIRECT_CONTROL_ROUTER_METHODS
    assert "_match_admin_qq_send_request" not in DIRECT_CONTROL_ROUTER_METHODS
    assert "_load_qq_group_runtime_status_details" not in DIRECT_CONTROL_ROUTER_METHODS
