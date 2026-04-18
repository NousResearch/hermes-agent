from gateway.direct_control_router import DIRECT_CONTROL_ROUTER_METHODS
from gateway.run import _DIRECT_CONTROL_ROUTER_METHODS as RUNNER_DIRECT_CONTROL_ROUTER_METHODS


def test_gateway_runner_uses_router_direct_control_registry():
    assert RUNNER_DIRECT_CONTROL_ROUTER_METHODS is DIRECT_CONTROL_ROUTER_METHODS
    assert "_try_handle_admin_qq_send_shortcut" in RUNNER_DIRECT_CONTROL_ROUTER_METHODS
    assert "_try_handle_admin_qq_group_moderation" in RUNNER_DIRECT_CONTROL_ROUTER_METHODS
    assert "_try_handle_admin_weixin_group_moderation" in RUNNER_DIRECT_CONTROL_ROUTER_METHODS
    assert "_match_admin_qq_send_request" not in RUNNER_DIRECT_CONTROL_ROUTER_METHODS
    assert "_load_qq_group_runtime_status_details" not in RUNNER_DIRECT_CONTROL_ROUTER_METHODS
