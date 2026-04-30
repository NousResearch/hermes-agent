"""Verify LINE integration points exist in the gateway runner and cron scheduler.

Guards against Platform.LINE being accidentally omitted from authorization
maps (which would silently drop every incoming LINE message) and from the
cron scheduler's platform_map (which would silently ignore deliver='line').
"""
import inspect

from gateway.run import GatewayRunner


def test_line_in_authorization_maps():
    """Source-text guard: the platform_env_map / platform_allow_all_map dicts
    inside ``_is_user_authorized`` are method-local (rebuilt per call), so we
    can't import them as data. Asserting on the source literal text is a
    pragmatic regression guard that catches the most likely failure mode —
    forgetting to add ``Platform.LINE`` to one of the maps when the rest of
    the wiring is in place. A full behavioural test would require building a
    GatewayConfig + GatewayRunner + MessageEvent fixture, an order of
    magnitude more setup for the same coverage."""
    src = inspect.getsource(GatewayRunner._is_user_authorized)
    assert "Platform.LINE" in src, (
        "Platform.LINE missing from GatewayRunner._is_user_authorized — "
        "every LINE message will fail authorization"
    )
    assert "LINE_ALLOWED_USERS" in src
    assert "LINE_ALLOW_ALL_USERS" in src


def test_line_in_cron_known_delivery_platforms():
    """'line' must be in _KNOWN_DELIVERY_PLATFORMS or bare deliver='line' is silently dropped."""
    from cron.scheduler import _KNOWN_DELIVERY_PLATFORMS
    assert "line" in _KNOWN_DELIVERY_PLATFORMS, (
        "'line' missing from _KNOWN_DELIVERY_PLATFORMS — "
        "cronjob(deliver='line') using a home channel will produce no delivery"
    )


def test_line_in_cron_home_target_env_vars():
    """LINE_HOME_CHANNEL must be registered so hermes setup-configured home channels work."""
    from cron.scheduler import _HOME_TARGET_ENV_VARS
    assert "line" in _HOME_TARGET_ENV_VARS, (
        "'line' missing from _HOME_TARGET_ENV_VARS — "
        "LINE_HOME_CHANNEL is unreachable for cron home-channel delivery"
    )
    assert _HOME_TARGET_ENV_VARS["line"] == "LINE_HOME_CHANNEL"


def test_line_group_room_chat_type_bypasses_user_allowlist_in_runner():
    """LINE group/room chat_type sources must be auto-authorized at the gateway
    layer because the LINE adapter's own LINE_ALLOWED_GROUPS/ROOMS check
    already validated them before dispatch.

    Regression guard: a previous version of this PR only registered
    LINE_ALLOWED_USERS in `_is_user_authorized`'s `platform_env_map`, which
    meant group/room messages were rejected at the gateway layer even when
    the LINE adapter passed them."""
    src = inspect.getsource(GatewayRunner._is_user_authorized)
    assert "Platform.LINE" in src and 'chat_type in {"group", "room"}' in src, (
        "_is_user_authorized must short-circuit LINE group/room sources — "
        "otherwise LINE_ALLOWED_GROUPS/ROOMS messages get gateway-level rejection"
    )
