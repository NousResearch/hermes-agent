"""The Telegram Mini App backend ships as a bundled dashboard plugin (plugins/telegram-miniapp/).

Its routes are mounted by the dashboard plugin API under ``/api/plugins/telegram-miniapp/``, not by
a core router; the ``dashboard_auth/telegram_miniapp`` provider registers those paths for token auth.
"""

from hermes_cli import web_server_dashboard


def test_discovered_as_bundled_api_only_plugin():
    plugin = next(p for p in web_server_dashboard._discover_dashboard_plugins() if p["name"] == "telegram-miniapp")

    assert plugin["source"] == "bundled"
    assert plugin["has_api"] is True
    assert plugin["tab"].get("hidden") is True


def test_routes_mounted_under_plugin_prefix_only():
    from hermes_cli.web_server import app

    paths = {getattr(route, "path", "") for route in app.router.routes}

    assert "/api/plugins/telegram-miniapp/me" in paths
    assert "/api/plugins/telegram-miniapp/allowlist" in paths
    assert "/api/plugins/telegram-miniapp/allowlist/{user_id}" in paths
    assert not {"/api/miniapp/me", "/api/telegram/allowlist"} & paths


def test_provider_registers_the_plugin_paths_for_token_auth():
    from plugins.dashboard_auth import telegram_miniapp

    assert telegram_miniapp.MINIAPP_API == "/api/plugins/telegram-miniapp"
