"""plugins.manage optional ``profile`` param — per-profile plugins scoping.

Mirrors ``cron.manage`` / ``mcp.servers.*``: when a ``profile`` is passed the
handler resolves ``get_profile_dir(profile)`` and wraps the action dispatch in
``set_hermes_home_override`` / ``reset_hermes_home_override``. Because
``_plugins_dir()`` keys off ``get_hermes_home()``, the list action must then
scan THAT profile's ``plugins/`` dir, not the launch profile's.
"""

from tui_gateway import server


def test_plugins_manage_profile_reads_that_profiles_dir(tmp_path, monkeypatch):
    # A temp profile home with one user plugin in its plugins dir.
    profile_home = tmp_path / "profiles" / "botA"
    plugin_dir = profile_home / "plugins" / "bota-only-plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: bota-only-plugin\nversion: '1.0'\ndescription: BotA-only plugin\n",
        encoding="utf-8",
    )

    # Route the profile name the handler resolves to our temp home.
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: profile_home)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "plugins.manage",
            "params": {"action": "list", "profile": "botA"},
        }
    )

    assert "result" in resp, resp
    user_rows = [
        p for p in resp["result"]["plugins"] if p.get("source") == "user"
    ]
    assert "bota-only-plugin" in [p["name"] for p in user_rows]

    # The override must not leak: an unscoped call after this one resolves the
    # launch profile again.
    from hermes_constants import get_hermes_home_override

    assert get_hermes_home_override() is None


def test_plugins_manage_unknown_profile_errors(tmp_path, monkeypatch):
    import hermes_cli.profiles as profiles

    missing = tmp_path / "profiles" / "ghost"
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: missing)

    resp = server.handle_request(
        {
            "id": "2",
            "method": "plugins.manage",
            "params": {"action": "list", "profile": "ghost"},
        }
    )

    assert "error" in resp, resp
    assert resp["error"]["code"] == 4064


def test_plugins_manage_unscoped_still_lists(monkeypatch):
    # No profile param — the pre-existing contract is unchanged.
    resp = server.handle_request(
        {"id": "3", "method": "plugins.manage", "params": {"action": "list"}}
    )

    assert "result" in resp, resp
    assert "plugins" in resp["result"]
