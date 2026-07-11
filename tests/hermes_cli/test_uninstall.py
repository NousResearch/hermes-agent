from hermes_cli import uninstall


def test_full_uninstall_extra_paths_include_macos_desktop_state(tmp_path):
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    app_root = tmp_path / "Applications"

    paths = uninstall.full_uninstall_extra_paths(
        hermes_home,
        home=home,
        system_name="Darwin",
        system_applications=app_root,
    )

    expected = {
        home / ".cache" / "hermes",
        home / ".config" / "hermes",
        home / ".local" / "share" / "hermes",
        home / "Applications" / "Hermes.app",
        app_root / "Hermes.app",
        home / "Library" / "Application Support" / "Hermes",
        home / "Library" / "Caches" / "com.nousresearch.hermes",
        home / "Library" / "Preferences" / "com.nousresearch.hermes.plist",
        home / "Library" / "Saved Application State" / "com.nousresearch.hermes.savedState",
    }

    assert expected.issubset(set(paths))
    assert hermes_home not in paths


def test_remove_full_uninstall_leftovers_removes_existing_targets(tmp_path):
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    app_root = tmp_path / "Applications"

    cache_dir = home / ".cache" / "hermes"
    pref_file = home / "Library" / "Preferences" / "com.nousresearch.hermes.plist"
    app_bundle = app_root / "Hermes.app"

    cache_dir.mkdir(parents=True)
    pref_file.parent.mkdir(parents=True)
    pref_file.write_text("plist", encoding="utf-8")
    app_bundle.mkdir(parents=True)

    removed = uninstall.remove_full_uninstall_leftovers(
        hermes_home,
        home=home,
        system_name="Darwin",
        system_applications=app_root,
    )

    assert cache_dir in removed
    assert pref_file in removed
    assert app_bundle in removed
    assert not cache_dir.exists()
    assert not pref_file.exists()
    assert not app_bundle.exists()


def test_remove_legacy_macos_launchd_plists_boots_out_and_unlinks(tmp_path, monkeypatch):
    home = tmp_path / "home"
    launch_agents = home / "Library" / "LaunchAgents"
    plist_a = launch_agents / "ai.hermes.gateway.plist"
    plist_b = launch_agents / "io.nousresearch.hermes-agent.gateway-old.plist"
    plist_a.parent.mkdir(parents=True)
    plist_a.write_text("plist-a", encoding="utf-8")
    plist_b.write_text("plist-b", encoding="utf-8")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return None

    monkeypatch.setattr(uninstall.subprocess, "run", fake_run)
    monkeypatch.setattr(uninstall.os, "getuid", lambda: 501, raising=False)

    removed = uninstall.remove_legacy_macos_launchd_plists(home=home)

    assert removed == [plist_a, plist_b]
    assert not plist_a.exists()
    assert not plist_b.exists()
    assert ["launchctl", "bootout", "gui/501/ai.hermes.gateway"] in calls
    assert [
        "launchctl",
        "bootout",
        "gui/501/io.nousresearch.hermes-agent.gateway-old",
    ] in calls
