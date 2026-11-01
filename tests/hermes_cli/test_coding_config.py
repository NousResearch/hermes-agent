import pytest

from hermes_cli.config import DEFAULT_CONFIG, load_config, save_config
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_coding_defaults_and_profile_saves_are_isolated(tmp_path):
    coding = DEFAULT_CONFIG["desktop"]["coding"]
    assert coding["show_controls"] is False and coding["default_checkout"] == "worktree"
    for name, enabled in (("coder", True), ("ordinary", False)):
        token = set_hermes_home_override(tmp_path / name)
        try:
            save_config({"desktop": {"coding": {"show_controls": enabled, "default_checkout": "worktree"}}})
            assert load_config()["desktop"]["coding"]["show_controls"] is enabled
        finally:
            reset_hermes_home_override(token)


def test_enabled_profile_does_not_change_absent_coding_preferences(tmp_path):
    from hermes_cli.web_server_config import _normalize_config_for_web
    for name, contents in (("coder", "desktop:\n  coding:\n    show_controls: true\n"),
                           ("chat", "desktop:\n  repo_scan_enabled: false\n")):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(contents)
    for name, expected in (("coder", True), ("chat", False), ("coder", True), ("chat", False)):
        token = set_hermes_home_override(tmp_path / name)
        try:
            assert load_config()["desktop"]["coding"]["show_controls"] is expected
            assert _normalize_config_for_web(load_config())["desktop"]["coding"]["show_controls"] is expected
        finally:
            reset_hermes_home_override(token)
    assert DEFAULT_CONFIG["desktop"]["coding"]["show_controls"] is False


def test_concurrent_config_api_reads_keep_absent_profile_preference_off(tmp_path, monkeypatch):
    import asyncio
    from hermes_cli import profiles
    from hermes_cli.web_routers.config_env import get_config
    root = tmp_path / "deployment"
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    for name, contents in (("coder", "desktop:\n  coding:\n    show_controls: true\n"),
                           ("chat", "desktop:\n  repo_scan_enabled: false\n")):
        home = root / "profiles" / name
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(contents)
    async def read():
        await get_config("coder")
        return await asyncio.gather(*(get_config(name) for name in ("coder", "chat", "chat", "coder")))
    results = asyncio.run(read())
    assert [r["desktop"]["coding"]["show_controls"] for r in results] == [True, False, False, True]
    assert DEFAULT_CONFIG["desktop"]["coding"]["show_controls"] is False


@pytest.mark.parametrize("coding", [{"show_controls": "false"}, {"default_checkout": "reset"}, False])
def test_invalid_coding_preferences_are_rejected_before_write(coding):
    with pytest.raises(ValueError, match="desktop.coding"):
        save_config({"desktop": {"coding": coding}})
