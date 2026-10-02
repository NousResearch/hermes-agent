"""A user's exposure preference applies before platform-specific publication."""

from hermes_cli import _launchers, config


def test_explicitly_disabled_exposure_does_not_publish_or_register_path(tmp_path, monkeypatch):
    home = tmp_path / "isolated-home"
    home.mkdir()
    (home / "config.yaml").write_text("cli:\n  expose_on_path: false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    config._LOAD_CONFIG_CACHE.clear()
    root = tmp_path / "source"
    root.mkdir()

    def unexpected(*args, **kwargs):
        raise AssertionError("Disabled exposure reached Windows PATH publication")

    monkeypatch.setattr(_launchers, "_register_windows_user_path", unexpected)
    try:
        assert _launchers.expose_cli(root) == {"ok": True, "skipped": "config-disabled"}
        assert not (home / "bin").exists()
    finally:
        config._LOAD_CONFIG_CACHE.clear()
