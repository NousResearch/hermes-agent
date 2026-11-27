"""Enabling one plugin never restarts adapters of platforms nothing changed.

Installing or enabling a plugin force-reloads plugins: the sweep unregisters every plugin's platform
first and re-registers each as it loads. A reconcile that read the registry in that window saw a live
platform as "no longer registered", hashed a different build print, and restarted it (enabling Yuanbao
restarted the live QQ Bot and Feishu adapters). The print now reads registrations between sweeps.
"""
import threading

from gateway.config import Platform, PlatformConfig
from gateway.run_platform_reconcile import platform_build_print


def _plugin(home, name, body):
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nversion: '1.0'\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(body, encoding="utf-8")


def test_a_build_print_taken_during_another_plugins_reload_matches_the_settled_one(tmp_path, monkeypatch):
    from hermes_cli import plugins

    home = tmp_path / "home"
    _plugin(home, "aa-slow", "import builtins\n\ndef register(ctx):\n    gate = getattr(builtins, '_hot_gate', None)\n"
                             "    if gate:\n        gate[0].set()\n        gate[1].wait(10)\n")
    _plugin(home, "zz-chat", "def register(ctx):\n    ctx.register_platform(name='zzchat', label='ZZ Chat', "
                             "adapter_factory=lambda cfg: None, check_fn=lambda: True)\n")
    (home / "config.yaml").write_text("plugins:\n  enabled: [aa-slow, zz-chat]\n  disabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = PlatformConfig(enabled=True)
    plugins._reset_plugin_managers_for_tests()
    entered, release = threading.Event(), threading.Event()
    try:
        plugins.discover_plugins()
        platform = Platform("zzchat")  # a plugin-registered platform exists once its plugin registers it
        settled = platform_build_print(platform, config, None)
        monkeypatch.setattr("builtins._hot_gate", (entered, release), raising=False)
        reload = threading.Thread(target=plugins.discover_plugins, kwargs={"force": True})
        reload.start()
        assert entered.wait(10)  # the reload is in flight: zzchat is unregistered until aa-slow returns
        taken = []
        reader = threading.Thread(target=lambda: taken.append(platform_build_print(platform, config, None)))
        reader.start()
        reader.join(0.5)
        release.set()
        reload.join(10)
        reader.join(10)
    finally:
        release.set()
        plugins._reset_plugin_managers_for_tests()

    assert taken == [settled]
