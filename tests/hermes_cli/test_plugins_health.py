"""Plugin load and hook failures are readable from another process (``hermes plugins list``/doctor)."""

import json
import textwrap

from hermes_cli.plugins import get_plugin_manager
from hermes_cli.plugins_health import describe, health_dir, read_plugin_health


def _home(tmp_path, monkeypatch, plugins):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    for name, source in plugins.items():
        plugin = home / "plugins" / name
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(f"name: {name}\nversion: 1.0.0\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(textwrap.dedent(source), encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  enabled: [" + ", ".join(plugins) + "]\n", encoding="utf-8")
    return home


def test_load_and_hook_failures_reach_another_reader(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch, {
        "broken": "def register(ctx):\n    raise RuntimeError('cannot start')\n",
        "noisy": """\
            def _boom(**kwargs):
                raise ValueError('hook went wrong')

            def register(ctx):
                ctx.register_hook('pre_llm_call', _boom)
        """,
    })
    manager = get_plugin_manager()
    try:
        manager.discover_and_load()
        manager.invoke_hook("pre_llm_call", session_id="s")
        # Read from disk only, the way a separate `hermes plugins list` process does.
        health = read_plugin_health(home)
        assert health["broken"]["load_error"] == "cannot start"
        assert describe(health["noisy"]) == [
            "hook pre_llm_call: 1 failed (last: ValueError: hook went wrong)"]
        assert "noisy" not in {k for k, v in health.items() if v["load_error"]}
    finally:
        manager.unload()


def test_records_of_exited_processes_are_ignored(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch, {})
    health_dir(home).mkdir(parents=True)
    dead = {"pid": 2 ** 22 + 12345, "start_time": 1, "load_failures": {"gone": {"error": "old"}},
            "hook_failures": {}}
    (health_dir(home) / "dead.json").write_text(json.dumps(dead), encoding="utf-8")
    assert read_plugin_health(home) == {}
