"""Bounded delayed retry of failed plugin loads (#126356).

A plugin whose import + register() overruns ``plugins.load_timeout_seconds`` during a transient boot
window (unclean-reboot I/O contention) used to stay failed for the process lifetime. These tests pin
the retry contract: bounded attempts, exponential backoff, normal registration on success, a loud
give-up, and no stale re-load after a force re-discovery healed the plugin first.
"""

import logging
import sys
import threading
import time

import pytest
import hermes_yaml as yaml

from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_load_retry import resolve_load_retry_policy, retry_delay_secs


def _write_plugin(base, name, register_body="pass"):
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(
        {"name": name, "version": "0.1.0", "description": f"test {name}"}))
    (plugin_dir / "__init__.py").write_text(f"def register(ctx):\n    {register_body}\n")
    return plugin_dir


def _write_config(home, plugins):
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": plugins}))


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    (home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty-bundled"))
    (tmp_path / "empty-bundled").mkdir()
    return home


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestLoadRetryPolicy:
    def test_defaults_when_unconfigured(self, hermes_home):
        assert resolve_load_retry_policy() == (5, 30.0)

    def test_config_overrides(self, hermes_home):
        _write_config(hermes_home, {"load_retry_attempts": 2, "load_retry_base_seconds": 0.5})
        assert resolve_load_retry_policy() == (2, 0.5)

    def test_zero_attempts_disables(self, hermes_home):
        _write_config(hermes_home, {"load_retry_attempts": 0})
        assert resolve_load_retry_policy() == (0, 30.0)

    def test_backoff_doubles_and_caps(self):
        assert retry_delay_secs(1, 30.0) == 30.0
        assert retry_delay_secs(2, 30.0) == 60.0
        assert retry_delay_secs(3, 30.0) == 120.0
        assert retry_delay_secs(99, 30.0) == 300.0


class TestFailedLoadRetries:
    def test_timed_out_load_is_retried_and_recovers(self, hermes_home):
        """The #126356 shape: register() overruns the deadline during a slow-boot window; once the
        window passes, a bounded retry loads the plugin normally and listeners fire."""
        sys._retry_gate = threading.Event()
        _write_plugin(hermes_home / "plugins", "b_flaky", register_body=(
            "import sys; sys._retry_gate.wait(5); "
            "ctx.register_hook('pre_tool_call', lambda **kw: None)"))
        _write_config(hermes_home, {
            "enabled": ["b_flaky"], "load_timeout_seconds": 0.3,
            "load_retry_attempts": 5, "load_retry_base_seconds": 0.05,
        })
        mgr = PluginManager()
        loaded_events = []
        mgr.on_plugin_loaded(lambda summaries: loaded_events.extend(s["key"] for s in summaries))
        try:
            mgr.discover_and_load()
            assert not mgr._plugins["b_flaky"].enabled
            assert "timed out" in (mgr._plugins["b_flaky"].error or "")
            sys._retry_gate.set()  # the transient window is over
            assert _wait_for(lambda: mgr._plugins["b_flaky"].enabled)
            assert mgr._plugins["b_flaky"].error is None
            assert len(mgr._hooks.get("pre_tool_call", [])) == 1  # registered exactly once
            assert "b_flaky" in loaded_events  # gateway re-wire equivalent fires on the retry
        finally:
            sys._retry_gate.set()
            del sys._retry_gate

    def test_retry_budget_exhausted_gives_up_loudly(self, hermes_home, caplog):
        """A deterministically broken plugin is retried exactly ``load_retry_attempts`` times, then
        left failed with a give-up warning — no infinite retry, no thread left pending."""
        sys._flaky_calls = 0
        _write_plugin(hermes_home / "plugins", "b_broken", register_body=(
            "import sys; sys._flaky_calls += 1; raise RuntimeError('always broken')"))
        _write_config(hermes_home, {
            "enabled": ["b_broken"], "load_retry_attempts": 2, "load_retry_base_seconds": 0.03,
        })
        mgr = PluginManager()
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
                mgr.discover_and_load()
                assert _wait_for(lambda: sys._flaky_calls >= 3)  # initial + 2 retries
                assert _wait_for(lambda: not mgr._load_retry._pending, timeout=5.0)
            assert not mgr._plugins["b_broken"].enabled
            assert "always broken" in (mgr._plugins["b_broken"].error or "")
            assert any("not retrying again" in r.message for r in caplog.records)
        finally:
            del sys._flaky_calls

    def test_zero_attempts_never_retries(self, hermes_home):
        sys._noretry_calls = 0
        _write_plugin(hermes_home / "plugins", "b_off", register_body=(
            "import sys; sys._noretry_calls += 1; raise RuntimeError('boom')"))
        _write_config(hermes_home, {
            "enabled": ["b_off"], "load_retry_attempts": 0, "load_retry_base_seconds": 0.02,
        })
        mgr = PluginManager()
        try:
            mgr.discover_and_load()
            time.sleep(0.3)
            assert sys._noretry_calls == 1
            assert not mgr._plugins["b_off"].enabled
        finally:
            del sys._noretry_calls

    def test_stale_retry_does_not_reload_after_force_rediscovery(self, hermes_home):
        """A retry pending behind a force re-discovery must not re-run register(): the healed
        plugin's manifest identity no longer matches, so the stale entry is a no-op."""
        sys._stale_calls = 0
        plugin_dir = _write_plugin(hermes_home / "plugins", "b_stale", register_body=(
            "import sys; sys._stale_calls += 1; raise RuntimeError('first boot only')"))
        _write_config(hermes_home, {
            "enabled": ["b_stale"], "load_retry_attempts": 3, "load_retry_base_seconds": 0.05,
        })
        mgr = PluginManager()
        try:
            mgr.discover_and_load()
            assert not mgr._plugins["b_stale"].enabled
            assert sys._stale_calls == 1
            # Operator fixes the plugin and force-reloads before the pending retry fires.
            (plugin_dir / "__init__.py").write_text("def register(ctx):\n    pass\n")
            mgr.discover_and_load(force=True)
            assert mgr._plugins["b_stale"].enabled
            time.sleep(0.4)  # several retry intervals pass
            assert sys._stale_calls == 1  # the stale retry never re-ran the module
            assert mgr._plugins["b_stale"].enabled
        finally:
            if hasattr(sys, "_stale_calls"):
                del sys._stale_calls
