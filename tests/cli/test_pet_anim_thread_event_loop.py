"""Regression tests for the CLI pet animation thread event-loop safety.

The pet animation loop runs on a daemon thread.  Before the fix it called
prompt_toolkit's ``app.invalidate()`` directly from that thread, which
crashed with ``RuntimeError: There is no current event loop`` when Python's
async-generator finalizers or prompt_toolkit's exception handler fired.

Issue: the daemon thread also started unconditionally even when no pet was
enabled, so users without a pet still hit the noisy error.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._app = None
    cli_obj._pet_lock = threading.Lock()
    # We only exercise _pet_start_anim / _pet_anim_loop; state used by those
    # methods is minimal and set explicitly below.
    cli_obj._pet_enabled = False
    cli_obj._pet_anim_running = False
    cli_obj._pet_anim_thread = None
    cli_obj._pet_cfg_checked = 0.0
    cli_obj._pet_cfg_interval = 0.01
    cli_obj._pet_frame_interval = 0.01
    cli_obj._pet_frame_idx = 0
    return cli_obj


class TestPetAnimThreadEventLoop:
    def test_start_anim_skips_thread_when_pet_disabled(self, monkeypatch, tmp_path):
        """When no pet is enabled the daemon thread must not start."""
        from hermes_cli.config import load_config, save_config

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))

        cfg = load_config()
        cfg.setdefault("display", {}).setdefault("pet", {})
        cfg["display"]["pet"]["enabled"] = False
        save_config(cfg)

        cli_obj = _make_cli()
        cli_obj._pet_start_anim()
        assert cli_obj._pet_anim_running is False
        assert cli_obj._pet_anim_thread is None

    def test_anim_loop_schedules_invalidate_on_app_loop(self):
        """_pet_anim_loop must call app.invalidate via the event loop thread-safely."""

        recorded = []

        class FakeLoop:
            def __init__(self):
                self._running = True

            def is_running(self):
                return self._running

            def call_soon_threadsafe(self, callback):
                recorded.append(callback)

        class FakeApp:
            def __init__(self):
                self.loop = FakeLoop()
                self.invalidated = False

            def invalidate(self):
                self.invalidated = True

        app = FakeApp()
        cli_obj = _make_cli()
        cli_obj._pet_enabled = True
        cli_obj._app = app
        cli_obj._pet_anim_running = True
        # Keep the pet enabled; don't let config resolution turn it off.
        cli_obj._pet_resolve_config = lambda: None

        # Run the loop for a few frames on a background thread, then stop it.
        thread = threading.Thread(target=cli_obj._pet_anim_loop, daemon=True)
        thread.start()
        time.sleep(0.05)
        cli_obj._pet_anim_running = False
        thread.join(timeout=1.0)
        assert not thread.is_alive(), "pet animation thread did not stop"

        assert recorded, "expected call_soon_threadsafe callbacks from the anim loop"
        # Each recorded callback should be app.invalidate; invoking it proves it
        # was scheduled correctly.
        for callback in recorded:
            callback()
        assert app.invalidated is True

    def test_anim_loop_never_calls_get_event_loop(self):
        """The daemon thread must not rely on asyncio.get_event_loop()."""

        class FakeLoop:
            def call_soon_threadsafe(self, callback):
                pass

        class FakeApp:
            loop = FakeLoop()

            def invalidate(self):
                pass

        cli_obj = _make_cli()
        cli_obj._pet_enabled = True
        cli_obj._app = FakeApp()
        cli_obj._pet_anim_running = True

        def _target():
            # Run a couple of iterations; the thread-safe scheduling path should
            # never touch asyncio.get_event_loop().
            for _ in range(3):
                time.sleep(0.01)
                cli_obj._pet_resolve_config = lambda: None
                with cli_obj._pet_lock:
                    cli_obj._pet_frame_idx += 1
                app = getattr(cli_obj, "_app", None)
                loop = getattr(app, "loop", None) if app is not None else None
                if loop is not None:
                    loop.call_soon_threadsafe(app.invalidate)

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
