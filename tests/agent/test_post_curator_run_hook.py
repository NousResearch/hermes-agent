"""Tests for the ``post_curator_run`` plugin hook.

Fires once per curator pass, after the per-run report directory is fully
written, with the ``run.json`` path in the payload. Goes through the
standard lifecycle hook surface (``hermes_cli.lifecycle.invoke_hook``) so both
Python plugins and shell-script hooks (``hermes_agent/agent/shell_hooks.py``) receive
it with no extra wiring. Observer-only: return values are ignored, and a
failing callback must never break the curator.
"""

from __future__ import annotations

import importlib
import json
import threading
from datetime import datetime, timezone, UTC
from pathlib import Path

import pytest


def _plugins():
    """Resolve hermes_cli.plugins at call time, not import time.

    Some test files (e.g. tests/plugins/test_security_guidance_plugin.py)
    delete ``hermes_cli.plugins`` from ``sys.modules`` and re-import it,
    which would leave a module-level binding here pointing at a stale
    module object whose PluginManager the curator never sees.
    """
    return importlib.import_module("hermes_cli.plugins")


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a skills/ dir + reset curator module state."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import hermes_constants
    importlib.reload(hermes_constants)
    from hermes_agent.agent import curator
    importlib.reload(curator)
    yield {"home": home, "curator": curator}


@pytest.fixture(autouse=True)
def _fresh_plugin_manager():
    """Each test gets a fresh PluginManager so hook callbacks don't
    leak between tests."""
    plugins = _plugins()
    original = plugins._plugin_manager
    plugins._plugin_manager = plugins.PluginManager()
    yield
    plugins._plugin_manager = original


def _run_pass(curator):
    """One real prune-only pass (no LLM fork), the path a scheduled tick and `hermes curator run` take."""
    curator.run_curator_review(synchronous=True, consolidate=False)
    return curator.load_state().get("last_report_path")


def test_hook_fires_once_after_the_pass_with_both_report_files(curator_env):
    curator = curator_env["curator"]
    received = []

    def _on(**kw):  # read inside the callback: the files and the pass's state are final when it fires
        payload = json.loads(Path(kw["run_json_path"]).read_text(encoding="utf-8"))
        received.append((kw, "added" in payload, Path(kw["report_md_path"]).is_file(),
                         "consolidation off" in curator.load_state()["last_run_summary"]))

    _plugins().get_plugin_manager()._hooks.setdefault("post_curator_run", []).append(_on)

    run_dir = Path(_run_pass(curator))

    assert [(kw["run_dir"], kw["run_json_path"], kw["report_md_path"], parsed, md, saved)
            for kw, parsed, md, saved in received] == [
        (str(run_dir), str(run_dir / "run.json"), str(run_dir / "REPORT.md"), True, True, True)]


def test_a_pass_without_its_own_complete_report_fires_nothing(curator_env, monkeypatch):
    """Observers are promised THIS pass's files: no hook when run.json fails to write, nor when the
    report dir can't be created while the previous pass's report is still on disk."""
    curator = curator_env["curator"]
    received = []
    _plugins().get_plugin_manager()._hooks.setdefault("post_curator_run", []).append(
        lambda **kw: received.append(kw))
    _run_pass(curator)  # a complete pass first: its report must never be announced again
    received.clear()
    real_write_text, real_mkdir = Path.write_text, Path.mkdir

    def _no_run_dir(self, *a, **k):
        if self.parent.name == "curator" and self.parent.parent.name == "logs":
            raise OSError(28, "No space left on device")
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", _no_run_dir)
    _run_pass(curator)  # no report dir this time; the previous pass's report is still on disk
    monkeypatch.setattr(Path, "mkdir", real_mkdir)

    def _disk_full(self, *a, **k):
        if self.name == "run.json":
            raise OSError(28, "No space left on device")
        return real_write_text(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", _disk_full)
    _run_pass(curator)
    assert received == []


def test_dry_run_pass_fires_flagged_with_the_hooks_test_payload_shape(curator_env):
    """A preview still announces its report, flagged so observers don't act on it; ``hermes hooks
    test post_curator_run`` sends the same keys a real pass does."""
    from hermes_cli.hooks import _DEFAULT_PAYLOADS
    curator = curator_env["curator"]
    received = []
    _plugins().get_plugin_manager()._hooks.setdefault("post_curator_run", []).append(
        lambda **kw: received.append(kw))
    curator.run_curator_review(synchronous=True, consolidate=False, dry_run=True)
    _run_pass(curator)
    assert [kw["dry_run"] for kw in received] == [True, False]
    assert set(received[0]) - {"telemetry_schema_version"} == set(_DEFAULT_PAYLOADS["post_curator_run"])


def test_hook_waits_for_in_flight_discovery_once_not_on_every_call(monkeypatch):
    """A hook fired mid-discovery sees the plugins it loads; a scan hung in a plugin's register()
    costs one bounded wait, not one per hook call."""
    plugins = _plugins()
    manager = plugins.get_plugin_manager()
    manager._discovered = True
    received, joins, release = [], [], threading.Event()

    class _Scan(threading.Thread):
        def join(self, timeout=None):
            joins.append(timeout)
            manager._hooks.setdefault("post_curator_run", []).append(lambda **kw: received.append(kw))

    hung = _Scan(target=release.wait, daemon=True)
    hung.start()
    monkeypatch.setattr(plugins, "_background_discovery_thread", hung)
    monkeypatch.setattr(plugins, "_delivery_join_abandoned", False, raising=False)
    try:
        plugins.invoke_hook("post_curator_run", run_dir="d")
        plugins.invoke_hook("post_curator_run", run_dir="d")
    finally:
        release.set()
    assert (len(joins), len(received)) == (1, 2)
