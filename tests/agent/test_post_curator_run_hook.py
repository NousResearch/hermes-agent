"""Tests for the ``post_curator_run`` plugin hook.

Fires once per curator pass, after the per-run report directory is fully
written, with the ``run.json`` path in the payload. Goes through the
standard lifecycle hook surface (``hermes_cli.lifecycle.invoke_hook``) so both
Python plugins and shell-script hooks (``agent/shell_hooks.py``) receive
it with no extra wiring. Observer-only: return values are ignored, and a
failing callback must never break the curator.
"""

from __future__ import annotations

import importlib
import json
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
    from agent import curator
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


def _llm_meta():
    return {
        "final": "short summary of the pass",
        "summary": "short summary",
        "model": "test-model",
        "provider": "test-provider",
        "tool_calls": [],
        "error": None,
    }


def _write_report(curator):
    return curator._write_run_report(
        started_at=datetime.now(UTC),
        elapsed_seconds=1.5,
        auto_counts={"checked": 2, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no changes",
        before_report=[],
        before_names=set(),
        after_report=[],
        llm_meta=_llm_meta(),
    )


def test_hook_fires_once_after_both_report_files_exist(curator_env):
    curator = curator_env["curator"]
    received = []

    def _on(**kw):  # read inside the callback: the files must already be on disk when it fires
        payload = json.loads(Path(kw["run_json_path"]).read_text(encoding="utf-8"))
        received.append((kw, "added" in payload, Path(kw["report_md_path"]).is_file()))

    _plugins().get_plugin_manager()._hooks.setdefault("post_curator_run", []).append(_on)

    run_dir = _write_report(curator)

    assert run_dir is not None
    assert [(kw["run_dir"], kw["run_json_path"], kw["report_md_path"], parsed, md) for kw, parsed, md in received] == [
        (str(run_dir), str(run_dir / "run.json"), str(run_dir / "REPORT.md"), True, True)]


def test_failed_report_write_fires_nothing(curator_env, monkeypatch):
    """Observers are promised both files: a run whose run.json could not be written fires no hook."""
    curator = curator_env["curator"]
    received = []
    _plugins().get_plugin_manager()._hooks.setdefault("post_curator_run", []).append(
        lambda **kw: received.append(kw))
    real_write_text = Path.write_text

    def _disk_full(self, *a, **k):
        if self.name == "run.json":
            raise OSError(28, "No space left on device")
        return real_write_text(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", _disk_full)

    assert _write_report(curator) is not None
    assert received == []
