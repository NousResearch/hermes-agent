"""Tests for the ``post_curator_run`` plugin hook.

Fires once per curator pass, after the per-run report directory is fully
written, with the ``run.json`` path in the payload. Goes through the
standard plugin hook surface (``hermes_cli.plugins.invoke_hook``) so both
Python plugins and shell-script hooks (``agent/shell_hooks.py``) receive
it with no extra wiring. Observer-only: return values are ignored, and a
failing callback must never break the curator.
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
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
        started_at=datetime.now(timezone.utc),
        elapsed_seconds=1.5,
        auto_counts={"checked": 2, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no changes",
        before_report=[],
        before_names=set(),
        after_report=[],
        llm_meta=_llm_meta(),
    )


def test_post_curator_run_is_a_valid_hook():
    """Shell-hook config validation and register_hook() warnings both key
    off VALID_HOOKS, so the event must be declared there."""
    assert "post_curator_run" in _plugins().VALID_HOOKS


def test_hook_fires_with_run_json_path(curator_env):
    curator = curator_env["curator"]
    received = []

    mgr = _plugins().get_plugin_manager()
    mgr._hooks.setdefault("post_curator_run", []).append(
        lambda **kw: received.append(kw)
    )

    run_dir = _write_report(curator)

    assert run_dir is not None
    assert len(received) == 1
    kw = received[0]
    assert kw["run_dir"] == str(run_dir)
    assert kw["run_json_path"] == str(run_dir / "run.json")
    assert kw["report_md_path"] == str(run_dir / "REPORT.md")
    # The report must be fully written by the time the hook fires
    payload = json.loads(Path(kw["run_json_path"]).read_text(encoding="utf-8"))
    assert "added" in payload
    assert Path(kw["report_md_path"]).is_file()


def test_raising_callback_does_not_break_report(curator_env):
    """invoke_hook() isolates per-callback errors; the report writer must
    still return the run dir when a plugin misbehaves."""
    curator = curator_env["curator"]

    def _boom(**kw):
        raise RuntimeError("plugin bug")

    mgr = _plugins().get_plugin_manager()
    mgr._hooks.setdefault("post_curator_run", []).append(_boom)

    run_dir = _write_report(curator)
    assert run_dir is not None
    assert (run_dir / "run.json").exists()


def test_dispatch_layer_failure_does_not_break_report(curator_env, monkeypatch):
    """Even if hook dispatch itself blows up (not just one callback), the
    curator's report path must be unaffected."""
    curator = curator_env["curator"]

    def _explode(*a, **kw):
        raise RuntimeError("dispatch layer down")

    monkeypatch.setattr(_plugins(), "invoke_hook", _explode)

    run_dir = _write_report(curator)
    assert run_dir is not None
    assert (run_dir / "run.json").exists()
