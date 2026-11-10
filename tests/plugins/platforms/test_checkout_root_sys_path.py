"""Importing a platform adapter or a gateway/cron core module must not reorder ``sys.path``.

These modules used to run ``sys.path.insert(0, <checkout root>)`` at import, so checkout-level
packages (``evals``, ``utils``, ``tests``, ...) shadowed same-named installed packages for every later
import in the process. ``sys.path`` is process-global, so each import runs in a fresh interpreter.

``hermes_agent.gateway.run`` is not listed: it imports ``hermes_bootstrap``, whose ``harden_import_path`` relocates
the root to the front on purpose (entry-point hardening against a cwd ``utils/`` package).
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULES = [
    "hermes_agent.plugins.platforms.telegram.adapter",
    "hermes_agent.plugins.platforms.whatsapp.adapter",
    "hermes_agent.plugins.platforms.discord.adapter",
    "hermes_agent.plugins.platforms.slack.adapter",
    "hermes_agent.plugins.platforms.raft.adapter",
    "gateway.platforms.base",
    "hermes_agent.cron.scheduler",
    "tools.cronjob_tools",
]


@pytest.mark.parametrize("module", MODULES)
def test_import_keeps_installed_package_ahead_of_checkout(module, tmp_path):
    """A same-named installed package still wins after the import, and sys.path is unchanged."""
    site = tmp_path / "site"
    (site / "evals").mkdir(parents=True)
    (site / "evals" / "__init__.py").write_text("MARKER = 'installed'\n", encoding="utf-8")
    script = textwrap.dedent(f"""
        import importlib, os, sys
        repo = {str(REPO_ROOT)!r}
        sys.path[:] = [{str(site)!r}, repo] + [
            p for p in sys.path if p and os.path.realpath(p) != os.path.realpath(repo)]
        before = list(sys.path)
        importlib.import_module({module!r})
        assert sys.path[:len(before)] == before, sys.path[:4]
        import evals
        assert getattr(evals, "MARKER", None) == "installed", evals.__file__
    """)
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"))
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True, cwd=str(tmp_path), env=env,
        timeout=180)
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize("module", [
    "hermes_agent.plugins.platforms.slack.adapter",
    "hermes_agent.plugins.platforms.raft.adapter",
    "hermes_agent.cron.scheduler",
])
def test_standalone_file_load_still_resolves_repo_siblings(module, tmp_path):
    """Loaded by file path with the checkout absent from sys.path, the appended root still lets the
    module resolve its repo siblings (the shim's real purpose)."""
    path = REPO_ROOT / (module.replace(".", "/") + ".py")
    script = textwrap.dedent(f"""
        import importlib.util, os, sys
        sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in getattr(f, "__module__", "")]
        repo = {str(REPO_ROOT)!r}
        sys.path[:] = [p for p in sys.path if p and os.path.realpath(p) != os.path.realpath(repo)]
        spec = importlib.util.spec_from_file_location({module!r}, {str(path)!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert sys.path[-1] == repo, sys.path[-3:]
    """)
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"))
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True, cwd=str(tmp_path), env=env,
        timeout=180)
    assert result.returncode == 0, result.stderr[-3000:]
