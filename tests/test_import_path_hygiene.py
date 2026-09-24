"""Checkout root must never shadow installed packages via ``sys.path``.

Import-time ``sys.path.insert(0, checkout_root)`` shims made checkout-level
modules (``evals``, ``utils``, ``tools``, ...) win over same-named installed
packages for every later import in the process. The shims are now guarded
appends: a real checkout root (``gateway/__init__.py`` marker) is appended
only when absent.

Subprocess-based: ``sys.path`` is process-global.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_ADAPTERS = ["telegram", "whatsapp", "raft", "discord", "slack"]
# Only these files can serve a bare file load: the guard precedes every repo
# import AND every later import resolves through the repo root. Elsewhere the
# guard sits below the first repo import (telegram, whatsapp, discord, base,
# cronjob_tools) or needs a flat sibling dir (slack's block_kit), so it is a
# harmless safety net, not a standalone entry point.
_STANDALONE_FILES = [
    "plugins/platforms/raft/adapter.py",
    "cron/scheduler.py",
]


def _run(script: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "hermes_home"))
    return subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120,
    )


def _shadow_probe(module: str) -> str:
    # An installed top-level package colliding with a checkout-root module
    # name (``evals``) must keep winning after the module import.
    return textwrap.dedent(f"""\
        import sys
        repo = {str(REPO_ROOT)!r}
        fake_site = {str(REPO_ROOT / "tests" / "test_import_path_hygiene_shadow")!r} + "/site"
        sys.path.insert(0, fake_site)
        sys.path.append(repo)
        import evals as victim
        assert getattr(victim, 'MARKER', None) == 'installed', victim.__file__
        before = list(sys.path)
        import importlib
        importlib.import_module({module!r})
        del sys.modules['evals']
        import evals as still
        assert getattr(still, 'MARKER', None) == 'installed', still.__file__
        assert sys.path.count(repo) <= 1, sys.path
        assert sys.path[0] == fake_site, sys.path[:3]
    """)


def test_adapter_import_never_reorders_or_duplicates(tmp_path):
    for adapter in _ADAPTERS:
        script = textwrap.dedent(f"""\
            import importlib, sys
            repo = {str(REPO_ROOT)!r}
            sys.path.append(repo)
            before = list(sys.path)
            importlib.import_module('plugins.platforms.{adapter}.adapter')
            assert sys.path == before, sys.path
        """)
        result = _run(script, tmp_path)
        assert result.returncode == 0, f"{adapter}: {result.stderr}"


def test_installed_package_wins_after_adapter_import(tmp_path):
    site = REPO_ROOT / "tests" / "test_import_path_hygiene_shadow" / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "evals.py").write_text("MARKER = 'installed'\n", encoding="utf-8")
    try:
        for adapter in _ADAPTERS:
            result = _run(
                _shadow_probe(f"plugins.platforms.{adapter}.adapter"), tmp_path)
            assert result.returncode == 0, f"{adapter}: {result.stderr}"
    finally:
        try:
            (site / "evals.py").unlink()
            site.rmdir()
            site.parent.rmdir()
        except OSError:
            pass


def test_standalone_file_load_still_resolves_repo_siblings(tmp_path):
    for rel in _STANDALONE_FILES:
        script = textwrap.dedent(f"""\
            import importlib.util, sys
            repo = {str(REPO_ROOT)!r}
            assert repo not in sys.path
            spec = importlib.util.spec_from_file_location('standalone_probe', repo + '/' + {rel!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert repo in sys.path, sys.path
        """)
        result = _run(script, tmp_path)
        assert result.returncode == 0, f"{rel}: {result.stderr}"


def test_cli_shims_are_noop_when_present(tmp_path):
    script = textwrap.dedent(f"""\
        import sys
        repo = {str(REPO_ROOT)!r}
        sys.path.append(repo)
        before = list(sys.path)
        from hermes_cli._startup_fast import ensure_project_root_on_path
        ensure_project_root_on_path()
        assert sys.path == before, sys.path
        assert sys.path.count(repo) == 1, sys.path
    """)
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stderr
