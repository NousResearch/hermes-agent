"""Regression test: code_skew._fingerprint must not re-execute hermes_cli.main.

Under ``python -m hermes_cli.main`` (how the systemd gateway unit launches the
process) the CLI module is registered in ``sys.modules`` as ``__main__``, NOT
as ``hermes_cli.main``. The original ``from hermes_cli.main import
_read_git_revision_fingerprint`` therefore triggered a *second* execution of
main.py, re-running its module-level ``_apply_profile_override()`` with the
already-consumed ``--profile`` flag stripped. That second pass fell back to the
sticky ``active_profile`` file and redirected ``HERMES_HOME`` into a different
profile, so the gateway tripped the duplicate-PID guard for that profile and
crash-looped.

The test asserts the module lookup happens in ``sys.modules`` first: importing
``hermes_cli.main`` would insert it into ``sys.modules``, which is the exact
side effect being guarded against.
"""

from __future__ import annotations

import sys
import types

from gateway import code_skew


def test_fingerprint_prefers_sys_modules_over_reimport(monkeypatch):
    """The reader is taken from __main__ without re-importing hermes_cli.main."""
    sentinel = object()
    seen: list[object] = []

    fake_main = types.ModuleType("__main__")

    def _read_git_revision_fingerprint(root):
        seen.append(root)
        return "deadbeef"

    fake_main._read_git_revision_fingerprint = _read_git_revision_fingerprint

    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    monkeypatch.delitem(sys.modules, "hermes_cli.main", raising=False)

    assert code_skew._fingerprint() == "deadbeef"
    assert seen == [code_skew._PROJECT_ROOT], "reader must be called with the project root"
    assert "hermes_cli.main" not in sys.modules, (
        "hermes_cli.main was imported as a side effect — that re-executes the "
        "CLI module and re-runs _apply_profile_override() (see module docstring)"
    )
    assert sentinel is not None


def test_fingerprint_uses_hermes_cli_main_when_already_imported(monkeypatch):
    """A normal (non ``-m``) launch keeps hermes_cli.main in sys.modules."""
    fake = types.ModuleType("hermes_cli.main")
    fake._read_git_revision_fingerprint = lambda root: "cafebabe"

    monkeypatch.setitem(sys.modules, "hermes_cli.main", fake)
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))

    assert code_skew._fingerprint() == "cafebabe"


def test_fingerprint_returns_none_when_reader_missing(monkeypatch):
    """No reader anywhere -> None, never raises (skew detection must no-op)."""
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    monkeypatch.delitem(sys.modules, "hermes_cli.main", raising=False)

    assert code_skew._fingerprint() is None
