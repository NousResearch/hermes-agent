"""Buffered provider-plugin discovery load failures stay off raw stderr.

Provider discovery runs before ``setup_logging()``, when ``logger.warning``
falls through to stdlib ``logging.lastResort`` and writes raw stderr — which
corrupts the fullscreen prompt_toolkit TUI. Load failures are therefore
buffered (bounded, debug-logged) and replayed once TUI startup output is
safe (see ``CLITuiRuntimeMixin._tui_startup_prewarm_and_warnings``).
"""

from __future__ import annotations

import logging
import sys

import pytest

import providers as _providers_pkg
from providers import (
    ProviderLoadFailure,
    format_provider_load_failures,
    get_provider_load_failures,
)

BROKEN_PLUGIN_NAME = "broken_test_plugin_zz"


@pytest.fixture(autouse=True)
def _clean_load_failure_buffer():
    with _providers_pkg._LOAD_FAILURES_LOCK:
        _providers_pkg._LOAD_FAILURES.clear()
    yield
    with _providers_pkg._LOAD_FAILURES_LOCK:
        _providers_pkg._LOAD_FAILURES.clear()
    sys.modules.pop(f"plugins.model_providers.{BROKEN_PLUGIN_NAME}", None)


def _write_broken_plugin(tmp_path):
    plugin_dir = tmp_path / BROKEN_PLUGIN_NAME
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("raise ImportError('boom-test-marker')\n")
    return plugin_dir


def test_broken_plugin_discovery_stays_off_stderr(tmp_path, capsys, caplog):
    """A plugin whose __init__ raises must not touch stderr nor log WARNING."""
    from providers import _import_plugin_dir

    plugin_dir = _write_broken_plugin(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="providers"):
        _import_plugin_dir(plugin_dir, "bundled")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []

    failures = get_provider_load_failures()
    assert len(failures) == 1
    failure = failures[0]
    assert failure.plugin_name == BROKEN_PLUGIN_NAME
    assert failure.source == "bundled"
    assert "boom-test-marker" in failure.error


def test_format_empty_failures():
    assert format_provider_load_failures(()) == []
    assert format_provider_load_failures([], limit=3) == []


def test_format_exact_limit_has_no_overflow_line():
    failures = tuple(
        ProviderLoadFailure(plugin_name=f"plugin-{i}", source="bundled", error=f"err-{i}")
        for i in range(3)
    )
    lines = format_provider_load_failures(failures)
    assert len(lines) == 3
    assert "plugin-0" in lines[0] and "err-0" in lines[0]
    assert "plugin-2" in lines[2] and "err-2" in lines[2]


def test_format_truncates_with_overflow_line():
    failures = tuple(
        ProviderLoadFailure(plugin_name=f"plugin-{i}", source="bundled", error=f"err-{i}")
        for i in range(5)
    )
    lines = format_provider_load_failures(failures)
    assert len(lines) == 4  # 3 failure lines + 1 overflow line
    assert "plugin-0" in lines[0]
    assert "plugin-2" in lines[2]
    assert "plugin-3" not in "".join(lines)
    assert "2" in lines[3]
    assert "agent.log" in lines[3]

    lines = format_provider_load_failures(failures, limit=2)
    assert len(lines) == 3
    assert "plugin-1" in lines[1]
    assert "3" in lines[2]


class _StubCLI:
    """Only what the replay path touches: the console seam plus config."""

    def __init__(self):
        self.printed: list[str] = []
        self.config = {}

    def _console_print(self, *args, **kwargs):
        self.printed.append(" ".join(str(a) for a in args))


def _run_startup_warnings(stub, monkeypatch):
    from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin

    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    monkeypatch.setattr(
        "agent.onboarding.detect_openclaw_residue", lambda *a, **k: False
    )
    CLITuiRuntimeMixin._tui_startup_prewarm_and_warnings(stub, None)


def test_tui_startup_replays_buffered_failures(monkeypatch):
    _providers_pkg._record_plugin_failure(
        "replay-plugin-zz", "bundled", RuntimeError("kaput-marker")
    )
    stub = _StubCLI()
    _run_startup_warnings(stub, monkeypatch)

    replay = [line for line in stub.printed if "replay-plugin-zz" in line]
    assert len(replay) == 1
    assert "kaput-marker" in replay[0]
    assert "yellow" in replay[0]


def test_tui_startup_quiet_with_empty_buffer(monkeypatch):
    assert get_provider_load_failures() == ()
    stub = _StubCLI()
    _run_startup_warnings(stub, monkeypatch)
    assert stub.printed == []
