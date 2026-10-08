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
        _providers_pkg._REPLAYED_COUNT = 0
    yield
    with _providers_pkg._LOAD_FAILURES_LOCK:
        _providers_pkg._LOAD_FAILURES.clear()
        _providers_pkg._REPLAYED_COUNT = 0
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


def test_replay_logs_buffered_failures_as_warnings_once(caplog):
    """Non-TUI commands surface pre-logging failures via replay into the logger."""
    from providers import replay_provider_load_failures

    _providers_pkg._record_plugin_failure(
        "replay-file-zz", "bundled", RuntimeError("file-marker-zz")
    )
    with caplog.at_level(logging.WARNING, logger="providers"):
        assert replay_provider_load_failures() == 1
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "replay-file-zz" in r.getMessage() and "file-marker-zz" in r.getMessage()
        for r in warnings
    )

    # Idempotent: a second replay must not double-report.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="providers"):
        assert replay_provider_load_failures() == 0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_replay_reaches_log_file(tmp_path):
    """A failure recorded pre-logging lands in agent.log once replay runs."""
    import logging as _logging

    from providers import replay_provider_load_failures

    _providers_pkg._record_plugin_failure(
        "logfile-zz", "bundled", RuntimeError("logfile-marker-zz")
    )
    log_file = tmp_path / "agent.log"
    handler = _logging.FileHandler(log_file)
    handler.setLevel(_logging.INFO)
    pkg_logger = _logging.getLogger("providers")
    pkg_logger.addHandler(handler)
    try:
        replay_provider_load_failures()
        handler.flush()
    finally:
        pkg_logger.removeHandler(handler)
        handler.close()
    content = log_file.read_text(encoding="utf-8")
    assert "logfile-zz" in content
    assert "logfile-marker-zz" in content


def test_host_isolation_refusal_is_buffered_with_remedy(monkeypatch):
    """An enabled entry point refused under host isolation must buffer, not stay silent."""
    import types
    import importlib.metadata as _md

    from providers import _discover_entry_point_providers

    refusal_text = (
        "pip-installed model-provider plugin 'refused-plugin-zz' runs only in-process, "
        "and plugins.isolation is 'host' (third-party plugin code never runs inside the "
        "Hermes process); set plugins.isolation: in_process to load it"
    )
    fake_ep = types.SimpleNamespace(name="refused-plugin-zz")
    fake_ep.load = lambda: (_ for _ in ()).throw(
        AssertionError("refused plugin must not load")
    )

    class _FakeEps:
        def select(self, *, group):
            assert group == "hermes_agent.plugins"
            return [fake_ep]

    monkeypatch.setattr(_md, "entry_points", lambda: _FakeEps())
    monkeypatch.setattr(
        "hermes_cli.plugins._get_enabled_plugins", lambda: {"refused-plugin-zz"}
    )
    monkeypatch.setattr("hermes_cli.plugins._get_disabled_plugins", lambda: set())
    monkeypatch.setattr(
        "hermes_cli.plugin_isolation.in_process_import_refusal",
        lambda what: refusal_text,
    )

    _discover_entry_point_providers()

    failures = get_provider_load_failures()
    assert len(failures) == 1
    assert failures[0].plugin_name == "refused-plugin-zz"
    assert failures[0].source == "entry-point"
    assert "plugins.isolation" in failures[0].error
