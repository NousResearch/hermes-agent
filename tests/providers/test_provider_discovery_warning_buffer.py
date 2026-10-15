"""Buffered provider-plugin discovery load failures stay off raw stderr.

Provider discovery runs before ``setup_logging()``, when ``logger.warning``
falls through to stdlib ``logging.lastResort`` and writes raw stderr — which
corrupts the fullscreen prompt_toolkit TUI. Load failures are therefore
buffered (bounded, debug-logged) and replayed once logging is ready
(``mark_logging_ready``) and once on the console when TUI startup output is
safe (see ``CLITuiRuntimeMixin._tui_startup_prewarm_and_warnings``).
"""

from __future__ import annotations

import logging
import subprocess
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
        _providers_pkg._CONSOLE_SHOWN_COUNT = 0
        _providers_pkg._LOGGING_READY = False
    _saved_discovered = _providers_pkg._discovered
    _providers_pkg._discovered = True
    yield
    with _providers_pkg._LOAD_FAILURES_LOCK:
        _providers_pkg._LOAD_FAILURES.clear()
        _providers_pkg._REPLAYED_COUNT = 0
        _providers_pkg._CONSOLE_SHOWN_COUNT = 0
        _providers_pkg._LOGGING_READY = False
    _providers_pkg._discovered = _saved_discovered
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
        ProviderLoadFailure(
            plugin_name=f"plugin-{i}", source="bundled", error=f"err-{i}"
        )
        for i in range(3)
    )
    lines = format_provider_load_failures(failures)
    assert len(lines) == 3
    assert "plugin-0" in lines[0] and "err-0" in lines[0]
    assert "plugin-2" in lines[2] and "err-2" in lines[2]


def test_format_truncates_with_overflow_line():
    failures = tuple(
        ProviderLoadFailure(
            plugin_name=f"plugin-{i}", source="bundled", error=f"err-{i}"
        )
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


def test_tui_startup_prints_each_failure_once(monkeypatch):
    _providers_pkg._record_plugin_failure("once-plugin-zz", "bundled", RuntimeError("once-marker"))
    first, second = _StubCLI(), _StubCLI()
    _run_startup_warnings(first, monkeypatch)
    _run_startup_warnings(second, monkeypatch)
    assert sum("once-plugin-zz" in line for line in first.printed) == 1
    assert not any("once-plugin-zz" in line for line in second.printed)


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


def test_second_replay_after_discovery_surfaces_late_failure(monkeypatch, caplog):
    """A failure buffered by lazy discovery AFTER an early replay still reaches the log.

    The early flush runs before lazy provider discovery, so a late-buffered
    failure needs a later flush: the guarded replay at the end of
    _discover_providers().
    """
    import logging as _logging

    from providers import mark_logging_ready, replay_provider_load_failures

    def _late_discovery_steps():
        _providers_pkg._record_plugin_failure(
            "late-discovery-zz", "bundled", RuntimeError("late-discovery-marker-zz")
        )

    monkeypatch.setattr(_providers_pkg, "_run_discovery_steps", _late_discovery_steps)
    monkeypatch.setattr(_providers_pkg, "_sync_auth_registry", lambda: None)
    _providers_pkg._discovered = False

    with caplog.at_level(_logging.WARNING, logger="providers"):
        assert mark_logging_ready() == 0
        _providers_pkg._discover_providers()
    assert any(
        "late-discovery-zz" in r.getMessage()
        and "late-discovery-marker-zz" in r.getMessage()
        for r in caplog.records
        if r.levelno >= _logging.WARNING
    )

    # Exactly once: a further replay must not double-report.
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="providers"):
        assert replay_provider_load_failures() == 0
    assert [r for r in caplog.records if r.levelno >= _logging.WARNING] == []


def test_replay_cursor_survives_max_rollover(caplog):
    """Filling the 50-slot buffer, replaying, then recording more still replays the new items.

    The replay cursor is an index into a bounded list, so eviction must move it
    back with the window, or later replays return 0 forever.
    """
    import logging as _logging

    from providers import MAX_LOAD_FAILURES, replay_provider_load_failures

    assert MAX_LOAD_FAILURES == 50
    for i in range(MAX_LOAD_FAILURES):
        _providers_pkg._record_plugin_failure(
            f"rollover-zz-{i}", "bundled", RuntimeError(f"rollover-marker-{i}")
        )
    with caplog.at_level(_logging.WARNING, logger="providers"):
        assert replay_provider_load_failures() == MAX_LOAD_FAILURES

    _providers_pkg._record_plugin_failure(
        "rollover-new-zz", "bundled", RuntimeError("rollover-new-marker-zz")
    )
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="providers"):
        assert replay_provider_load_failures() == 1
    assert any(
        "rollover-new-zz" in r.getMessage()
        and "rollover-new-marker-zz" in r.getMessage()
        for r in caplog.records
        if r.levelno >= _logging.WARNING
    )

    # Multi-eviction variant: N records past MAX surface exactly once.
    for i in range(3):
        _providers_pkg._record_plugin_failure(
            f"rollover-multi-zz-{i}",
            "bundled",
            RuntimeError(f"rollover-multi-marker-{i}"),
        )
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="providers"):
        assert replay_provider_load_failures() == 3
    warned = [r.getMessage() for r in caplog.records if r.levelno >= _logging.WARNING]
    for i in range(3):
        assert any(
            f"rollover-multi-zz-{i}" in m and f"rollover-multi-marker-{i}" in m
            for m in warned
        )
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="providers"):
        assert replay_provider_load_failures() == 0


def _property_warned_names(caplog):
    import logging as _logging

    return [r.getMessage() for r in caplog.records if r.levelno >= _logging.WARNING]


def _property_assert_cursor_bounded():
    assert 0 <= _providers_pkg._REPLAYED_COUNT <= len(_providers_pkg._LOAD_FAILURES)
    assert len(_providers_pkg._LOAD_FAILURES) <= _providers_pkg.MAX_LOAD_FAILURES


def test_property_buffer_overflow_keeps_newest_and_replays_once(caplog):
    """Randomized un-replayed overflow: survivors replay exactly once, cursor bounded.

    Piles records past MAX before any replay (the rollover test always replays
    in between). Fixed seed.
    """
    import logging as _logging
    import random as _random

    from providers import MAX_LOAD_FAILURES, replay_provider_load_failures

    rng = _random.Random(13423501)
    recorded: list[str] = []
    tag = 0
    with caplog.at_level(_logging.WARNING, logger="providers"):
        for _ in range(30):
            for _ in range(rng.randint(0, MAX_LOAD_FAILURES + 20)):
                name = f"prop-buf-zz-{tag}"
                _providers_pkg._record_plugin_failure(
                    name, "bundled", RuntimeError(f"prop-buf-marker-{tag}")
                )
                recorded.append(name)
                tag += 1
            _property_assert_cursor_bounded()

        survivors = recorded[-MAX_LOAD_FAILURES:] if recorded else []
        assert [f.plugin_name for f in get_provider_load_failures()] == survivors

        caplog.clear()
        assert replay_provider_load_failures() == len(survivors)
        warned = _property_warned_names(caplog)
        assert len(warned) == len(survivors)
        for name in survivors:
            assert sum(f"'{name}'" in m for m in warned) == 1
        _property_assert_cursor_bounded()

        caplog.clear()
        with caplog.at_level(_logging.WARNING, logger="providers"):
            assert replay_provider_load_failures() == 0
        assert _property_warned_names(caplog) == []


def test_property_late_arrivals_surface_exactly_once(caplog):
    """Randomized late-arrival timing: post-replay records surface on next replay, once.

    Each trial records a random pre-batch, replays, records a random late
    batch, and proves the late batch (and only it) arrives on the second
    replay. Trial totals stay under MAX so overflow cannot mask a timing
    drop. Fixed seed 13423503.
    """
    import logging as _logging
    import random as _random

    from providers import MAX_LOAD_FAILURES, replay_provider_load_failures

    rng = _random.Random(13423503)
    tag = 0
    with caplog.at_level(_logging.WARNING, logger="providers"):
        for _ in range(20):
            n_pre = rng.randint(0, 5)
            for _ in range(n_pre):
                _providers_pkg._record_plugin_failure(
                    f"prop-late-pre-zz-{tag}",
                    "bundled",
                    RuntimeError(f"prop-late-pre-marker-{tag}"),
                )
                tag += 1
            assert replay_provider_load_failures() == n_pre
            _property_assert_cursor_bounded()

            n_late = rng.randint(1, 8)
            assert n_pre + n_late <= MAX_LOAD_FAILURES
            late = []
            for _ in range(n_late):
                name = f"prop-late-zz-{tag}"
                _providers_pkg._record_plugin_failure(
                    name, "bundled", RuntimeError(f"prop-late-marker-{tag}")
                )
                late.append(name)
                tag += 1
            _property_assert_cursor_bounded()

            caplog.clear()
            assert replay_provider_load_failures() == n_late
            batch = _property_warned_names(caplog)
            assert len(batch) == n_late
            for name in late:
                assert sum(f"'{name}'" in m for m in batch) == 1

            caplog.clear()
            with caplog.at_level(_logging.WARNING, logger="providers"):
                assert replay_provider_load_failures() == 0
            assert _property_warned_names(caplog) == []
            _property_assert_cursor_bounded()


_SETUP_FAILURE_MARKER = "mozi-setup-fail-7c41"
_SETUP_FAILURE_ERRMARK = "mozi-setup-err-7c41"

_SETUP_FAILURE_PROBE = (
    "import sys\n"
    "import providers as _p\n"
    f"_p._record_plugin_failure({_SETUP_FAILURE_MARKER!r}, 'bundled', RuntimeError({_SETUP_FAILURE_ERRMARK!r}))\n"
    "import hermes_logging\n"
    "def _boom(*a, **k):\n"
    "    raise RuntimeError('simulated setup_logging failure')\n"
    "hermes_logging.setup_logging = _boom\n"
    "import hermes_cli.main\n"
    "import logging as _pl\n"
    "assert len(_pl.getLogger().handlers) == 0, 'precondition: root has handlers'\n"
    "assert not _p._LOGGING_READY, 'logging must not be marked ready after a setup failure'\n"
    "_p._replay_if_logging_ready()\n"
    "assert len(_p.get_provider_load_failures()) >= 1, 'failures must stay buffered'\n"
    "sys.stdout.write('PROBE_OK')\n"
)


def test_setup_failure_keeps_failures_buffered_and_stderr_clean():
    """setup_logging raising must not leak buffered failures to raw stderr.

    With a failure buffered and ``setup_logging()`` forced to raise, importing
    ``hermes_cli.main`` must leave logging unmarked, and the late-discovery
    flush must stay a no-op (a warning with zero handlers would fall through to
    ``logging.lastResort`` raw stderr).
    """
    proc = subprocess.run(
        [sys.executable, "-c", _SETUP_FAILURE_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"probe failed: {proc.stderr[-2000:]}"
    assert proc.stdout == "PROBE_OK", f"probe output: {proc.stdout!r} {proc.stderr[-2000:]!r}"
    assert proc.stderr == "", f"probe polluted stderr: {proc.stderr[:500]!r}"


def test_mark_logging_ready_flushes_and_opens_late_flush(caplog):
    from providers import mark_logging_ready

    _providers_pkg._record_plugin_failure("early-zz", "bundled", RuntimeError("early-marker"))
    assert _providers_pkg._replay_if_logging_ready() == 0  # not ready: stays buffered
    with caplog.at_level(logging.WARNING, logger="providers"):
        assert mark_logging_ready() == 1
        _providers_pkg._record_plugin_failure("late-zz", "bundled", RuntimeError("late-marker"))
        assert _providers_pkg._replay_if_logging_ready() == 1
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("early-zz" in m for m in messages) and any("late-zz" in m for m in messages)


def test_identical_failure_is_buffered_once():
    for _ in range(3):
        _providers_pkg._record_plugin_failure("dup-zz", "user", RuntimeError("same"))
    assert len(get_provider_load_failures()) == 1
    _providers_pkg._record_plugin_failure("dup-zz", "user", RuntimeError("different"))
    assert len(get_provider_load_failures()) == 2


def test_failure_records_are_immutable():
    _providers_pkg._record_plugin_failure("frozen-zz", "bundled", RuntimeError("x"))
    with pytest.raises(Exception):
        get_provider_load_failures()[0].error = "changed"


def test_console_drain_shows_each_failure_once():
    from providers import drain_console_failures

    _providers_pkg._record_plugin_failure("c1-zz", "bundled", RuntimeError("a"))
    assert [f.plugin_name for f in drain_console_failures()] == ["c1-zz"]
    assert drain_console_failures() == ()
    _providers_pkg._record_plugin_failure("c2-zz", "bundled", RuntimeError("b"))
    assert [f.plugin_name for f in drain_console_failures()] == ["c2-zz"]


def test_console_line_keeps_first_line_and_caps_length():
    long_error = "first line " + "x" * 400 + "\nsecond line"
    failure = ProviderLoadFailure(plugin_name="long-zz", source="bundled", error=long_error)
    (line,) = format_provider_load_failures([failure])
    assert "second line" not in line
    assert len(line) < 400 and line.endswith("…")


def test_failures_evicted_before_any_replay_are_still_logged_when_ready(caplog):
    from providers import MAX_LOAD_FAILURES, mark_logging_ready

    mark_logging_ready()
    with caplog.at_level(logging.WARNING, logger="providers"):
        for i in range(MAX_LOAD_FAILURES + 2):
            _providers_pkg._record_plugin_failure(f"evict-zz-{i}", "bundled", RuntimeError("e"))
    # Not replayed yet: the 2 oldest were evicted and logged on the way out.
    names = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("evict-zz-0" in m for m in names) and any("evict-zz-1" in m for m in names)


def test_setup_logging_announces_readiness_to_the_provider_registry(monkeypatch):
    import hermes_logging

    calls = []
    monkeypatch.setattr(_providers_pkg, "mark_logging_ready", lambda: calls.append(1) or 0)
    hermes_logging._announce_logging_ready()
    assert calls == [1]
