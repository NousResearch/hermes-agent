"""The test suite must never write into the operator's real Hermes logs.

`hermes_cli/main.py` calls `setup_logging()` at module scope, which resolves
`get_hermes_home()` and attaches rotating file handlers to the ROOT logger.
Importing it - which many test modules do, directly or transitively - wires
the whole pytest session's logging to `<HERMES_HOME>/logs/agent.log`.

If HERMES_HOME is not already sandboxed at that moment, that is the
operator's real log. Measured on a live install, 126 warnings in a personal
`agent.log` came from test runs rather than the running gateway: phantom
`FakeTree` Discord failures and `rejected invalid API key` entries from
`test_api_server_runs.py`. Noise like that makes genuine warnings hard to
find precisely when someone is debugging.

The per-test env fixture cannot close this: fixtures run after collection has
imported the test modules, and by then the handler holds an absolute path.
`tests/conftest.py` sets HERMES_HOME at module scope for that reason - this
guards the property so a refactor cannot quietly undo it.
"""

import logging
import os
from pathlib import Path

import pytest


def _real_hermes_home() -> Path:
    """Where the operator's logs live, ignoring any test sandboxing."""
    return Path.home() / ".hermes"


def _all_file_destinations() -> list[str]:
    """Every file path the root logger can reach, including via a QueueHandler.

    Logging is routed through a queue, so the file handlers hang off the
    listener rather than the root logger - checking `root.handlers` alone
    reports nothing and looks falsely clean.
    """
    seen: list[str] = []

    def collect(handlers) -> None:
        for handler in handlers or ():
            path = getattr(handler, "baseFilename", None)
            if path:
                seen.append(str(path))
            listener = getattr(handler, "listener", None)
            if listener is not None:
                collect(getattr(listener, "handlers", ()))

    collect(logging.getLogger().handlers)

    try:
        import hermes_logging

        listener = getattr(hermes_logging, "_queue_listener", None)
        if listener is not None:
            collect(getattr(listener, "handlers", ()))
    except Exception:
        pass

    return seen


class TestLogIsolation:
    def test_hermes_home_is_sandboxed_before_imports(self):
        # Deliberately NOT os.environ: by test time the per-test `_isolate_env`
        # fixture has sandboxed HERMES_HOME, so reading it here would pass even
        # with the conftest block deleted. Assert the value captured at conftest
        # import, which is the moment that actually matters.
        from tests.conftest import HERMES_HOME_AT_CONFTEST_IMPORT as home

        assert home, "conftest must set HERMES_HOME before test modules import"
        assert Path(home).resolve() != _real_hermes_home().resolve(), (
            f"HERMES_HOME pointed at the operator's real home ({home}) when "
            "conftest loaded; import-time setup_logging() writes to their agent.log"
        )

    def test_importing_the_cli_does_not_target_the_real_logs(self):
        pytest.importorskip("hermes_cli.main")

        real_logs = str(_real_hermes_home() / "logs")
        offenders = [p for p in _all_file_destinations() if p.startswith(real_logs)]

        assert offenders == [], (
            "the test session is writing into the operator's real Hermes logs:\n  "
            + "\n  ".join(offenders)
        )
