"""Regression tests for #103056 — one-shot (-z) runs must still write log files.

``hermes -z`` (``hermes_cli/oneshot.run_oneshot``) used to call
``logging.disable(logging.CRITICAL)`` to silence the console. That sets
``Logger.manager.disable``, which blocks record creation at the root logger BEFORE
any handler runs — so the queued file handlers behind ``setup_logging()`` received
nothing either, and one-shot runs wrote neither agent.log nor errors.log.

The fix mutes only stdout/stderr-bound handlers (``mute_console_logging()``) for the
duration of the redirected run, so file logging keeps flowing while the console stays
silent.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import pytest

import hermes_logging
from hermes_cli import oneshot as oneshot_mod


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Isolate the queued file-logging state for each test in this module."""
    hermes_logging._logging_initialized = False
    hermes_logging._reset_queued_handlers()
    yield
    hermes_logging._reset_queued_handlers()
    hermes_logging._logging_initialized = False
    hermes_logging.clear_session_context()


def _fake_run_agent_logging(prompt, **kwargs):
    """Stand-in for _run_agent: emit records like an agent turn would, then succeed."""
    logging.getLogger("agent").info("oneshot-turn-info-marker")
    logging.getLogger("agent").warning("oneshot-turn-warning-marker")
    return ("done", {"model": "test-model", "provider": "test", "completed": True})


class TestOneShotKeepsFileLogging:
    def test_one_shot_writes_files_and_silences_console(self, monkeypatch, capsys):
        """`hermes -z` equivalent: files receive records, the console receives none."""
        home = Path(os.environ["HERMES_HOME"])
        hermes_logging.setup_logging(hermes_home=home)

        # A real console handler bound to stderr (as setup_verbose_logging creates).
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(console)

        monkeypatch.setattr(oneshot_mod, "_run_agent", _fake_run_agent_logging)
        captured_stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured_stdout)
        try:
            rc = oneshot_mod.run_oneshot("hello", usage_file=None)
        finally:
            logging.getLogger().removeHandler(console)
            if console.stream not in (sys.stdout, sys.stderr):
                console.close()

        assert rc == 0
        # Only the final content block reached stdout.
        assert captured_stdout.getvalue() == "done\n"
        # The console handler stayed silent for the whole run.
        assert capsys.readouterr().err == ""

        hermes_logging.flush_log_queue()
        agent_log = (home / "logs" / "agent.log").read_text()
        errors_log = (home / "logs" / "errors.log").read_text()
        assert "oneshot-turn-info-marker" in agent_log
        assert "oneshot-turn-warning-marker" in errors_log
        # Logging state is restored once the run is over: a later record reaches files.
        logging.getLogger("agent").info("post-run marker")
        hermes_logging.flush_log_queue()
        assert "post-run marker" in (home / "logs" / "agent.log").read_text()

    def test_failed_one_shot_still_writes_files(self, monkeypatch):
        """The failure path (agent raised) must log too — that is what errors.log is for."""
        home = Path(os.environ["HERMES_HOME"])
        hermes_logging.setup_logging(hermes_home=home)

        def _failing_run_agent(prompt, **kwargs):
            logging.getLogger("agent").error("oneshot-failure-marker")
            raise RuntimeError("boom")

        monkeypatch.setattr(oneshot_mod, "_run_agent", _failing_run_agent)
        rc = oneshot_mod.run_oneshot("hello", usage_file=None)
        assert rc == 1

        hermes_logging.flush_log_queue()
        assert "oneshot-failure-marker" in (home / "logs" / "errors.log").read_text()
        assert "oneshot-failure-marker" in (home / "logs" / "agent.log").read_text()

    def test_no_console_handler_is_still_silent(self, monkeypatch, capsys):
        """Default -z setup (no console handler) prints nothing while files log."""
        home = Path(os.environ["HERMES_HOME"])
        hermes_logging.setup_logging(hermes_home=home)

        monkeypatch.setattr(oneshot_mod, "_run_agent", _fake_run_agent_logging)
        captured_stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured_stdout)
        rc = oneshot_mod.run_oneshot("hello", usage_file=None)

        assert rc == 0
        assert captured_stdout.getvalue() == "done\n"
        assert capsys.readouterr().err == ""
        hermes_logging.flush_log_queue()
        assert "oneshot-turn-info-marker" in (home / "logs" / "agent.log").read_text()
