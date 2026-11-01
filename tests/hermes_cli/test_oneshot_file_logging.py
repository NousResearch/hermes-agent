"""``hermes -z`` keeps log records off the terminal, but agent.log / errors.log still get them.

The console used to be muted with ``logging.disable(logging.CRITICAL)``; that gate sits in
``Logger.isEnabledFor`` ahead of every handler, so a failing one-shot left no WARNING or
ERROR in the log files either.
"""

import io
import logging
from unittest import mock

import hermes_cli.oneshot as oneshot
import hermes_logging


def test_oneshot_logs_reach_the_files_but_not_the_console():
    root = logging.getLogger()
    saved_level = root.level
    console = io.StringIO()
    console_handler = logging.StreamHandler(console)  # stands in for e.g. the ``-v`` stderr handler
    hermes_logging._reset_queued_handlers()
    hermes_logging._logging_initialized = False

    def fake_run_agent(*_args, **_kwargs):
        logging.getLogger("run_agent").warning("oneshot-warning-probe")
        logging.getLogger("run_agent").error("oneshot-error-probe")
        return "done", {"final_response": "done", "completed": True, "failed": False}

    try:
        log_dir = hermes_logging.setup_logging(mode="cli")  # hermes_cli.main at import
        root.addHandler(console_handler)
        with mock.patch.object(oneshot, "_run_agent", side_effect=fake_run_agent):
            assert oneshot.run_oneshot("q") == 0
        hermes_logging.flush_log_queue()

        for name in ("agent.log", "errors.log"):
            text = (log_dir / name).read_text(encoding="utf-8")
            assert "oneshot-warning-probe" in text and "oneshot-error-probe" in text, name
        assert console.getvalue() == ""
    finally:
        root.removeHandler(console_handler)
        hermes_logging._reset_queued_handlers()
        hermes_logging._logging_initialized = False
        root.setLevel(saved_level)
