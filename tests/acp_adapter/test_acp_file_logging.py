"""``hermes acp`` must keep writing agent.log / errors.log.

``hermes_cli.main`` runs ``setup_logging()`` at import, which puts the Hermes queue
handler on the root logger. ACP's ``_setup_logging`` then swaps the console handlers
for its stderr one; it used to ``clear()`` every root handler, and the later
``setup_logging()`` calls are no-ops for already-registered files, so nothing ever
reached the log files for the whole ACP session.
"""

import logging

import hermes_logging
from acp_adapter.entry import _setup_logging


def test_file_logging_survives_acp_logging_setup():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    hermes_logging._reset_queued_handlers()
    hermes_logging._logging_initialized = False
    try:
        log_dir = hermes_logging.setup_logging(mode="cli")  # hermes_cli.main at import
        _setup_logging()  # acp_adapter.entry.main()
        hermes_logging.setup_logging()  # agent init, later in the session

        logging.getLogger("acp_adapter.server").warning("acp-file-logging-probe")
        hermes_logging.flush_log_queue()

        for name in ("agent.log", "errors.log"):
            assert "acp-file-logging-probe" in (log_dir / name).read_text(encoding="utf-8"), name
    finally:
        hermes_logging._reset_queued_handlers()
        hermes_logging._logging_initialized = False
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
