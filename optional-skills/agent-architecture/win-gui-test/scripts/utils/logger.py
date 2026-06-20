"""Rotating file + console logger."""

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import TextIO


def setup_logger(
    name: str = "win-gui-test",
    log_dir: str = "",
    level: int = logging.DEBUG,
    console: bool = True,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Create or return a configured logger.

    Args:
        name: Logger name.
        log_dir: Directory for rotating log files.  Empty = no file log.
        level: Logging level.
        console: Whether to log to stderr.
        stream: Custom stream (default: sys.stderr).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove pre-existing handlers so re-config works
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with daily rotation
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = TimedRotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler(stream or sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
