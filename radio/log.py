"""Radio-specific logging.

Writes structured, human-readable logs to ~/.hermes/radio/radio.log.
Separate from the main hermes log so radio activity doesn't drown
in tool/agent noise.

Format:
  16:08:18  PLAY   Mamman Sani -- Bodo  [1980s NER weird]
  16:08:22  LEVEL  ffmpeg meter started
  16:12:09  SKIP   user skip
  16:12:10  PLAY   Black Sugar -- Viajecito  [1970s PER weird]
  16:15:30  MIC    "That was Black Sugar from Lima..."
  16:15:45  END    track finished naturally
  16:20:00  STOP   radio stopped
"""

import logging
import os
from datetime import datetime
from pathlib import Path

RADIO_LOG_DIR = Path(os.path.expanduser("~/.hermes/radio"))
RADIO_LOG_FILE = RADIO_LOG_DIR / "radio.log"

_logger = None


def _get_logger() -> logging.Logger:
    """Get or create the radio file logger."""
    global _logger
    if _logger is not None:
        return _logger

    RADIO_LOG_DIR.mkdir(parents=True, exist_ok=True)

    _logger = logging.getLogger("hermes.radio")
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False  # don't spam the root logger

    # File handler with simple format
    handler = logging.FileHandler(RADIO_LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(handler)

    return _logger


def play(artist: str, title: str, decade: int = 0, country: str = "", mood: str = "", source: str = "crate"):
    tag = f"[{decade}s {country} {mood}]" if decade else f"[{source}]"
    _get_logger().info("PLAY   %s -- %s  %s", artist, title, tag)


def stream(station: str, url: str = ""):
    _get_logger().info("STREAM %s  %s", station, url[:60])


def skip(reason: str = "user skip"):
    _get_logger().info("SKIP   %s", reason)


def stop(reason: str = "user stop"):
    _get_logger().info("STOP   %s", reason)


def pause(paused: bool):
    _get_logger().info("PAUSE  %s", "paused" if paused else "resumed")


def mic(text: str):
    short = text[:80] + "..." if len(text) > 80 else text
    _get_logger().info("MIC    \"%s\"", short)


def end(reason: str = "eof"):
    _get_logger().info("END    %s", reason)


def error(msg: str):
    _get_logger().error("ERROR  %s", msg)


def dig(decade: int, country: str, mood: str):
    _get_logger().info("DIG    %ds %s %s", decade, country, mood)


def meter(status: str):
    _get_logger().debug("METER  %s", status)


def info(msg: str):
    _get_logger().info("INFO   %s", msg)
