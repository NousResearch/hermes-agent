"""hermes debug — diagnostic helpers for sharing agent state with maintainers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

_MAX_SHARE_BYTES = 2 * 1024 * 1024  # 2 MB default upload ceiling
_TRUNCATION_BANNER = (
    "[... log truncated — showing last {size:.1f} KB ...]\n"
)


def _read_full_log(path: Path, max_bytes: int) -> str:
    """Return the tail of *path* up to *max_bytes*, aligned to line boundaries.

    Seeks to ``size - max_bytes`` and skips any partial first line so the
    caller always receives complete lines.  Crucially, the partial-line skip
    is conditional: when the seek position lands exactly on a newline (i.e.
    the byte immediately before the seek point is ``\\n``), no line was
    split and readline() must NOT be called — doing so would silently discard
    the first complete line of the retained tail.
    """
    if not path.exists():
        return ""

    size = path.stat().st_size
    if size == 0:
        return ""

    with path.open("rb") as fh:
        if size <= max_bytes:
            return fh.read().decode("utf-8", errors="replace")

        banner = _TRUNCATION_BANNER.format(size=max_bytes / 1024)
        seek_pos = size - max_bytes
        fh.seek(seek_pos)

        # Only skip the first (potentially partial) line when we landed in
        # the middle of one.  If the byte just before our seek position is
        # a newline, we are already at a clean line boundary.
        if seek_pos > 0:
            fh.seek(seek_pos - 1)
            preceding_byte = fh.read(1)
            if preceding_byte != b"\n":
                # We cut a line — skip the remainder of this partial line.
                fh.readline()

        tail = fh.read().decode("utf-8", errors="replace")

    return banner + tail


def _resolve_log_path(log_name: str) -> Optional[Path]:
    log_dir = get_hermes_home() / "logs"
    candidates = {
        "agent": log_dir / "agent.log",
        "errors": log_dir / "errors.log",
        "gateway": log_dir / "gateway.log",
    }
    return candidates.get(log_name.lower())


def cmd_debug_share(args) -> None:
    """Print log content suitable for pasting into a bug report."""
    log_name = getattr(args, "log", "agent")
    max_bytes = getattr(args, "max_bytes", None) or _MAX_SHARE_BYTES

    path = _resolve_log_path(log_name)
    if path is None:
        print(f"Unknown log '{log_name}'. Choose: agent, errors, gateway", file=sys.stderr)
        sys.exit(1)

    content = _read_full_log(path, max_bytes)
    if not content:
        print(f"Log is empty or does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    print(content, end="")


def register_debug_parser(subparsers) -> None:
    """Register the ``hermes debug`` sub-command tree."""
    debug_parser = subparsers.add_parser(
        "debug",
        help="Diagnostic helpers",
        description="Helpers for capturing agent state for bug reports",
    )
    debug_sub = debug_parser.add_subparsers(dest="debug_action")

    share_parser = debug_sub.add_parser(
        "share",
        help="Print log content for sharing in a bug report",
        description=(
            "Print the tail of a log file (agent / errors / gateway) to stdout.\n"
            "Pipe to a paste tool or redirect to a file for easy sharing.\n\n"
            "Examples:\n"
            "  hermes debug share                     # print agent.log tail\n"
            "  hermes debug share errors              # print errors.log tail\n"
            "  hermes debug share --max-kb 512        # limit to 512 KB\n"
        ),
    )
    share_parser.add_argument(
        "log", nargs="?", default="agent",
        help="Which log to share: agent (default), errors, gateway",
    )
    share_parser.add_argument(
        "--max-kb", dest="max_bytes", type=lambda v: int(v) * 1024,
        metavar="KB", default=None,
        help="Maximum bytes to include (default: 2048 KB)",
    )
    share_parser.set_defaults(func=cmd_debug_share)

    def _debug_default(args):
        debug_parser.print_help()

    debug_parser.set_defaults(func=_debug_default)
