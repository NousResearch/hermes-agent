"""Structured crash frames for a Python target, stdlib only.

Runs ``TARGET`` (a Python file plus its argv) in-process and, when it raises,
prints every frame with source line and truncated locals instead of a bare
stack dump. Used by the systematic-debugging skill's "attach a real
debugger" step: the agent reads actual values at each frame rather than
sprinkling print statements.

Usage::

    python scripts/debug_trace.py path/to/target.py [args...] [--json]

Exit code mirrors the target: 0 on success, 1 when the target raised.
Cross-platform: pathlib + tempfile only, no POSIX-only primitives.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

LOCAL_REPR_LIMIT = 500
FRAME_LIMIT = 40


def _short_repr(value: object) -> str:
    try:
        text = repr(value)
    except Exception:  # health: allow BLE001 -- repr() of arbitrary target locals can raise anything; the fallback string is the point
        return "<unrepresentable>"
    if len(text) > LOCAL_REPR_LIMIT:
        return text[:LOCAL_REPR_LIMIT] + f"... <truncated {len(text) - LOCAL_REPR_LIMIT} chars>"
    return text


def _frame_record(frame, lineno) -> dict:
    """One structured frame: file, line, function, code line, truncated locals."""
    try:
        locals_map = {
            key: _short_repr(val)
            for key, val in sorted(frame.f_locals.items())
            if not key.startswith("__")
        }
        filename = os.path.relpath(frame.f_code.co_filename)
        line = None
        lines = traceback.linecache.getlines(frame.f_code.co_filename)
        if 1 <= (lineno or 0) <= len(lines):
            line = lines[lineno - 1].strip()
    except (OSError, ValueError, AttributeError):
        locals_map, filename, line = {}, frame.f_code.co_filename, None
    return {
        "file": filename,
        "line": lineno,
        "function": frame.f_code.co_name,
        "code": line,
        "locals": locals_map,
    }


def trace_crash(target: str, argv: list[str]) -> dict:
    """Run *target* with *argv* and return a structured crash report.

    Returns ``{"crashed": False}`` when the target exits cleanly, else
    ``{"crashed": True, "exception": ..., "frames": [...]}``. Frames are
    ordered outermost-first; each carries file, line, function, code line,
    and truncated locals. Only the first FRAME_LIMIT frames are kept.
    """
    saved_argv = sys.argv
    sys.argv = [target, *argv]
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code not in (None, 0):
            return {
                "crashed": True,
                "exception": f"SystemExit({code!r})",
                "frames": [],
            }
        return {"crashed": False}
    except BaseException as exc:  # health: allow BLE001 -- the tracer reports whatever the target raised, including KeyboardInterrupt; anything narrower would crash the report
        tb = exc.__traceback__
        frames = []
        while tb is not None and len(frames) < FRAME_LIMIT:
            frames.append(_frame_record(tb.tb_frame, tb.tb_lineno))
            tb = tb.tb_next
        return {
            "crashed": True,
            "exception": f"{type(exc).__name__}: {exc}",
            "frames": frames,
        }
    finally:
        sys.argv = saved_argv
    return {"crashed": False}


def format_text(report: dict, target: str) -> str:
    if not report.get("crashed"):
        return f"OK: {target} exited cleanly"
    lines = [f"CRASH in {target}: {report['exception']}"]
    for frame in report["frames"]:
        where = f"  {frame['file']}:{frame['line']} in {frame['function']}"
        lines.append(where + (f": {frame['code']}" if frame.get("code") else ""))
        for key, val in frame.get("locals", {}).items():
            lines.append(f"    {key} = {val}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Python target and dump structured crash frames.")
    parser.add_argument("target", help="Python file to run")
    parser.add_argument("args", nargs="*", help="argv passed to the target")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parsed = parser.parse_args(argv)
    if not Path(parsed.target).is_file():
        print(f"debug_trace: target not found: {parsed.target}", file=sys.stderr)
        return 2
    report = trace_crash(os.path.realpath(parsed.target), parsed.args)
    if parsed.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_text(report, parsed.target))
    return 1 if report.get("crashed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
