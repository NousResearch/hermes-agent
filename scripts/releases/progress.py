"""Progress logging for release automation.

Everything goes to stderr, flushed, so stdout stays the machine-readable
result (the sequencer prints its steps as JSON there). On GitHub Actions the
``::group::`` / ``::notice::`` lines render as collapsible sections and
annotations; locally they read as plain prefixed lines.
"""
from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Callable, Iterator

# Long argv values (release bodies, JSON payloads) are cut in command logs.
_ARG_LIMIT = 160


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def notice(message: str, *, title: str | None = None) -> None:
    log(f"::notice{f' title={title}' if title else ''}::{message}")


def error(message: str, *, title: str | None = None) -> None:
    log(f"::error{f' title={title}' if title else ''}::{message}")


@contextlib.contextmanager
def group(title: str) -> Iterator[None]:
    """A collapsible section that reports how long it took and whether it failed."""
    log(f"::group::{title}")
    started = time.monotonic()
    try:
        yield
    except BaseException as exc:
        log(f"✗ {title} failed after {time.monotonic() - started:.1f}s: {type(exc).__name__}: {exc}")
        log("::endgroup::")
        raise
    log(f"✓ {title} ({time.monotonic() - started:.1f}s)")
    log("::endgroup::")


def describe_command(argv: list[str]) -> str:
    shown = []
    for arg in argv:
        text = str(arg)
        if len(text) > _ARG_LIMIT:
            text = f"{text[:_ARG_LIMIT]}…(+{len(text) - _ARG_LIMIT} chars)"
        shown.append(text.replace("\n", "\\n"))
    return " ".join(shown)


def logged_run(run: Callable[[list[str]], str]) -> Callable[[list[str]], str]:
    """Wrap a ``run(argv) -> str`` so every command logs, with its time and outcome."""
    if getattr(run, "_progress_logged", False):
        return run

    def wrapper(argv: list[str]) -> str:
        log(f"  $ {describe_command(argv)}")
        started = time.monotonic()
        try:
            result = run(argv)
        except BaseException as exc:
            log(f"    ↳ failed after {time.monotonic() - started:.1f}s: {type(exc).__name__}: {exc}")
            raise
        lines = result.count("\n") + 1 if result else 0
        log(f"    ↳ ok in {time.monotonic() - started:.1f}s ({lines} line{'s' if lines != 1 else ''})")
        return result

    wrapper._progress_logged = True  # type: ignore[attr-defined]
    return wrapper
