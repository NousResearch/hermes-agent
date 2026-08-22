"""Cross-thread ANSI printing helper shared between the CLI and the banner.

Originally ``cli.py`` carried a private ``_cprint`` whose docstring captured
the exact failure mode this module exists to fix:

  When called from a background thread while a prompt_toolkit
  ``Application`` is running (the common case for the self-improvement
  background review's ``💾 …`` summary, curator summaries, and other
  bg-thread emissions), a direct ``_pt_print`` races with the input
  area's redraw and the line can end up visually buried behind the
  prompt.  Route those cases through ``run_in_terminal`` via
  ``loop.call_soon_threadsafe``, which pauses the input area, prints
  the line above it, and redraws the prompt cleanly.

``hermes_cli/banner.py`` needed the same routing for ``_defer_update_notice``
(issue #83969) but importing ``cli.py`` from ``hermes_cli/`` would create
a circular dependency (``cli.py`` imports banner.py). This module is the
shared sink both call sites reach for. Keep it lean: only depends on
``prompt_toolkit``, never on cli.py, banner.py, or anything that owns a
prompt_toolkit ``Application`` instance.
"""
from __future__ import annotations

import asyncio as _asyncio
from typing import Any

try:
    from prompt_toolkit import print_formatted_text as _pt_print
    from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
except Exception:  # pragma: no cover — prompt_toolkit is a hard dep
    _pt_print = None  # type: ignore[assignment]
    _PT_ANSI = None  # type: ignore[assignment]


def cprint(text: str) -> None:
    """Print ANSI-colored text safely from any thread.

    Three branches, in order:

    1. No prompt_toolkit at all (CI, subprocess worker, plain ``print``
       fallback) — degrade to ``print(text)`` and don't crash the caller.
    2. prompt_toolkit available but no active ``Application`` (banner
       render path, plain CLI mode) — direct ``print_formatted_text`` is
       safe and matches the behavior of spinners / streamed chunks.
    3. Active ``Application`` AND we are *not* on its event-loop thread —
       schedule ``run_in_terminal`` via ``loop.call_soon_threadsafe`` so
       the line prints above the prompt instead of getting buried under
       it. Cross-thread background emissions are the canonical case.

    Branch 3 is the fix for the "deferred update notice doesn't show up
    at all" variant of issue #83969 — the first attempt routed through
    ``prompt_toolkit.print_formatted_text`` directly from the deferred
    background thread, which races the prompt redraw and visually loses
    the line (the "third form" of the bug).
    """
    if _pt_print is None or _PT_ANSI is None:
        try:
            print(text)
        except Exception:
            pass
        return

    # Branch 1: app context isn't loaded yet (banner render) or not
    # running (non-interactive CLI). Direct print is safe.
    app: Any | None = None
    try:
        from prompt_toolkit.application import get_app_or_none

        app = get_app_or_none()
    except Exception:
        app = None

    if app is None or not getattr(app, "_is_running", False):
        try:
            _pt_print(_PT_ANSI(text))
        except Exception:
            try:
                print(text)
            except Exception:
                pass
        return

    # Branch 2: app is running. If we are already on the app's event loop
    # thread, a direct print is also safe (and matches the streaming path
    # used by token deltas / spinner frames).
    try:
        loop = app.loop  # type: ignore[attr-defined]
    except Exception:
        loop = None
    if loop is None:
        try:
            _pt_print(_PT_ANSI(text))
        except Exception:
            try:
                print(text)
            except Exception:
                pass
        return

    current_loop: Any | None = None
    try:
        current_loop = _asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    except Exception:
        current_loop = None
    if current_loop is loop and loop.is_running():
        try:
            _pt_print(_PT_ANSI(text))
        except Exception:
            try:
                print(text)
            except Exception:
                pass
        return

    # Branch 3: cross-thread emission. Schedule ``run_in_terminal`` on the
    # app's loop so the input area pauses, the line prints above it, and
    # the prompt redraws cleanly.
    def _schedule() -> None:
        from prompt_toolkit.application import run_in_terminal

        try:
            import asyncio as _aio
            import inspect as _inspect

            _result = run_in_terminal(lambda: _pt_print(_PT_ANSI(text)))
            if _inspect.isawaitable(_result):
                # prompt_toolkit >= 3.0 returns a coroutine; ensure_future
                # so it actually awaits (calling it bare would drop the
                # output silently — same fix as cli.py:3686).
                _aio.ensure_future(_result)
        except Exception:
            # Fallback: ``run_in_terminal`` already invoked the lambda
            # synchronously on some prompt_toolkit builds / mocks, so do
            # NOT re-print here or the line shows twice.
            pass

    try:
        loop.call_soon_threadsafe(_schedule)
    except Exception:
        try:
            _pt_print(_PT_ANSI(text))
        except Exception:
            try:
                print(text)
            except Exception:
                pass


def cprint_safe(text: str) -> None:
    """Cprint wrapper that always lands somewhere, even if everything fails.

    Used by background threads that must never raise — caller behavior
    contract is "show this somewhere or silently drop", matching the
    pre-existing ``pass # never break the session over an update notice``
    pattern in ``banner._defer_update_notice``.
    """
    try:
        cprint(text)
    except Exception:
        try:
            print(text)
        except Exception:
            pass
