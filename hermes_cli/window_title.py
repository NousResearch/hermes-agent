"""Session titles for the classic CLI's terminal window."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from collections.abc import Callable

from prompt_toolkit.output import Output
from prompt_toolkit.output.vt100 import Vt100_Output

logger = logging.getLogger(__name__)
_CONTROLS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


class WindowTitle:
    """Own the title while the REPL runs; all writes stay on the app loop."""

    poll_interval = 1.5

    def __init__(self, output: Output, get_title: Callable[[], str], *, cwd: str):
        self.output = output
        self.get_title = get_title
        self.fallback = os.path.basename(cwd.rstrip(os.sep)) or cwd
        self._last_title = None
        self._saved = False
        self._closed = False

    def refresh(self, app=None) -> None:
        if self._closed or (app is not None and app.is_done):
            return
        label = self.get_title() or self.fallback
        label = _CONTROLS.sub("", label.replace("\n", " ")).strip()
        title = f"Hermes · {label}"[:120]
        if title == self._last_title:
            return
        try:
            if self._last_title is None and isinstance(self.output, Vt100_Output):
                # Save inside the terminal, without querying or exposing the old title.
                self.output.write_raw("\x1b[22;2t")
                self._saved = True
            self.output.set_title(title)
            self.output.flush()
            self._last_title = title
        except (OSError, ValueError):
            self._closed = True
            logger.debug("Terminal title output unavailable", exc_info=True)

    async def poll(self, app) -> None:
        # Automatic titles may arrive while idle, with no input or screen repaint.
        while not self._closed and not app.is_done:
            self.refresh(app)
            await asyncio.sleep(self.poll_interval)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._last_title is None:
            return
        try:
            self.output.clear_title()
            if self._saved:
                self.output.write_raw("\x1b[23;2t")
            self.output.flush()
        except (OSError, ValueError):
            logger.debug("Could not restore terminal title", exc_info=True)


def install_window_title(app, get_title: Callable[[], str], *, enabled: bool = True):
    """Attach only to an interactive REPL; one-shot and redirected output stay clean."""
    if not enabled or not sys.stdout.isatty() or os.environ.get("TERM") == "dumb":
        return None
    title = WindowTitle(app.output, get_title, cwd=os.getcwd())
    app.before_render += title.refresh
    app.pre_run_callables.append(lambda: app.create_background_task(title.poll(app)))
    return title
