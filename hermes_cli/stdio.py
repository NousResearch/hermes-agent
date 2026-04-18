"""Windows stdio helpers for the Hermes CLI and agent entrypoints.

Native Windows shells still sometimes start Python with a legacy console code
page and non-UTF-8 stdio encoding. Hermes prints Unicode in many user-facing
paths, so best-effort startup normalization is needed to avoid
UnicodeEncodeError crashes on otherwise healthy systems.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any


_UTF8_CODEPAGE = 65001


class _WindowsSafeWriter:
    """Transparent stdio wrapper that degrades Unicode to ASCII escapes.

    We prefer to keep Unicode output when the terminal supports it. This
    wrapper is only a last-resort safety net for streams that still raise
    UnicodeEncodeError after reconfiguration.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)

    def write(self, data: Any):
        try:
            return self._inner.write(data)
        except UnicodeEncodeError:
            if not isinstance(data, str):
                return 0
            fallback = data.encode("ascii", errors="backslashreplace").decode("ascii")
            try:
                self._inner.write(fallback)
                return len(data)
            except (UnicodeEncodeError, OSError, ValueError):
                return len(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _set_console_utf8_codepage() -> None:
    """Switch the attached Windows console to UTF-8 when possible."""
    try:
        kernel32 = ctypes.windll.kernel32
    except AttributeError:
        return

    try:
        kernel32.SetConsoleOutputCP(_UTF8_CODEPAGE)
        kernel32.SetConsoleCP(_UTF8_CODEPAGE)
    except Exception:
        return


def _reconfigure_stream(stream: Any) -> Any:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    return stream


def install_windows_stdio(*, force: bool = False) -> None:
    """Best-effort UTF-8 stdio normalization for native Windows sessions."""
    if not force and sys.platform != "win32":
        return

    _set_console_utf8_codepage()

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        if isinstance(stream, _WindowsSafeWriter):
            continue
        stream = _reconfigure_stream(stream)
        setattr(sys, stream_name, _WindowsSafeWriter(stream))
