"""Windows ConPTY bridge for the `hermes dashboard` chat tab."""

from __future__ import annotations

import sys
import time
from typing import Optional, Sequence

try:
    from winpty import PtyProcess  # type: ignore
    _PTY_AVAILABLE = sys.platform.startswith("win")
except ImportError:  # pragma: no cover - non-Windows or pywinpty missing
    PtyProcess = None  # type: ignore
    _PTY_AVAILABLE = False


__all__ = ["WinPtyBridge", "PtyUnavailableError"]


# Same clamp ceiling as the POSIX bridge so a broken winsize probe never reaches the resize call.
_MIN_DIMENSION = 1
_MAX_COLS = 2000
_MAX_ROWS = 1000


def _clamp(value: int, maximum: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return _MIN_DIMENSION
    return max(_MIN_DIMENSION, min(n, maximum))


class PtyUnavailableError(RuntimeError):
    """Raised when a PTY cannot be created on this platform."""


class WinPtyBridge:
    """pywinpty-backed bridge with the same interface as ``PtyBridge``. ``read`` runs inside
    ``run_in_executor``; ConPTY has no selectable fd, so it polls with a short sleep."""

    def __init__(self, proc: "PtyProcess") -> None:  # type: ignore[name-defined]
        self._proc = proc
        self._closed = False

    @classmethod
    def is_available(cls) -> bool:
        return bool(_PTY_AVAILABLE)

    @classmethod
    def spawn(
        cls, argv: Sequence[str], *, cwd: Optional[str] = None, env: Optional[dict] = None,
        cols: int = 80, rows: int = 24) -> "WinPtyBridge":
        if not _PTY_AVAILABLE:
            if PtyProcess is None:
                raise PtyUnavailableError("pywinpty is not installed. Install with: pip install pywinpty")
            raise PtyUnavailableError("ConPTY is unavailable on this platform.")
        # See pty_bridge.py: exact-preservation factory for the env=None fallback.
        from tools.environments.local import build_subprocess_env
        spawn_env = (
            build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
            if env is None else dict(env))
        spawn_env["TERM"] = spawn_env.get("TERM") or "xterm-256color"
        # pywinpty mirrors ptyprocess: dimensions=(rows, cols).
        return cls(PtyProcess.spawn(list(argv), cwd=cwd, env=spawn_env, dimensions=(rows, cols)))  # type: ignore[union-attr]

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def is_alive(self) -> bool:
        try:
            return not self._closed and bool(self._proc.isalive())
        except Exception:
            return False

    def read(self, timeout: float = 0.2) -> Optional[bytes]:
        """Up to 64 KiB of child output."""
        if self._closed:
            return None
        try:
            data = self._proc.read(65536)  # pywinpty returns str
        except Exception:
            return None
        if not data:
            # No fd to select on; sleep so the executor thread doesn't pin a core while idle.
            time.sleep(min(timeout, 0.02))
            return b""
        if isinstance(data, bytes):
            return data
        # pywinpty decodes internally, so a multibyte UTF-8 sequence can split across reads;
        # xterm.js tolerates the rare replacement char (the one fidelity tradeoff vs POSIX).
        return data.encode("utf-8", errors="replace")

    def write(self, data: bytes) -> None:
        if self._closed or not data:
            return
        try:
            self._proc.write(data.decode("utf-8", errors="replace"))  # pywinpty wants text
        except Exception:
            return

    def resize(self, cols: int, rows: int) -> None:
        if self._closed:
            return
        cols = _clamp(cols, _MAX_COLS)
        rows = _clamp(rows, _MAX_ROWS)
        try:
            self._proc.setwinsize(rows, cols)  # pywinpty: (rows, cols)
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.terminate(force=True)
        except Exception:
            pass

    def __enter__(self) -> "WinPtyBridge":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
