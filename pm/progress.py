"""One output policy for long child processes: verbose in CI, contained elsewhere.

CI logs are the only record of a remote failure, so CI (``CI`` /
``GITHUB_ACTIONS``) or ``HERMES_VERBOSE=1`` streams child output unchanged.
Everywhere else a child collapses into one status line: on a terminal it is
rewritten in place with the child's latest line, off a terminal only the
start and finish are printed. A failure always prints the captured tail, so
containment never hides an error. ``HERMES_VERBOSE=0`` forces containment.

Stdlib-only: the bootstrap runner imports this from a pre-3.11 system Python.
"""
from __future__ import annotations

from collections import deque
import os
import re
import shutil
import subprocess
import sys
import time
from typing import IO, Callable, Mapping, Optional, Protocol, Sequence

TAIL_LINES = 80
# A child that never prints a newline (a bare progress stream, a binary blob) must not
# grow memory without bound; the end of an over-long line is the informative part.
MAX_LINE = 4096
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# A redraw per child line would dominate a fast install's runtime on slow terminals.
_REDRAW_INTERVAL = 0.05


class TextSink(Protocol):
    def write(self, text: str, /) -> int: ...

    def flush(self) -> None: ...


def verbose_output(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether child output streams unchanged instead of being contained."""
    source = os.environ if env is None else env
    explicit = source.get("HERMES_VERBOSE", "").strip().lower()
    if explicit in _TRUE:
        return True
    if explicit in _FALSE:
        return False
    return any(source.get(key, "").strip().lower() not in ("", *_FALSE) for key in ("CI", "GITHUB_ACTIONS"))


def _is_terminal(stream: IO[str]) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


class LiveTail:
    """A text sink that contains a child's output behind one status line.

    ``label=None`` runs silently and speaks only on failure, for steps too
    quick to deserve a line of their own.
    """

    def __init__(self, label: Optional[str], stream: Optional[IO[str]] = None, *,
                 hide: Optional[Callable[[str], bool]] = None, indent: str = "") -> None:
        self.label = label
        self.stream = sys.stdout if stream is None else stream
        self.hide = hide
        self.indent = indent
        self.live = label is not None and _is_terminal(self.stream)
        self.tail: deque[str] = deque(maxlen=TAIL_LINES)
        self._partial = ""
        self._drawn = 0
        self._last_draw = 0.0
        if label is not None:
            if self.live:
                self._draw(f"{indent}→ {label}…")
            else:
                self._emit(f"{indent}→ {label}…\n")

    def write(self, text: str) -> int:
        # npm and uv redraw with bare CRs; each redraw is a line of progress.
        lines = (self._partial + text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self._partial = lines.pop()[-MAX_LINE:]
        for line in lines:
            self._line(line)
        return len(text)

    def flush(self) -> None:
        return None

    def close(self, ok: bool) -> None:
        if self._partial:
            self._line(self._partial)
            self._partial = ""
        if self.live:
            self._draw("")
            self._emit("\r")
        if ok:
            if self.label is not None:
                self._emit(f"{self.indent}✓ {self.label}\n")
            return
        name = self.label or "command"
        if not self.tail:
            self._emit(f"{self.indent}✗ {name} failed\n")
            return
        self._emit(f"{self.indent}✗ {name} failed; last {len(self.tail)} lines of output:\n")
        self._emit("".join(f"{self.indent}    {line}\n" for line in self.tail))

    def _line(self, line: str) -> None:
        line = _ANSI.sub("", line[-MAX_LINE:]).rstrip()
        if not line.strip():
            return
        self.tail.append(line)
        if not self.live or (self.hide is not None and self.hide(line)):
            return
        now = time.monotonic()
        if now - self._last_draw < _REDRAW_INTERVAL:
            return
        self._last_draw = now
        self._draw(f"{self.indent}→ {self.label}… {line.strip()}")

    def _draw(self, text: str) -> None:
        # Pad instead of an ANSI erase: a bare CR works on every Windows console.
        width = max(20, shutil.get_terminal_size().columns - 1)
        text = text[:width]
        self._emit("\r" + text + " " * max(0, self._drawn - len(text)))
        self._drawn = len(text)

    def _emit(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()


def run_contained(command: Sequence[str], label: str, *, stream: Optional[IO[str]] = None,
                  hide: Optional[Callable[[str], bool]] = None, indent: str = "",
                  **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run(check=True)`` under the output policy."""
    if verbose_output():
        out = sys.stdout if stream is None else stream
        out.write(f"{indent}→ {label}…\n")
        out.flush()
        return subprocess.run(command, check=True, **kwargs)
    tail = LiveTail(label, stream, hide=hide, indent=indent)
    captured = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", errors="replace")
    if not tail.live:
        try:
            result = subprocess.run(command, check=True, **captured, **kwargs)
        except subprocess.CalledProcessError as exc:
            tail.write(exc.output or "")
            tail.close(False)
            raise
        except BaseException:
            tail.close(False)
            raise
        tail.close(True)
        return result
    try:
        with subprocess.Popen(command, **captured, **kwargs) as proc:
            assert proc.stdout is not None  # stdout=PIPE above.
            for line in proc.stdout:
                tail.write(line)
            code = proc.wait()
    except BaseException:
        tail.close(False)
        raise
    tail.close(code == 0)
    output = "\n".join(tail.tail)
    if code:
        raise subprocess.CalledProcessError(code, list(command), output=output)
    return subprocess.CompletedProcess(list(command), code, output, None)
