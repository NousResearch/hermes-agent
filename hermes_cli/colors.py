"""Shared ANSI color utilities for Hermes CLI modules."""

import os
import sys


def _is_sgr_mangling_proxy(stream) -> bool:
    """True when *stream* is a prompt_toolkit ``StdoutProxy`` (``patch_stdout``).

    ``StdoutProxy.isatty()`` forwards the *underlying* terminal's answer, so a TTY
    check alone concludes "color is fine" — but the proxy feeds text through
    ``Vt100_Output.write()``, which replaces every ESC byte with ``?``
    (prompt_toolkit/output/vt100.py). SGR codes written there render as
    ``?[33m...?[0m`` instead of color.

    Matched by class name on purpose: the CLI imports prompt_toolkit lazily and
    tolerates a broken install (#96075), so this module must not grow a hard
    ``prompt_toolkit`` import.
    """
    return type(stream).__name__ == "StdoutProxy"


def should_use_color(stream=None) -> bool:
    """Return True when colored output is appropriate for *stream* (default: stdout).

    Pass ``stream=sys.stderr`` from stderr writers: the TTY decision belongs to the
    stream the text is written to, and a terminal whose stderr is redirected to a
    log must not receive raw escape sequences.
    """
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return False
    target = sys.stdout if stream is None else stream
    isatty = getattr(target, "isatty", None)
    if not (isatty and isatty()):
        return False
    return not _is_sgr_mangling_proxy(target)


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def color(text: str, *codes, stream=None) -> str:
    """Apply color codes to text (only when color output is appropriate).

    ``stream`` selects the stream the result will be written to (default stdout);
    the TTY check applies to that stream. The default path calls
    ``should_use_color()`` with no arguments, so the existing zero-arg
    ``monkeypatch.setattr("hermes_cli.colors.should_use_color", lambda: True)``
    sites keep working.
    """
    use_color = should_use_color() if stream is None else should_use_color(stream)
    if not use_color:
        return text
    return "".join(codes) + text + Colors.RESET
