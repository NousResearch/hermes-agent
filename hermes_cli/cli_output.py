"""Shared CLI output helpers for Hermes CLI modules.

Extracts the identical ``print_info/success/warning/error`` and ``prompt()``
functions previously duplicated across setup.py, tools_config.py,
mcp_config.py, and memory_setup.py.
"""

import getpass

from hermes_cli.colors import Colors, color


# ─── Print Helpers ────────────────────────────────────────────────────────────


def print_info(text: str) -> None:
    """Print a dim informational message."""
    print(color(f"  {text}", Colors.DIM))


def print_success(text: str) -> None:
    """Print a green success message with ✓ prefix."""
    print(color(f"✓ {text}", Colors.GREEN))


def print_warning(text: str) -> None:
    """Print a yellow warning message with ⚠ prefix."""
    print(color(f"⚠ {text}", Colors.YELLOW))


def print_error(text: str) -> None:
    """Print a red error message with ✗ prefix."""
    print(color(f"✗ {text}", Colors.RED))


def print_header(text: str) -> None:
    """Print a bold yellow header."""
    print(color(f"\n  {text}", Colors.YELLOW))


# ─── Input Prompts ────────────────────────────────────────────────────────────


def read_secret_line(prompt: str = "") -> str:
    """Read a password/secret line with ASCII control characters stripped.

    Wraps ``getpass.getpass()`` and removes ``\\x00``-``\\x1f`` from the
    returned string.

    On Windows, ``getpass.getpass()`` uses ``msvcrt.getwch()``, which emits
    ``\\x00`` (or ``\\xe0``) followed by a scan code for special keys
    (arrow keys, function keys, etc.) instead of filtering them out. An
    inadvertent arrow keypress just before/during paste therefore injects
    e.g. ``\\x00K`` (Left Arrow) into the value. When that value is later
    assigned to ``os.environ`` it raises ``ValueError: embedded null
    character``; when sent as an HTTP header it fails ASCII encoding.

    The caller is responsible for ``.strip()`` if needed — callers that
    feed the value through additional sanitization (e.g. paste cleanup)
    may want the raw, control-char-free string.
    """
    return getpass.getpass(prompt).translate({i: None for i in range(32)})


def prompt(
    question: str,
    default: str | None = None,
    password: bool = False,
) -> str:
    """Prompt the user for input with optional default and password masking.

    Replaces the four independent ``_prompt()`` / ``prompt()`` implementations
    in setup.py, tools_config.py, mcp_config.py, and memory_setup.py.

    Returns the user's input (stripped), or *default* if the user presses Enter.
    Returns empty string on Ctrl-C or EOF.
    """
    suffix = f" [{default}]" if default else ""
    display = color(f"  {question}{suffix}: ", Colors.YELLOW)

    try:
        if password:
            value = read_secret_line(display)
        else:
            value = input(display)
        value = value.strip()
        return value if value else (default or "")
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer. Returns bool."""
    hint = "Y/n" if default else "y/N"
    answer = prompt(f"{question} ({hint})")
    if not answer:
        return default
    return answer.lower().startswith("y")
