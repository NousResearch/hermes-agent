"""Shared CLI output helpers for Hermes CLI modules.

Extracts the identical ``print_info/success/warning/error`` and ``prompt()``
functions previously duplicated across setup.py, tools_config.py,
mcp_config.py, and memory_setup.py.
"""

from collections.abc import Collection, Sequence

from wcwidth import wcswidth

from hermes_cli.colors import Colors, color
from hermes_cli.secret_prompt import masked_secret_prompt


STANDARD_BOX_WIDTH = 59


# ─── Box Formatting ──────────────────────────────────────────────────────────


def format_box(
    rows: Sequence[tuple[str, int]],
    *,
    divider_after: Collection[int] = (),
    width: int = STANDARD_BOX_WIDTH,
) -> tuple[str, ...]:
    """Return a complete box whose rows all have the requested cell width.

    Each row is a ``(content, left_padding)`` pair. ``divider_after`` contains
    zero-based row indexes after which a horizontal divider should be added.
    """
    if width < 2:
        raise ValueError("box width must leave room for both borders")

    inner_width = width - 2
    lines = [f"┌{'─' * inner_width}┐"]
    for index, (content, left_padding) in enumerate(rows):
        content_width = wcswidth(content)
        if content_width < 0:
            raise ValueError("box content contains a non-printable character")

        right_padding = inner_width - left_padding - content_width
        if left_padding < 0 or right_padding < 0:
            raise ValueError("box content does not fit within the requested width")

        lines.append(f"│{' ' * left_padding}{content}{' ' * right_padding}│")
        if index in divider_after:
            lines.append(f"├{'─' * inner_width}┤")
    lines.append(f"└{'─' * inner_width}┘")
    return tuple(lines)


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
            value = masked_secret_prompt(display)
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
