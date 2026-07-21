"""Shared CLI output helpers for Hermes CLI modules.

Extracts the identical ``print_info/success/warning/error`` and ``prompt()``
functions previously duplicated across setup.py, tools_config.py,
mcp_config.py, and memory_setup.py.
"""

from hermes_cli.colors import Palette, semantic
from hermes_cli.secret_prompt import masked_secret_prompt


# ─── Print Helpers ────────────────────────────────────────────────────────────


def print_info(text: str) -> None:
    """Print a dim informational message (Palette.INFO)."""
    print(semantic(f"  {text}", Palette.INFO))


def print_success(text: str) -> None:
    """Print a success message with ✓ prefix (Palette.SUCCESS)."""
    print(semantic(f"✓ {text}", Palette.SUCCESS))


def print_warning(text: str) -> None:
    """Print a warning message with ⚠ prefix (Palette.WARNING)."""
    print(semantic(f"⚠ {text}", Palette.WARNING))


def print_error(text: str) -> None:
    """Print an error message with ✗ prefix (Palette.ERROR)."""
    print(semantic(f"✗ {text}", Palette.ERROR))


def print_header(text: str) -> None:
    """Print a section header in the design-system heading style.

    Uses :data:`Palette.HEADING` (cyan + bold) so section titles read
    consistently across the CLI instead of varying per call site.
    """
    print(semantic(f"\n  {text}", Palette.HEADING))


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
    display = semantic(f"  {question}{suffix}: ", Palette.PROMPT)

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
