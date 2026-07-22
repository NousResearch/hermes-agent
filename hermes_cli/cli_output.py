"""Shared CLI output helpers for Hermes CLI modules.

Extracts the identical ``print_info/success/warning/error`` and ``prompt()``
functions previously duplicated across setup.py, tools_config.py,
mcp_config.py, and memory_setup.py.
"""

from hermes_cli.colors import Colors, color, RichColors, rich_color
from hermes_cli.secret_prompt import masked_secret_prompt
import rich
from rich.prompt import Prompt

# ─── Print Helpers ────────────────────────────────────────────────────────────


def print_info(text: str) -> None:
    """Print a dim informational message."""
    rich.print(rich_color(f"  {text}", RichColors.DIM))


def print_success(text: str) -> None:
    """Print a green success message with ✓ prefix."""
    rich.print(rich_color(f"✓ {text}", RichColors.GREEN))


def print_warning(text: str) -> None:
    """Print a yellow warning message with ⚠ prefix."""
    rich.print(rich_color(f"⚠ {text}", RichColors.YELLOW))


def print_error(text: str) -> None:
    """Print a red error message with ✗ prefix."""
    rich.print(rich_color(f"✗ {text}", RichColors.RED))


def print_header(text: str) -> None:
    """Print a bold yellow header."""
    rich.print(rich_color(f"\n  {text}", RichColors.YELLOW))


# ─── Input Prompts ────────────────────────────────────────────────────────────
class CustomPrompt(Prompt):
    prompt_suffix = ""

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
    display = rich_color(f"  {question}{suffix}: ", RichColors.YELLOW)

    try:
        if password:
            value = masked_secret_prompt(display)
        else:
            value = CustomPrompt.ask(display)
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
