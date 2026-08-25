"""Shared CLI output helpers (``print_*`` + ``prompt()``) for the setup/config wizards."""

import sys

import sys

from hermes_cli.colors import Colors, color
from hermes_cli.secret_prompt import masked_secret_prompt


def print_info(text: str) -> None:
    print(color(f"  {text}", Colors.DIM))


def print_success(text: str) -> None:
    print(color(f"✓ {text}", Colors.GREEN))


def print_warning(text: str) -> None:
    print(color(f"⚠ {text}", Colors.YELLOW))


def print_error(text: str) -> None:
    print(color(f"✗ {text}", Colors.RED))


def print_header(text: str) -> None:
    print(color(f"\n  {text}", Colors.YELLOW))


def line_input(prompt_text: str) -> str:
    """Read non-secret text with normal cursor-editing keys on a real TTY.


def line_input(prompt_text: str) -> str:
    """Read non-secret text with normal cursor-editing keys on a real TTY.

    Setup and model-selection commands run outside the interactive chat's
    prompt-toolkit application, so they can safely use a short-lived prompt
    here. Redirected input and output retain the built-in ``input`` behavior
    used by scripts, tests, and numbered fallbacks.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return input(prompt_text)

    try:
        from prompt_toolkit import prompt as prompt_toolkit_prompt
        from prompt_toolkit.formatted_text import ANSI
    except ImportError:
        return input(prompt_text)

    try:
        return prompt_toolkit_prompt(ANSI(prompt_text))
    except (KeyboardInterrupt, EOFError):
        raise
    except Exception:
        # Some terminals report isatty() == True yet reject registering stdin
        # with the asyncio event-loop selector (observed on macOS, where kqueue
        # raises EINVAL / "Invalid argument" for fd 0). prompt_toolkit cannot
        # attach its input there, so fall back to the built-in line reader,
        # which needs no selector and works in cooked mode.  Any prompt_toolkit
        # runtime failure (OSError, ValueError, RuntimeError) degrades the same
        # way — the wizard proceeds instead of crashing.
        return input(prompt_text)


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
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return input(prompt_text)
    try:
        from prompt_toolkit import prompt as prompt_toolkit_prompt
        from prompt_toolkit.formatted_text import ANSI
    except ImportError:
        return input(prompt_text)
    try:
        return prompt_toolkit_prompt(ANSI(prompt_text))
    except (KeyboardInterrupt, EOFError):
        raise
    except Exception:
        # Some terminals report isatty() yet reject registering stdin with the asyncio selector
        # (macOS kqueue raises EINVAL for fd 0). Any prompt_toolkit runtime failure degrades to
        # the built-in reader, which needs no selector — the wizard proceeds instead of crashing.
        return input(prompt_text)


def prompt(question: str, default: str | None = None, password: bool = False) -> str:
    """Prompt for input (stripped), or ``default`` on plain Enter; "" on Ctrl-C/EOF."""
    suffix = f" [{default}]" if default else ""
    display = color(f"  {question}{suffix}: ", Colors.YELLOW)
    try:
        if password:
            value = masked_secret_prompt(display)
        else:
            value = line_input(display)
        value = value.strip()
        return value if value else (default or "")
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


def prompt_yes_no(question: str, default: bool = True) -> bool:
    answer = prompt(f"{question} ({'Y/n' if default else 'y/N'})")
    if not answer:
        return default
    return answer.lower().startswith("y")
