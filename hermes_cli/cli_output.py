"""Shared CLI output helpers (``print_*`` + ``prompt()``) for the setup/config wizards."""

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

    Setup/model-selection commands run outside the chat's prompt_toolkit application, so a
    short-lived prompt is safe here. Redirected stdin/stdout keep the built-in ``input`` used by
    scripts, tests and numbered fallbacks.
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
        value = (masked_secret_prompt(display) if password else line_input(display)).strip()
        return value if value else (default or "")
    except (KeyboardInterrupt, EOFError):
        print()
        return ""


def prompt_yes_no(question: str, default: bool = True) -> bool:
    answer = prompt(f"{question} ({'Y/n' if default else 'y/N'})")
    if not answer:
        return default
    return answer.lower().startswith("y")
