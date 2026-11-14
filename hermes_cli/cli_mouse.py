"""``/mouse [on|off|status]``: live mouse support for the classic CLI.

The Application's ``mouse_support`` is a prompt_toolkit filter over ``cli._mouse_enabled``; the
renderer re-reads it on every paint and sends the enable/disable sequences itself, so flipping
the attribute and repainting is the whole switch. ``display.cli_mouse`` persists it.
"""
from __future__ import annotations

from agent.i18n import t

_ON = ("on", "true", "yes", "1")
_OFF = ("off", "false", "no", "0")


def mouse_filter(cli):
    """The ``mouse_support`` filter the CLI Application is built with."""
    from prompt_toolkit.filters import Condition
    return Condition(lambda: bool(getattr(cli, "_mouse_enabled", False)))


def set_mouse(cli, enabled: bool) -> None:
    """Flip mouse reporting now (next paint) and persist ``display.cli_mouse``."""
    from cli import save_config_value
    cli._mouse_enabled = bool(enabled)
    save_config_value("display.cli_mouse", cli._mouse_enabled)
    app = getattr(cli, "_app", None)
    if app is not None:
        app.invalidate()


def handle_mouse_command(cli, cmd_original: str) -> None:
    parts = (cmd_original or "").split()
    arg = parts[1].lower() if len(parts) > 1 else "toggle"
    current = bool(getattr(cli, "_mouse_enabled", False))
    if arg in ("status", "show"):
        state = t("cli.shared.label_on") if current else t("cli.shared.label_off")
        cli._console_print(f"  {t('cli.display.mouse_state', state=state)}", highlight=False, markup=False)
        return
    if arg in _ON:
        target = True
    elif arg in _OFF:
        target = False
    elif arg == "toggle":
        target = not current
    else:
        cli._console_print(f"  {t('cli.display.mouse_usage')}", highlight=False, markup=False)
        return
    set_mouse(cli, target)
    key = "cli.display.mouse_on" if target else "cli.display.mouse_off"
    cli._console_print(f"  {t(key)}", highlight=False, markup=False)
