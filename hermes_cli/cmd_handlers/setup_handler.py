"""`hermes setup` subcommand handler.

F-C1 step 5 — fifth subcommand extracted out of
``hermes_cli/main.py``. The setup handler gates on TTY (unless
``--non-interactive`` is passed) and then hands off to the setup
wizard in ``hermes_cli.setup``. ``_require_tty`` stays in main.py
as a shared helper; the extracted handler imports it lazily so we
don't reshape that helper under F-C1.
"""

from __future__ import annotations


def cmd_setup(args):
    """Interactive setup wizard."""
    if not getattr(args, "non_interactive", False):
        from hermes_cli.main import _require_tty

        _require_tty("setup")
    from hermes_cli.setup import run_setup_wizard

    run_setup_wizard(args)
