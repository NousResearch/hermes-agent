"""`hermes gateway` subcommand handler.

F-C1 step 6 — sixth subcommand extracted out of
``hermes_cli/main.py``. The gateway handler is a thin pass-through
that hands off to ``hermes_cli.gateway.gateway_command``. All
gateway sub-action parsing (`run`, `status`, `stop`, etc.) happens
inside ``gateway_command``; this module only owns the dispatcher.
"""

from __future__ import annotations


def cmd_gateway(args):
    """Gateway management commands."""
    from hermes_cli.gateway import gateway_command

    gateway_command(args)
