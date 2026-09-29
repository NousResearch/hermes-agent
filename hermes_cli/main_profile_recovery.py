"""Which commands may run in the default home when the sticky ``active_profile`` is gone.

``_apply_profile_override`` fails closed when ``active_profile`` names a deleted profile:
running an ordinary command in the default profile would read or write the wrong profile's
state. The commands here are the way out of that state, so they must still reach their handler.
"""

from __future__ import annotations

import argparse
import contextlib
import io


def _uninstall_keeps_data(args: list[str]) -> bool:
    """True when ``uninstall <args>`` removes no user data.

    Parsed with the real ``uninstall`` subparser so abbreviations (``--dat``, ``--fu``) resolve
    exactly as they will for the handler. ``--full`` wipes the default root, which also holds
    every other profile under ``profiles/``, without asking, so it is refused like ``--data``.
    The plain interactive menu still offers a full wipe, but only behind its own confirmations.
    """
    from hermes_cli.subcommands.uninstall import build_uninstall_parser

    parser = argparse.ArgumentParser(add_help=False)
    build_uninstall_parser(parser.add_subparsers(dest="command"), cmd_uninstall=lambda _args: None)
    # The real parser prints help/usage errors once the handler runs; this probe stays silent.
    # --help exits 0 (harmless, let it through); an unparseable argv fails closed.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            ns, _ = parser.parse_known_args(["uninstall", *args])
        except SystemExit as exit_:
            return exit_.code == 0
    return not (ns.data or ns.full)


def is_stale_profile_recovery_command(argv: list[str]) -> bool:
    """``profile list``, ``profile use default``, or an ``uninstall`` that keeps user data."""
    from hermes_cli._parser import command_argv

    command = command_argv(argv)
    if command[:2] == ["profile", "list"]:
        return True
    if command[:2] == ["profile", "use"]:
        return len(command) > 2 and command[2].casefold() == "default"
    return command[:1] == ["uninstall"] and _uninstall_keeps_data(command[1:])
