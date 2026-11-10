"""Run the parsed ``hermes <subcommand>`` handler."""

import sys


def run_subcommand(args, parser) -> None:
    """A handler's int return code becomes the exit code (None = success). An expected config
    save refusal (``ConfigWriteError``) is its message on stderr and rc 1, not a traceback."""
    if not hasattr(args, "func"):
        parser.print_help()
        return
    from hermes_cli.config_backend import ConfigWriteError
    try:
        rc = args.func(args)
    except ConfigWriteError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    if isinstance(rc, int) and rc != 0:
        sys.exit(rc)
