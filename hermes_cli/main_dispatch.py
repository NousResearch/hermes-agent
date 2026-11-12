"""Run the parsed ``hermes`` command."""

import contextlib
import sys
from typing import Iterator


@contextlib.contextmanager
def config_refusals_reported() -> Iterator[None]:
    """An expected config save refusal (``ConfigWriteError``) is its message on stderr and rc 1,
    not a traceback (the same shape as ``hermes config set``)."""
    from hermes_cli.config_backend import ConfigWriteError
    try:
        yield
    except ConfigWriteError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def run_subcommand(args, parser) -> None:
    """A handler's int return code becomes the exit code (None = success)."""
    if not hasattr(args, "func"):
        parser.print_help()
        return
    with config_refusals_reported():
        rc = args.func(args)
    if isinstance(rc, int) and rc != 0:
        sys.exit(rc)
