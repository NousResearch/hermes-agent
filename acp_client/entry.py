"""Self-test entry for the ACP client skeleton (no server, no CLI launch).

Unlike ``acp_adapter/entry.py`` (which *runs* an ACP server), the client is
library-only in Phase 1.  This entry exists solely to satisfy the Phase-1
acceptance bullet "``hermes acp-client --check`` reports module importable":

    python -m acp_client --check     # verify acp + acp_client import cleanly
    python -m acp_client --version    # print Hermes version

It never spawns an external CLI and never touches credentials.
"""

from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hermes-acp-client",
        description="Self-test for the Hermes ACP client subsystem (library-only).",
    )
    parser.add_argument("--version", action="store_true", help="Print Hermes version and exit")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the acp dependency and acp_client modules import, then exit",
    )
    return parser.parse_args(argv)


def _run_check() -> None:
    import acp  # noqa: F401

    from acp_client.connection import OutboundConnection  # noqa: F401
    from acp_client.event_translator import EventTranslator  # noqa: F401
    from acp_client.outbound_session import OutboundSessionManager  # noqa: F401
    from acp_client.permission_relay import PermissionRelay  # noqa: F401
    from acp_client.transport_registry import TransportRegistry  # noqa: F401

    print("Hermes ACP client check OK")


def _print_version() -> None:
    from hermes_cli import __version__ as hermes_version

    print(hermes_version)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.version:
        _print_version()
        return
    if args.check:
        _run_check()
        return
    # No server mode in Phase 1 — print usage and exit non-zero so callers
    # don't mistake a bare invocation for a running transport.
    print(
        "acp_client is library-only in Phase 1; use --check or import "
        "acp_client.connection.OutboundConnection.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
