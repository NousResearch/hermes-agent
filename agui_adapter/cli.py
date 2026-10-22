"""Shared, dependency-light launch interface for all AG-UI entry points."""

# Bootstrap precedes every other import, including argparse, on Windows.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise
else:
    hermes_bootstrap.harden_import_path()

import argparse
import importlib
import sys


def add_arguments(parser):
    """Original standalone launch flags, shared by the module and Hermes CLI."""
    parser.add_argument(
        "--host",
        dest="agui_host",
        help="Override config.yaml agui.host (default 127.0.0.1). A network bind requires a token.",
    )
    parser.add_argument(
        "--port",
        dest="agui_port",
        type=int,
        help="Override config.yaml agui.port (default 8000).",
    )
    parser.add_argument(
        "--token",
        dest="agui_token",
        help="Override the profile's HERMES_AGUI_SESSION_TOKEN secret.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies/imports and exit without starting a server.",
    )


def run(args):
    """Check optional dependencies, then delegate to the standalone server."""
    try:
        for name in ("ag_ui.core", "fastapi", "uvicorn", "agui_adapter.server"):
            importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name.split(".")[0] not in {"ag_ui", "fastapi", "uvicorn"}:
            raise
        from pm import install_hint

        print(f"AG-UI dependency unavailable: {exc}", file=sys.stderr)
        print(install_hint("agui"), file=sys.stderr)
        raise SystemExit(1) from exc
    if args.check:
        print("Hermes AG-UI check OK")
        return
    from agui_adapter.entry import main as serve

    return serve(host=args.agui_host, port=args.agui_port, token=args.agui_token)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Hermes as an AG-UI HTTP/SSE server."
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))
