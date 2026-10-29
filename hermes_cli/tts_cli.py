"""``hermes tts`` subcommand — serve the configured TTS provider over HTTP.

``hermes tts serve`` starts an OpenAI-compatible TTS server backed by the
provider in ``tts.provider``. The server lives in :mod:`hermes_cli.tts_server`;
this module owns the command dispatch.
"""

from __future__ import annotations

import argparse


def cmd_tts(args: argparse.Namespace) -> int:
    """Handle ``hermes tts`` subcommands."""
    subcmd = getattr(args, "tts_command", None)
    if subcmd == "serve":
        return _cmd_tts_serve(args)
    if subcmd:
        print(f"Unknown tts command: {subcmd}")
    else:
        print("Usage: hermes tts serve [--host HOST] [--port PORT]")
    print("Run 'hermes tts --help' for the available sub-actions.")
    return 1


def _cmd_tts_serve(args: argparse.Namespace) -> int:
    """``hermes tts serve`` — run the OpenAI-compatible TTS server."""
    from hermes_cli.tts_server import serve

    return serve(
        host=args.host,
        port=args.port,
        allow_remote=args.allow_remote,
    )
