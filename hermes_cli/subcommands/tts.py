"""``hermes tts`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_tts_parser(subparsers, *, cmd_tts: Callable) -> None:
    """Attach the ``tts`` subcommand (and its sub-actions) to ``subparsers``."""
    tts_parser = subparsers.add_parser(
        "tts",
        help="Text-to-speech utilities",
        description="Serve the configured TTS provider over an OpenAI-compatible API",
    )
    tts_subparsers = tts_parser.add_subparsers(dest="tts_command")

    serve_parser = tts_subparsers.add_parser(
        "serve",
        help="Serve the configured TTS provider on an OpenAI-compatible endpoint",
        description=(
            "Run an OpenAI-compatible TTS server backed by the provider in tts.provider. "
            "Point a client (e.g. the Hermes WebUI's 'OpenAI TTS (server)' engine) at "
            "http://HOST:PORT/v1 to use a local voice."
        ),
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1",
        help="Address to bind (default: 127.0.0.1; loopback only)")
    serve_parser.add_argument(
        "--port", type=int, default=5099,
        help="Port to listen on (default: 5099)")
    serve_parser.add_argument(
        "--allow-remote", action="store_true",
        help="Allow binding a non-loopback address (the server has no authentication)")
    serve_parser.set_defaults(func=cmd_tts)
