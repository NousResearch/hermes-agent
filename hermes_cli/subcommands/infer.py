"""Parser for a single model completion without an agent session."""

from __future__ import annotations

from typing import Callable


def build_infer_parser(subparsers, *, cmd_infer: Callable) -> None:
    parser = subparsers.add_parser(
        "infer", help="Call a configured model once without an agent session",
        description="Run one model completion using this profile's provider and credentials. "
                    "No agent, tools, memory, or chat session is created.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt text, or '-' to read standard input")
    parser.add_argument("--provider", help="Configured provider to use (requires --model)")
    parser.add_argument("--model", help="Model ID or Hermes model alias (default: selected model)")
    parser.add_argument("--schema", metavar="PATH", help="JSON Schema file for structured output")
    parser.add_argument("--json", action="store_true", help="Print text and route/usage metadata as JSON")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum output tokens")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds")
    parser.set_defaults(func=cmd_infer)
