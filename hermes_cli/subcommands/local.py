"""``hermes local`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_local_parser(subparsers, *, cmd_local: Callable) -> None:
    """Attach the ``local`` subcommand to ``subparsers``."""

    local_parser = subparsers.add_parser(
        "local",
        help="Inspect local model runtimes (LM Studio, Ollama, OpenAI-compatible)",
    )
    local_subparsers = local_parser.add_subparsers(dest="local_action")

    models = local_subparsers.add_parser(
        "models",
        aliases=["ls", "list"],
        help="List models installed in local runtimes",
        description=(
            "Discover local models without changing configuration or starting "
            "servers. LM Studio is queried via `lms ls`; Ollama uses `ollama "
            "list` with a manifest fallback when the daemon is stopped."
        ),
    )
    models.add_argument(
        "--backend",
        choices=["all", "lmstudio", "ollama", "openai-compatible", "openai"],
        default="all",
        help="Runtime backend to inspect (default: all)",
    )
    models.add_argument(
        "--base-url",
        default="",
        help="OpenAI-compatible base URL to probe, e.g. http://127.0.0.1:1234/v1",
    )
    models.add_argument(
        "--api-key",
        default="",
        help="Optional bearer token for --base-url",
    )
    models.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Include embedding models when a runtime reports them",
    )
    models.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-backend probe timeout in seconds (default: 10)",
    )
    models.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )

    local_parser.set_defaults(func=cmd_local)
