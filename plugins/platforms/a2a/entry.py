"""CLI entry point for the hermes-agent A2A server.

Usage::

    hermes-a2a                       # serve on 127.0.0.1:9100
    hermes-a2a --host 0.0.0.0 --port 9100
    python -m plugins.platforms.a2a --check

The Agent Card is served at ``/.well-known/agent-card.json`` and the JSON-RPC
endpoint at ``/``.
"""

# IMPORTANT: hermes_bootstrap must be the very first import — it configures
# UTF-8 stdio on Windows. No-op on POSIX. See hermes_bootstrap.py for the
# full rationale (mirrors acp_adapter/entry.py).
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet (e.g. a half-finished ``hermes update``). UTF-8 stdio setup is then
    # skipped on Windows; POSIX is unaffected.
    pass
else:
    hermes_bootstrap.harden_import_path()

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from gateway.platforms.base import is_network_accessible

from .config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    A2ASettings,
    load_a2a_settings,
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _load_env() -> None:
    """Load ``~/.hermes/.env`` so the agent picks up provider credentials."""
    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        from hermes_constants import get_hermes_home

        load_hermes_dotenv(hermes_home=get_hermes_home())
    except Exception:
        logging.getLogger(__name__).debug(
            "Could not load ~/.hermes/.env; using system env", exc_info=True
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hermes-a2a",
        description="Run Hermes Agent as an A2A (Agent2Agent) server.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=(
            f"Bind host (default {DEFAULT_HOST}). Use 0.0.0.0 to expose on the "
            "network — the endpoint is UNAUTHENTICATED, so put it behind a "
            "reverse proxy or auth layer."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Bind port (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--public-url",
        default=None,
        help="Base URL advertised in the Agent Card (default http://<host>:<port>/).",
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify A2A deps + card build, then exit."
    )
    parser.add_argument(
        "--version", action="store_true", help="Print Hermes version and exit."
    )
    return parser.parse_args(argv)


def _default_service_url(host: str, port: int) -> str:
    advertised_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{advertised_host}:{port}/"


def build_app(
    host: str,
    port: int,
    public_url: str | None = None,
    *,
    settings: A2ASettings | None = None,
):
    """Build the A2A Starlette ASGI app backed by the real Hermes agent."""
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler

    from .card import build_agent_card
    from .executor import HermesAgentExecutor
    from .sessions import ContextSessionStore
    from .task_store import BoundedTaskStore

    resolved = settings or load_a2a_settings()
    url = public_url or _default_service_url(host, port)
    executor = HermesAgentExecutor(
        ContextSessionStore(max_sessions=resolved.max_sessions),
        max_concurrency=resolved.max_concurrency,
        tool_io_mode=resolved.tool_io,
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=BoundedTaskStore(
            max_tasks=resolved.max_tasks,
            max_history_messages=resolved.max_task_history,
        ),
    )

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await executor.aclose()

    return A2AStarletteApplication(
        agent_card=build_agent_card(url),
        http_handler=handler,
    ).build(lifespan=lifespan)


def _run_check() -> None:
    import a2a  # noqa: F401

    from .card import build_agent_card
    from .executor import HermesAgentExecutor  # noqa: F401

    card = build_agent_card("http://127.0.0.1:9100/")
    assert card.name and card.skills, "Agent card is missing name/skills"
    print("Hermes A2A check OK")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.version:
        from .card import _hermes_version

        print(_hermes_version())
        return
    if args.check:
        _run_check()
        return

    _setup_logging()
    _load_env()
    logger = logging.getLogger(__name__)

    settings = load_a2a_settings()
    if args.host is not None:
        settings = replace(settings, host=args.host)
    if args.port is not None:
        if not 1 <= args.port <= 65535:
            raise SystemExit("--port must be between 1 and 65535")
        settings = replace(settings, port=args.port)
    if args.public_url is not None:
        settings = replace(settings, public_url=args.public_url)

    # Ensure the project root is importable so ``from run_agent import AIAgent``
    # works when launched as a console script.
    project_root = str(Path(__file__).resolve().parents[3])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # MCP tool discovery from config.yaml — run before serving so the agents
    # spawned per A2A context expose the user's configured MCP tools (mirrors
    # acp_adapter/entry.py).
    try:
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()
    except Exception:
        logger.debug("MCP tool discovery failed at A2A startup", exc_info=True)

    import uvicorn

    app = build_app(
        settings.host,
        settings.port,
        settings.public_url,
        settings=settings,
    )
    logger.info(
        "Starting hermes-agent A2A server on http://%s:%d "
        "(card: /.well-known/agent-card.json)",
        settings.host,
        settings.port,
    )
    if is_network_accessible(settings.host):
        logger.warning(
            "Binding %s — the A2A endpoint is UNAUTHENTICATED. "
            "Put it behind a reverse proxy or auth layer.",
            settings.host,
        )

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
