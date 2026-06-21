from __future__ import annotations

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from .agent_card import build_agent_card
from .auth import A2AAuthMiddleware, A2AServerCallContextBuilder
from .config import SidecarConfig, load_sidecar_config
from .executor import HermesKanbanExecutor
from .store import HermesKanbanTaskStore


async def _healthz(_request):  # noqa: ANN001
    return JSONResponse({"ok": True, "service": "hermes-a2a-sidecar"})


def create_app(config: SidecarConfig | None = None) -> Starlette:
    """Create a Starlette app using the official A2A SDK route factories."""

    config = config or load_sidecar_config()
    public_card = build_agent_card(config, extended=False)
    extended_card = build_agent_card(config, extended=True)
    store = HermesKanbanTaskStore(config.audit_db_path, default_board=config.board)
    executor = HermesKanbanExecutor(config, store)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=store,
        agent_card=public_card,
        extended_agent_card=extended_card,
    )
    context_builder = A2AServerCallContextBuilder()
    routes = [Route("/healthz", endpoint=_healthz, methods=["GET"])]
    routes.extend(create_agent_card_routes(public_card))
    routes.extend(
        create_jsonrpc_routes(
            handler,
            rpc_url=config.rpc_path,
            context_builder=context_builder,
        )
    )
    app = Starlette(routes=routes)
    app.add_middleware(A2AAuthMiddleware, config=config)
    app.state.a2a_config = config
    app.state.a2a_task_store = store
    return app
