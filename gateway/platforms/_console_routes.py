"""Master-console route registration — fork-only, not in upstream.

Called from api_server.connect() via a try/import hook so the entire
console API surface lives here rather than scattered across the core
api_server.py connect() method.  Moving it here means future upstream
edits to connect() never conflict with fork-specific route additions.

Registered APIs:
  Kanban    /api/kanban/...
  Actions   /v1/actions/...          (tool-gate deferred approvals)
  Config    /api/config, /api/profiles, /api/gateway/*, /api/snapshot
  Soul      /api/soul
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.platforms.api_server import APIServerAdapter


def register(server: "APIServerAdapter") -> None:
    """Register all console routes on *server*'s aiohttp app."""
    app = server._app

    from gateway.platforms import kanban_api
    app.router.add_get("/api/kanban/tasks", partial(kanban_api.handle_list_tasks, server))
    app.router.add_post("/api/kanban/tasks", partial(kanban_api.handle_create_task, server))
    app.router.add_get("/api/kanban/assignees", partial(kanban_api.handle_assignees, server))
    app.router.add_get("/api/kanban/dispatch/state", partial(kanban_api.handle_dispatch_state, server))
    app.router.add_get("/api/kanban/tasks/{task_id}", partial(kanban_api.handle_get_task, server))
    app.router.add_patch("/api/kanban/tasks/{task_id}", partial(kanban_api.handle_patch_task, server))
    app.router.add_post("/api/kanban/tasks/{task_id}/assign", partial(kanban_api.handle_assign_task, server))
    app.router.add_post("/api/kanban/tasks/{task_id}/comment", partial(kanban_api.handle_comment_task, server))

    from gateway.platforms import actions_api
    app.router.add_get("/v1/actions", partial(actions_api.handle_list_actions, server))
    app.router.add_get("/v1/actions/{pending_id}", partial(actions_api.handle_get_action, server))
    app.router.add_post("/v1/actions/{pending_id}/approve", partial(actions_api.handle_approve_action, server))
    app.router.add_post("/v1/actions/{pending_id}/reject", partial(actions_api.handle_reject_action, server))

    from gateway.platforms import config_api
    app.router.add_get("/api/config", partial(config_api.handle_get_config, server))
    app.router.add_put("/api/config", partial(config_api.handle_put_config, server))
    app.router.add_get("/api/profiles", partial(config_api.handle_list_profiles, server))
    app.router.add_post("/api/profiles", partial(config_api.handle_create_profile, server))
    app.router.add_post("/api/gateway/restart", partial(config_api.handle_restart_gateway, server))
    app.router.add_post("/api/gateway/start", partial(config_api.handle_start_gateway, server))
    app.router.add_post("/api/gateway/stop", partial(config_api.handle_stop_gateway, server))
    app.router.add_post("/api/profiles/{name}/archive", partial(config_api.handle_archive_profile, server))
    app.router.add_post("/api/snapshot", partial(config_api.handle_snapshot, server))

    from gateway.platforms import soul_api
    app.router.add_get("/api/soul", partial(soul_api.handle_get_soul, server))
    app.router.add_put("/api/soul", partial(soul_api.handle_put_soul, server))
