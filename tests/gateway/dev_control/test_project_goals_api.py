import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.dev_control.project_goals import DevProjectGoalStore, create_project_goal
from gateway.dev_execution import DevExecutionStore
from gateway.dev_control.routes import register_dev_control_routes
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware
from gateway.config import PlatformConfig


@pytest.mark.asyncio
async def test_project_goals_api_create_and_tree(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    register_dev_control_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        created = await cli.post(
            "/v1/dev/goals",
            json={"kind": "vision", "title": "North star", "project_id": "OrynWorkspace"},
            headers={"Authorization": "Bearer sk-secret"},
        )
        assert created.status == 200
        vision = await created.json()
        milestone_parent = create_project_goal(
            store=DevProjectGoalStore(tmp_path / "state.db"),
            kind="goal",
            title="Feature theme",
            project_id="OrynWorkspace",
            parent_goal_id=vision["goal_id"],
        )
        assert milestone_parent["parent_goal_id"] == vision["goal_id"]

        tree_resp = await cli.get(
            "/v1/dev/goals/tree?project_id=OrynWorkspace",
            headers={"Authorization": "Bearer sk-secret"},
        )
        assert tree_resp.status == 200
        tree = await tree_resp.json()
        assert tree["total"] >= 2
        assert tree["roots"][0]["title"] == "North star"
