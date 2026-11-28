"""``api_server_request`` hooks turn an API server request into its agent turn's RAM-only
``request_context``, which middleware receives; each admission executes with its own context only."""

import asyncio
import os
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import request_context
from gateway.session_api_turn import admit_api_turn, release_request_context
from tests.fakes.fake_llm_provider import (
    FakeLLMServer,
    Text,
    ToolCall,
    write_hermes_home,
)

_PLUGIN = """
HEADERS, SEEN = [], []

def register(ctx):
    def tenant(headers, **_):
        HEADERS.append(sorted(k.lower() for k in headers))
        return {"tenant": headers.get("X-Tenant", ""), "not-a-string": 1}

    def record(tool_name, request_context, next_call, args, **_):
        SEEN.append((tool_name, request_context))
        return next_call(args)

    ctx.register_hook("api_server_request", tenant)
    ctx.register_middleware("tool_execution", record)
"""

_CONFIG = (
    "updates:\n  check: false\n"
    "models_dev:\n  url: http://127.0.0.1:9/api.json\n"
    "skills:\n  creation_nudge_interval: 0\nmemory:\n  nudge_interval: 0\n"
    'tools:\n  tool_search:\n    enabled: "off"\n'
    "platform_toolsets:\n  api_server: [todo]\n"
    "plugins:\n  enabled:\n    - tenant_context\n"
)


@pytest.mark.asyncio
async def test_request_context_reaches_the_turns_tool_middleware():
    from hermes_cli import plugins

    home = Path(os.environ["HERMES_HOME"])
    plugin = home / "plugins" / "tenant_context"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: tenant_context\nversion: 0.1.0\n", encoding="utf-8"
    )
    (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    turn = [ToolCall("todo_list", {}), Text("done")]
    with FakeLLMServer(turn + turn) as srv:
        write_hermes_home(home, srv.base_url, extra_config=_CONFIG)
        plugins._reset_plugin_managers_for_tests()
        try:
            from gateway.config import PlatformConfig
            from gateway.platforms.api_server import APIServerAdapter

            adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
            app = web.Application()
            app.router.add_post(
                "/v1/chat/completions", adapter._handle_chat_completions
            )
            app.router.add_post("/v1/runs", adapter._handle_runs)
            app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
            async with TestClient(TestServer(app)) as client:
                reply = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Tenant": "acme",
                        "Authorization": "Bearer k",
                        "Cookie": "c=1",
                    },
                    json={
                        "model": "fake-model",
                        "messages": [{"role": "user", "content": "todo?"}],
                    },
                )
                assert reply.status == 200, await reply.text()
                # /v1/runs answers at once and runs the turn on the API worker pool.
                accepted = await client.post(
                    "/v1/runs", headers={"X-Tenant": "globex"}, json={"input": "todo?"}
                )
                assert accepted.status == 202, await accepted.text()
                run_id = (await accepted.json())["run_id"]
                for _ in range(300):
                    run = await (await client.get(f"/v1/runs/{run_id}")).json()
                    if run["status"] not in ("queued", "running"):
                        break
                    await asyncio.sleep(0.05)
                assert run["status"] == "completed", run
            module = plugins.get_plugin_manager()._plugins["tenant_context"].module
        finally:
            plugins._reset_plugin_managers_for_tests()

    # Strings only; the API server's own credentials never reach the hook.
    assert module.SEEN == [
        ("todo_list", {"tenant": "acme"}),
        ("todo_list", {"tenant": "globex"}),
    ]
    assert "x-tenant" in module.HEADERS[0]
    assert not {"authorization", "cookie"} & set(module.HEADERS[0])
    # Outside the turn nothing is bound.
    assert request_context.current() == {}


def _admit(api, text, context=None):
    with request_context.bound(context or {}):
        return admit_api_turn(
            api, session_id="shared", user_message=text, conversation_history=[]
        )


@pytest.mark.asyncio
async def test_each_queued_admission_executes_with_its_own_request_context(api, owner):
    from gateway.session_results import execution_result

    alice = _admit(api, "a", {"user": "alice"})
    _admit(api, "b", {"user": "bob"})
    _admit(api, "c")  # no context, like a turn recovered after a restart
    seen = []

    async def handle(event):
        seen.append(request_context.current())
        execution_result.get()["result"] = {"final_response": "ok"}
        return "ok"

    owner.runner._handle_message = handle
    # The drain runs inside another request's context: no admission may inherit it.
    with request_context.bound({"user": "mallory"}):
        await owner._drain(alice[1])

    assert seen == [{"user": "alice"}, {"user": "bob"}, {}]


@pytest.mark.asyncio
async def test_a_cancelled_admission_releases_its_request_context(api, owner):
    from gateway.session_contract import SessionActor

    authority, ref, row = _admit(api, "a", {"user": "alice"})
    actor = SessionActor(
        "api",
        authority.profile_id,
        frozenset({"session:submit", "session:control"}),
        "api-run:x",
    )
    await authority.cancel_queued(actor, ref, row["admission_id"])

    assert release_request_context(authority, row) == {}
