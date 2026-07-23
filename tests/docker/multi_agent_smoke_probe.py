"""In-container probe for the multi-agent single-gateway smoke (T2).

Run as a REAL separate OS process INSIDE the container (``docker exec python3
/host_repo/tests/docker/multi_agent_smoke_probe.py``). The worktree is mounted
at ``/host_repo`` and prepended to ``sys.path`` so this exercises the FEATURE
code under test (the cached image predates it), not the image's baked-in copy.

What it proves across the real process boundary
-----------------------------------------------
Two agents (``coder``/``research``) are routed through the REAL
``APIServerAdapter`` over REAL aiohttp HTTP. Only the LLM turn is stubbed: the
spy runs INSIDE the real profile+secret scope, in the real ``run_in_executor``
thread, and WRITES a ``run_dump.json`` to ``get_hermes_home()``. Because the
scope redirects the home per agent, coder's dump lands under
``profiles/coder/`` and research's under ``profiles/research/`` — an on-disk
artifact proving the ContextVar profile+secret scope propagated across the real
executor-thread boundary in a real process. The host test reads those files.

Exit 0 on success; non-zero (with a diagnostic on stderr) otherwise.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# The mounted worktree (feature code) must win over the image's /opt/hermes.
HOST_REPO = os.environ.get("HOST_REPO", "/host_repo")
sys.path.insert(0, HOST_REPO)

HOME = Path(os.environ["HERMES_HOME"])
API_KEY = "sk-smoke-caller"
AGENTS = {"coder": "sk-coder-smoke", "research": "sk-research-smoke"}


def _build_home() -> dict:
    profiles_root = HOME / "profiles"
    cfg_agents, routes = {}, []
    for aid, key in AGENTS.items():
        home = profiles_root / aid
        home.mkdir(parents=True, exist_ok=True)
        (home / "SOUL.md").write_text(f"I am {aid.upper()}. Scope: {aid}.\n")
        (home / ".env").write_text(f"OPENROUTER_API_KEY={key}\nCUSTOM_API_KEY={key}\n")
        cfg_agents[aid] = {"home_dir": str(home)}
        routes.append({"match": {"platform": "api_server", "chat_id": aid}, "agent": aid})
    (HOME / ".env").write_text("OPENROUTER_API_KEY=sk-ROOT-env\nCUSTOM_API_KEY=sk-ROOT-env\n")
    return {
        "model": {"default": "echo-model", "provider": "openrouter",
                  "base_url": "http://127.0.0.1:1/v1", "max_tokens": 32},
        "default_agent": "main",
        "agents": cfg_agents,
        "routes": routes,
        "gateway": {"multiplex_profiles": True,
                    "api_server": {"max_concurrent_runs": 256}},
    }


class _SpyAgent:
    """Captures the live scope and persists it to the scoped home on disk."""

    def __init__(self):
        self.session_id = None
        self.session_prompt_tokens = 1
        self.session_completion_tokens = 1
        self.session_total_tokens = 2

    def run_conversation(self, user_message=None, conversation_history=None,
                         task_id=None, **kw):
        from agent.profile import get_active_profile
        from agent.secret_scope import get_secret
        from hermes_constants import get_hermes_home

        prof = get_active_profile()
        home = Path(get_hermes_home())
        try:
            key = get_secret("OPENROUTER_API_KEY")
        except Exception as e:  # noqa: BLE001
            key = f"<{type(e).__name__}>"
        soul_path = home / "SOUL.md"
        soul = soul_path.read_text().splitlines()[0] if soul_path.exists() else ""
        obs = {
            "nonce": (user_message or "").strip(),
            "agent_id": getattr(prof, "id", None),
            "home": str(home),
            "soul_first_line": soul,
            "resolved_key": key,
        }
        # The on-disk artifact under the SCOPED home — the thing the host asserts.
        (home / "run_dump.json").write_text(json.dumps(obs))
        return {"final_response": json.dumps(obs), "session_id": task_id}


async def _amain() -> int:
    from unittest.mock import MagicMock

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from agent.profile import load_agent_registry
    from agent.secret_scope import set_multiplex_active
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    config = _build_home()
    set_multiplex_active(True)

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
    registry = load_agent_registry(GatewayConfig.from_dict(config))
    fake_gw = MagicMock()
    fake_gw._agent_registry = registry
    adapter.set_routing_context(routes=config["routes"], default_agent="main",
                                gateway=fake_gw)
    adapter._create_agent = lambda **kw: _SpyAgent()
    adapter._max_concurrent_runs = 0

    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for aid in AGENTS:
            r = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}",
                         "X-Hermes-Chat-Id": aid,
                         "Content-Type": "application/json"},
                json={"model": "echo-model",
                      "messages": [{"role": "user", "content": f"{aid}-nonce"}]},
            )
            body = await r.json()
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            obs = json.loads(content)
            if obs.get("agent_id") != aid:
                print(f"PROBE-FAIL: {aid} ran as {obs.get('agent_id')}", file=sys.stderr)
                return 2
    finally:
        await client.close()
    print("PROBE-OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_amain()))
