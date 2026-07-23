"""Integration harness for single-gateway / multi-agent routing (PR #62944).

Unlike the unit suites (which mock ``resolve_agent_id`` / ``_attach_agent_id`` /
the registry in isolation), these tests drive a **real inbound HTTP request**
through the actual ``APIServerAdapter`` — real routing, real profile/secret
scope, real per-agent credential + home resolution — and observe what the run
*actually saw*.

Observation model
-----------------
The only thing stubbed is the LLM network turn. A spy ``_create_agent`` runs
INSIDE the real profile+agent scope (installed by ``_use_profile_and_secret_scope``
nested in ``_profile_scope``) and captures, at that moment:

* ``get_active_profile().id``   — which agent the run is executing as
* ``get_hermes_home()``         — which per-agent home path getters resolve to
* the SOUL first line           — proves memory/skills/SOUL come from that home
* the resolved LLM api_key       — proves per-agent credential isolation

It records these keyed by a per-request nonce (embedded in the user message) so
concurrent, interleaved requests can be correlated to their own response.
Credential resolution itself is REAL (``_resolve_runtime_agent_kwargs`` under the
scope), so a broken scope surfaces as the wrong key/home, not a passing mock.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

API_KEY = "sk-test-caller"  # global caller bearer (API_SERVER_KEY), not a provider key


# --------------------------------------------------------------------------
# Per-agent home + config builders
# --------------------------------------------------------------------------
def _write_profile(home: Path, agent_id: str, provider_key: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "SOUL.md").write_text(f"I am {agent_id.upper()}. Scope: {agent_id}.\n")
    # distinct provider key per agent home .env — the credential-isolation discriminator
    (home / ".env").write_text(
        f"OPENROUTER_API_KEY={provider_key}\n"
        f"CUSTOM_API_KEY={provider_key}\n"
    )


def build_multi_agent_home(root: Path, agents: dict, *, default_agent="main",
                           multiplex=True) -> dict:
    """Create a HERMES_HOME with per-agent profiles + a multi-agent config.

    *agents*: ``{agent_id: provider_key}``. Returns the parsed config dict.
    """
    profiles_root = root / "profiles"
    cfg_agents = {}
    routes = []
    for aid, key in agents.items():
        home = profiles_root / aid
        _write_profile(home, aid, key)
        cfg_agents[aid] = {"home_dir": str(home)}
        routes.append({"match": {"platform": "api_server", "chat_id": aid}, "agent": aid})
    # The root/process-global value. Two roles: it must NEVER leak into a
    # scoped per-agent read (C), and it IS what a legacy single-agent install
    # legitimately reads (H).
    (root / ".env").write_text("OPENROUTER_API_KEY=sk-ROOT-env\nCUSTOM_API_KEY=sk-ROOT-env\n")

    config = {
        "model": {"default": "echo-model", "provider": "openrouter",
                  "base_url": "http://127.0.0.1:1/v1", "max_tokens": 32},
        "default_agent": default_agent,
        "agents": cfg_agents,
        "routes": routes,
        "gateway": {"multiplex_profiles": multiplex,
                    "api_server": {"max_concurrent_runs": 256}},
    }
    return config


# --------------------------------------------------------------------------
# The aiohttp app (mirrors gateway.platforms.api_server route wiring)
# --------------------------------------------------------------------------
def make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    # Register the REAL route table the adapter exposes in production
    # (``connect()`` uses the same ``_http_route_table()``), not just the two
    # chat endpoints — the session-identity tests need the /api/sessions
    # create/fork/get/chat routes wired exactly as the live server wires them.
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


# --------------------------------------------------------------------------
# Spy agent: captures the live scope context, returns a canned one-turn result
# --------------------------------------------------------------------------
class _SpyAgent:
    def __init__(self, captures: list):
        self._captures = captures
        self.session_id = None
        self.session_prompt_tokens = 1
        self.session_completion_tokens = 1
        self.session_total_tokens = 2

    def run_conversation(self, user_message=None, conversation_history=None,
                         task_id=None, **kw):
        # Executed INSIDE the real profile+agent scope, in the executor thread.
        from agent.profile import get_active_profile
        from agent.secret_scope import get_secret
        from hermes_constants import get_hermes_home

        prof = get_active_profile()
        home = str(get_hermes_home())
        soul = ""
        soul_path = Path(home) / "SOUL.md"
        if soul_path.exists():
            soul = soul_path.read_text().splitlines()[0]
        try:
            key = get_secret("OPENROUTER_API_KEY")
        except Exception as e:  # e.g. UnscopedSecretError
            key = f"<{type(e).__name__}>"
        obs = {
            "nonce": (user_message or "").strip(),
            "agent_id": getattr(prof, "id", None),
            "home": home,
            "soul_first_line": soul,
            "resolved_key": key,
        }
        self._captures.append(obs)
        return {"final_response": json.dumps(obs), "session_id": task_id}


@pytest_asyncio.fixture
async def integ(tmp_path, monkeypatch):
    """Factory: build an adapter wired to a multi-agent home, driving real HTTP
    through ONE shared TestClient (so concurrent requests share the adapter, as
    a real server does).

    Usage::

        env = integ({"coder": "sk-coder", "research": "sk-research"})
        resp = await env.post("coder", "hello")
        assert resp["agent_id"] == "coder"
    """
    from unittest.mock import MagicMock

    envs: list = []

    class _Env:
        def __init__(self, agents, *, default_agent="main", multiplex=True):
            self.captures: list = []
            self.home = tmp_path  # root HERMES_HOME; state.db lives here
            self.config = build_multi_agent_home(
                tmp_path, agents, default_agent=default_agent, multiplex=multiplex)
            monkeypatch.setenv("HERMES_HOME", str(tmp_path))
            from agent.secret_scope import set_multiplex_active
            set_multiplex_active(multiplex)

            pcfg = PlatformConfig(enabled=True, extra={"key": API_KEY})
            self.adapter = APIServerAdapter(pcfg)

            # Wire the real routing context + agent registry from the config.
            from agent.profile import load_agent_registry
            from gateway.config import GatewayConfig
            registry = load_agent_registry(GatewayConfig.from_dict(self.config))
            fake_gw = MagicMock()
            fake_gw._agent_registry = registry
            self.adapter.set_routing_context(
                routes=self.config["routes"],
                default_agent=default_agent,
                gateway=fake_gw,
            )
            # Stub ONLY the agent build/LLM turn; routing + scope + credential
            # resolution around it stay real.
            self.adapter._create_agent = lambda **kw: _SpyAgent(self.captures)
            # Disable the concurrent-run admission limit (read from on-disk
            # config.yaml, which this in-process harness doesn't write) so the
            # concurrency invariant can stress many simultaneous runs. 0 = off.
            self.adapter._max_concurrent_runs = 0
            self._client = None
            self._lock = asyncio.Lock()

        async def _cli(self):
            # Guard against the gather() race where many concurrent posts would
            # each create a client; one shared TestClient serves all requests.
            async with self._lock:
                if self._client is None:
                    self._client = TestClient(TestServer(make_app(self.adapter)))
                    await self._client.start_server()
            return self._client

        async def post(self, chat_id, message, *, extra_headers=None):
            headers = {"Authorization": f"Bearer {API_KEY}",
                       "Content-Type": "application/json"}
            if chat_id is not None:
                headers["X-Hermes-Chat-Id"] = chat_id
            if extra_headers:
                headers.update(extra_headers)
            cli = await self._cli()
            r = await cli.post("/v1/chat/completions", headers=headers, json={
                "model": "echo-model",
                "messages": [{"role": "user", "content": message}],
            })
            body = await r.json()
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                return json.loads(content)
            except Exception:
                return {"raw": body, "status": r.status}

        # --- Session resource API helpers (D — session identity) -----------
        # These drive the REAL /api/sessions/* routes wired in make_app so the
        # persisted-agent invariant can be exercised end-to-end.
        async def _session_request(self, method, path, *, chat_id=None,
                                   extra_headers=None, json_body=None):
            headers = {"Authorization": f"Bearer {API_KEY}",
                       "Content-Type": "application/json"}
            if chat_id is not None:
                headers["X-Hermes-Chat-Id"] = chat_id
            if extra_headers:
                headers.update(extra_headers)
            cli = await self._cli()
            r = await cli.request(method, path, headers=headers,
                                  json=json_body if json_body is not None else {})
            return r, await r.json()

        async def create_session(self, chat_id, *, session_id=None,
                                 extra_headers=None):
            """POST /api/sessions with an optional routing header. Returns the
            client-safe session dict (agent_id is NOT exposed there — read it
            with ``persisted_agent_id``)."""
            body = {"id": session_id} if session_id else {}
            r, data = await self._session_request(
                "POST", "/api/sessions", chat_id=chat_id,
                extra_headers=extra_headers, json_body=body)
            return (data.get("session") or {}), r.status

        async def get_session(self, session_id):
            r, data = await self._session_request(
                "GET", f"/api/sessions/{session_id}")
            return (data.get("session") or {}), r.status

        async def fork_session(self, session_id, *, new_id=None):
            body = {"id": new_id} if new_id else {}
            r, data = await self._session_request(
                "POST", f"/api/sessions/{session_id}/fork", json_body=body)
            return (data.get("session") or {}), r.status

        async def session_chat(self, session_id, message, *,
                               chat_id_header=None, extra_headers=None):
            """POST /api/sessions/{id}/chat. ``chat_id_header`` lets a caller
            send a CONFLICTING X-Hermes-Chat-Id to prove it cannot hijack the
            session's persisted agent. Returns the spy's captured run context."""
            r, data = await self._session_request(
                "POST", f"/api/sessions/{session_id}/chat",
                chat_id=chat_id_header, extra_headers=extra_headers,
                json_body={"message": message})
            content = ((data.get("message") or {}).get("content", "")
                       if isinstance(data, dict) else "")
            try:
                return json.loads(content)
            except Exception:
                return {"raw": data, "status": r.status}

        def persisted_agent_id(self, session_id):
            """Read the agent_id persisted on the session row directly from the
            real SessionDB (state.db under HERMES_HOME) — the ground truth the
            client-safe session view intentionally does not expose."""
            from hermes_state import SessionDB
            db = SessionDB(db_path=Path(self.home) / "state.db")
            row = db.get_session(session_id) or {}
            return row.get("agent_id")

        async def aclose(self):
            if self._client is not None:
                await self._client.close()

    def factory(agents, **kw):
        env = _Env(agents, **kw)
        envs.append(env)
        return env

    yield factory
    for env in envs:
        await env.aclose()
