"""Concurrent native CLI preloads keep task-bound homes through SDK serialization.

Explicit-only skills intentionally make this independent of pinned-skill changes.
All provider responses are synthetic; sockets and child processes are forbidden.
"""

import asyncio
import json
import socket
import subprocess
from threading import Barrier

import pytest


def test_concurrent_cli_preload_context(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network or subprocesses in this regression")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    import hermes_cli.env_loader as env_loader
    import hermes_cli.banner as banner
    import hermes_cli.plugins as plugins
    import hermes_cli.mcp_startup as startup

    monkeypatch.setattr(env_loader, "load_hermes_dotenv", lambda *a, **k: [])
    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(startup, "wait_for_mcp_discovery", lambda: None)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    from hermes_constants import (
        get_hermes_home,
        set_hermes_home_override,
        reset_hermes_home_override,
    )

    homes = [tmp_path / "alpha", tmp_path / "beta"]
    cfg = {
        "model": {
            "default": "synthetic-model",
            "provider": "openai",
            "base_url": "http://127.0.0.1:9/v1",
            "context_length": 128000,
        },
        "agent": {"coding_context": "off"},
        "platform_toolsets": {"cli": []},
        "mcp_servers": {},
        "skills": {"auto_load": []},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "compression": {"enabled": False},
        "checkpoints": {"enabled": False},
        "display": {"streaming": False},
    }
    bodies = {}
    for home in homes:
        skill = home / "skills" / "explicit"
        skill.mkdir(parents=True)
        bodies[home] = (
            f"{home.name}: complete first line\n{home.name}: complete last line"
        )
        (skill / "SKILL.md").write_text(
            "---\nname: explicit\ndescription: Synthetic fixture.\n---\n"
            + bodies[home]
            + "\n",
            encoding="utf-8",
        )
        (home / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")

    import cli
    import hermes_cli.cli_agent_setup_mixin as setup
    import httpx

    # Same non-secret process-global policy, different task-bound skill homes.
    monkeypatch.setattr(cli, "CLI_CONFIG", cfg)
    monkeypatch.setattr(cli, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr(
        startup, "ensure_mcp_discovery_before_agent_build", lambda **k: None
    )
    monkeypatch.setattr(cli.HermesCLI, "_ensure_tirith_security", lambda self: None)
    monkeypatch.setattr(cli.HermesCLI, "_ensure_runtime_credentials", lambda self: True)
    monkeypatch.setattr(
        setup,
        "_current_runtime",
        lambda self: {
            "provider": "openai",
            "api_key": "synthetic-not-a-token",
            "base_url": "http://127.0.0.1:9/v1",
            "api_mode": "chat_completions",
        },
    )
    from agent.secret_scope import (
        is_multiplex_active,
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    barrier = Barrier(2, timeout=20)
    requests = {home: [] for home in homes}

    def capture(self, request, **kwargs):
        assert str(request.url) == "http://127.0.0.1:9/v1/chat/completions"
        home = get_hermes_home()
        assert is_multiplex_active()
        requests[home].append(json.loads(request.content))
        barrier.wait()  # Both native SDK requests coexist before either completes.
        chunk = {
            "id": "synthetic",
            "created": 0,
            "model": "synthetic-model",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "SYNTHETIC OK", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
        )

    monkeypatch.setattr(httpx.Client, "send", capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    def launch(home):
        with pytest.raises(SystemExit) as exited:
            cli.main(
                query=f"SYNTHETIC QUERY {home.name}",
                oneshot=True,
                quiet=True,
                model="synthetic-model",
                provider="openai",
                api_key="synthetic-not-a-token",
                base_url="http://127.0.0.1:9/v1",
                skills="explicit",
                max_turns=1,
            )
        assert exited.value.code == 0

    async def run():
        async def task(sequence):
            for home in sequence:
                token = set_hermes_home_override(home)
                secret_token = set_secret_scope({})
                try:
                    await asyncio.to_thread(launch, home)
                    assert get_hermes_home() == home
                finally:
                    reset_secret_scope(secret_token)
                    reset_hermes_home_override(token)
                assert get_hermes_home() == tmp_path

        # Revisit A after B in the same task, with an opposing concurrent lane.
        await asyncio.gather(
            task([homes[0], homes[1], homes[0]]),
            task([homes[1], homes[0], homes[1]]),
        )

    multiplex_before = is_multiplex_active()
    set_multiplex_active(True)
    try:
        asyncio.run(run())
    finally:
        set_multiplex_active(multiplex_before)
    assert all(len(wires) == 3 for wires in requests.values())
    for home, wire in (
        (home, wire) for home, wires in requests.items() for wire in wires
    ):
        prompt = "\n".join(
            m["content"]
            for m in wire["messages"]
            if m["role"] in ("system", "developer")
        )
        assert prompt.count(bodies[home]) == 1
        for other in homes:
            if other != home:
                assert bodies[other] not in prompt
        assert not wire.get("tools")
        assert any(
            m.get("content") == f"SYNTHETIC QUERY {home.name}" for m in wire["messages"]
        )
    assert get_hermes_home() == tmp_path
