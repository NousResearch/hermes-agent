"""Client launch must not silently downgrade policy or own execution."""
import argparse

import pytest


def test_unsupported_launch_options_fail_before_connection(monkeypatch, capsys):
    from hermes_cli import gateway_chat
    calls = []
    monkeypatch.setattr(gateway_chat, "connect_gateway", lambda: calls.append(True))
    for option in ("image", "worktree", "run_budget"):
        args = argparse.Namespace(**{option: True})
        assert gateway_chat.launch_from_args(args) == 2
        assert option.replace("_", "-") in capsys.readouterr().err
    # A nameless --create-if-missing stays refused (bare -c resolves on the owner, see
    # test_gateway_chat_flags_live.py).
    assert gateway_chat.launch_from_args(argparse.Namespace(create_if_missing=True)) == 2
    assert "create-if-missing" in capsys.readouterr().err
    # Creation flags on resume are judged against the frozen route: repeating the launch
    # flags is fine (scripts re-run one command line), changing one is refused.
    frozen = {"info": {"launch_request": {"source": "cli", "model": "m", "toolsets": ["file"], "reasoning": "high"},
                       "cwd": "/w"}}
    gateway_chat.check_resume_policy(argparse.Namespace(resume="stored", model="m", toolsets="file",
                                                        reasoning="high", query="x"), frozen)
    for override in ({"model": "other"}, {"source": "tui"}, {"toolsets": "browser"}, {"in_dir": "/elsewhere"}):
        with pytest.raises(gateway_chat.GatewayClientError):
            gateway_chat.check_resume_policy(argparse.Namespace(resume="stored", query="x", **override), frozen)
    # Bypass launches read no profile default model, so one must be explicit.
    assert gateway_chat.launch_from_args(argparse.Namespace(safe_mode=True, query="x")) == 1
    assert "--model" in capsys.readouterr().err
    assert calls == []


def test_refusals_name_the_replacement_and_a_runnable_safe_mode_example(monkeypatch, capsys):
    """An exit-2 refusal tells the user where every refused flag went; the safe-mode
    refusal prints a command they can run as-is."""
    from hermes_cli import gateway_chat
    monkeypatch.setattr(gateway_chat, "connect_gateway", lambda: pytest.fail("connected"))
    assert gateway_chat.launch_from_args(argparse.Namespace(worktree=True, run_budget=30.0)) == 2
    err = capsys.readouterr().err
    assert "--worktree: use" in err and "hermes --tui -w" in err
    assert "--run-budget: use" in err and "run_budget_seconds" in err
    for name in gateway_chat._UNSUPPORTED:
        assert name in gateway_chat._RELOCATED, f"{name} refused without saying where it went"
    assert gateway_chat.launch_from_args(argparse.Namespace(safe_mode=True, query="x")) == 1
    err = capsys.readouterr().err
    example = next(line.split("Example: ", 1)[1] for line in err.splitlines() if "Example: " in line)
    import shlex
    from hermes_cli._parser import build_top_level_parser
    parsed = build_top_level_parser()[0].parse_args(shlex.split(example)[1:])
    assert parsed.safe_mode and parsed.model and parsed.provider and parsed.query


@pytest.mark.asyncio
async def test_creation_preserves_advertised_cwd_model_and_toolsets(monkeypatch, tmp_path):
    from contextlib import asynccontextmanager
    from hermes_cli import gateway_chat
    from hermes_cli.gateway_chat_view import GatewayChatView
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append((method, params))
            if method == "runtime.describe":
                return {"session_create": {"sources": ["cli"], "parameters": [
                    "cwd", "model", "toolsets", "request_id", "source", "skills", "checkpoints", "accept_hooks",
                    "pass_session_id"]}}
            return {"stored_session_id": "stored"}

    @asynccontextmanager
    async def connected():
        yield Peer()

    async def rendered(self, query, *, oneshot):
        assert query == "literal"
        return 0

    monkeypatch.setattr(gateway_chat, "connect_gateway", connected)
    monkeypatch.setattr(GatewayChatView, "run", rendered)
    monkeypatch.chdir(tmp_path)
    # `-s a,b -s a` + the session-scoped launch flags + an exported HERMES_ACCEPT_HOOKS=1 ride the
    # create (the classic client is a gateway client; nothing is refused or dropped).
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")
    args = argparse.Namespace(query="literal", model="explicit-model", toolsets="terminal, file", quiet=True,
                              skills=["a,b", "a"], checkpoints=True, pass_session_id=True)
    assert await gateway_chat.run_gateway_chat(args) == 0
    create = calls[1][1]
    assert create["cwd"] == str(tmp_path)
    assert create["model"] == "explicit-model"
    assert create["toolsets"] == ["terminal", "file"]
    assert create["skills"] == ["a", "b"] and create["checkpoints"] is create["pass_session_id"] is True
    assert create["accept_hooks"] is True
    assert create["source"] == "cli" and create["request_id"]


@pytest.mark.asyncio
async def test_finite_runs_create_oneshot_sessions_and_explicit_source_wins(monkeypatch, tmp_path):
    """Finite ``-q``/``-z`` runs are stored as ``oneshot`` (kept out of human pickers); ``--source``
    always wins, and a gateway that does not advertise ``oneshot`` still gets a plain ``cli`` run."""
    from contextlib import asynccontextmanager
    from hermes_cli import gateway_chat
    from hermes_cli.gateway_chat_view import GatewayChatView
    created, sources = [], ["cli", "tool", "oneshot"]

    class Peer:
        async def rpc(self, method, **params):
            if method == "runtime.describe":
                return {"session_create": {"sources": sources, "parameters": ["cwd", "request_id", "source"]}}
            created.append(params["source"])
            return {"stored_session_id": "stored"}

    @asynccontextmanager
    async def connected():
        yield Peer()

    async def rendered(self, query, *, oneshot):
        return 0

    monkeypatch.setattr(gateway_chat, "connect_gateway", connected)
    monkeypatch.setattr(GatewayChatView, "run", rendered)
    monkeypatch.chdir(tmp_path)
    for args in (argparse.Namespace(query="q", quiet=True), argparse.Namespace(oneshot="z"),
                 argparse.Namespace(query="q", quiet=True, source="tool")):
        assert await gateway_chat.run_gateway_chat(args) == 0
    sources.remove("oneshot")
    assert await gateway_chat.run_gateway_chat(argparse.Namespace(query="q", quiet=True)) == 0
    assert created == ["oneshot", "oneshot", "tool", "cli"]


@pytest.mark.asyncio
async def test_rpc_preserves_notifications_and_errors():
    import asyncio
    import json
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect
    from hermes_cli.gateway_client import GatewayClient, GatewayClientError

    async def peer(ws):
        async for raw in ws:
            request = json.loads(raw)
            await ws.send(json.dumps({"method": "message.complete", "params": {"text": "reply"}}))
            await ws.send(json.dumps({"id": request["id"], "error": {"message": "stale_generation"}}))

    async with serve(peer, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            async with GatewayClient(ws) as client:
                with pytest.raises(GatewayClientError, match="stale_generation"):
                    await client.rpc("session.interrupt", session_id="stored", execution_generation=4)
                event = await asyncio.wait_for(client.events.get(), 2)
                assert event["params"]["text"] == "reply"


@pytest.mark.asyncio
async def test_yolo_slash_toggles_the_session_bypass_on_the_owner(capsys):
    """`/yolo` in an attached `hermes chat` was refused as an unsupported command, so a `--yolo`
    launch could not be revoked from the classic CLI; it is the owner's session-scoped config.set."""
    from hermes_cli.gateway_chat_view import GatewayChatView
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append((method, params))
            return {"key": "yolo", "value": params.get("value", "0"), "scope": "session"}

    view = GatewayChatView(Peer(), {"stored_session_id": "sid"})
    assert await view.command("/yolo off") is True
    assert await view.command("/yolo") is True
    assert calls == [("config.set", {"session_id": "sid", "key": "yolo", "value": "0"}),
                     ("config.set", {"session_id": "sid", "key": "yolo"})]
    assert "YOLO off for this session" in capsys.readouterr().out


def test_every_relocation_hint_names_a_command_that_parses():
    """A refusal that points at `hermes --tui --checkpoints` / `hermes --tui -v` must not send the
    user to a command argparse rejects."""
    import re
    import shlex
    from hermes_cli import gateway_chat
    from hermes_cli.main import _build_cli_parser
    parser = _build_cli_parser()
    parser = parser[0] if isinstance(parser, tuple) else parser
    for hint in gateway_chat._RELOCATED.values():
        for command in re.findall(r"`(hermes [^`]*)`", hint):
            parser.parse_args([arg for arg in shlex.split(command)[1:] if not arg.startswith("<")])


def test_one_letter_alias_refusal_names_a_flag_the_user_can_type(monkeypatch, capsys):
    """`cli.main(w=True)` (`python cli.py -w`) is refused as `-w`; `--w` is not an option anywhere."""
    from hermes_cli import gateway_chat
    monkeypatch.setattr(gateway_chat, "connect_gateway", lambda: pytest.fail("connected"))
    assert gateway_chat.launch_from_args(argparse.Namespace(w=True)) == 2
    err = capsys.readouterr().err
    assert "options: -w." in err and "\n  -w: use `hermes --tui -w`" in err and "--w" not in err


@pytest.mark.asyncio
async def test_documented_env_opt_ins_ride_the_classic_create_and_bypass_needs_a_model(monkeypatch, tmp_path, capsys):
    """N24: the auto-started owner no longer inherits the environment, so HERMES_YOLO_MODE /
    HERMES_IGNORE_RULES / HERMES_SAFE_MODE / HERMES_IGNORE_USER_CONFIG / HERMES_ACCEPT_HOOKS
    must ride session.create (Ink's localCreationOptions set), and an env bypass without -m is
    refused before connecting, exactly as the flag is."""
    from contextlib import asynccontextmanager
    from hermes_cli import gateway_chat
    from hermes_cli.gateway_chat_view import GatewayChatView
    flags = ["yolo", "ignore_rules", "safe_mode", "ignore_user_config", "accept_hooks"]
    created = []

    class Peer:
        async def rpc(self, method, **params):
            if method == "runtime.describe":
                return {"session_create": {"sources": ["cli"], "parameters": [
                    "cwd", "model", "request_id", "source", *flags]}}
            created.append(params)
            return {"stored_session_id": "stored"}

    @asynccontextmanager
    async def connected():
        yield Peer()

    async def rendered(self, query, *, oneshot):
        return 0

    monkeypatch.setattr(gateway_chat, "connect_gateway", connected)
    monkeypatch.setattr(GatewayChatView, "run", rendered)
    monkeypatch.chdir(tmp_path)
    for name in ("HERMES_YOLO_MODE", "HERMES_IGNORE_RULES", "HERMES_SAFE_MODE", "HERMES_ACCEPT_HOOKS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "true")
    assert await gateway_chat.run_gateway_chat(argparse.Namespace(query="q", quiet=True, model="m")) == 0
    assert all(created[0].get(flag) is True for flag in flags), created[0]
    monkeypatch.setattr(gateway_chat, "connect_gateway", lambda: pytest.fail("connected"))
    for name in ("HERMES_YOLO_MODE", "HERMES_IGNORE_RULES", "HERMES_SAFE_MODE", "HERMES_ACCEPT_HOOKS"):
        monkeypatch.delenv(name)
    assert gateway_chat.launch_from_args(argparse.Namespace(query="q")) == 1
    assert "--model" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_launch_key_flags_are_never_silently_dropped(monkeypatch, tmp_path, capsys):
    """N23: `--tui` cannot carry --api-key/--base-url/--reasoning into Ink's create yet, so it
    refuses them (exit 2) instead of running on the profile's endpoint; classic chat re-supplies
    the launch key on resume (an owner restart revoked it) and `/new` keeps it."""
    from contextlib import asynccontextmanager
    from hermes_cli import gateway_chat, gateway_chat_commands, main as main_mod
    from hermes_cli.gateway_chat_view import GatewayChatView
    monkeypatch.setattr(main_mod, "_launch_tui", lambda *_a, **_k: pytest.fail("TUI launched without the flags"))
    for flag in ("api_key", "base_url", "reasoning"):
        with pytest.raises(SystemExit) as refused:
            main_mod.cmd_chat(argparse.Namespace(tui=True, **{flag: "x"}))
        assert refused.value.code == 2 and "--" + flag.replace("_", "-") in capsys.readouterr().err
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append((method, params))
            if method == "runtime.describe":
                return {"session_create": {"sources": ["cli"], "parameters": ["cwd", "model", "api_key", "source"]}}
            if method == "session.info":
                return {"launch_request": {"source": "cli", "model": "m"}, "cwd": str(tmp_path)}
            return {"stored_session_id": "stored", "execution_generation": 0,
                    "info": {"launch_request": {"source": "cli", "model": "m"}}}

    @asynccontextmanager
    async def connected():
        yield Peer()

    async def rendered(self, query, *, oneshot):
        await gateway_chat_commands._new(self, "")
        return 0

    monkeypatch.setattr(gateway_chat, "connect_gateway", connected)
    monkeypatch.setattr(GatewayChatView, "run", rendered)
    args = argparse.Namespace(resume="stored", api_key="launch-key", query="q", quiet=True)
    assert await gateway_chat.run_gateway_chat(args) == 0
    sent = {method: params for method, params in calls}
    assert sent["session.resume"] == {"session_id": "stored", "api_key": "launch-key"}
    assert sent["session.create"]["api_key"] == "launch-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/new", "/branch"])
async def test_new_and_branch_move_the_terminal_continue_pointer(monkeypatch, tmp_path, command):
    """P3: a bare `hermes -c` after `/new` or `/branch` continues the session this terminal is
    now on (the in-process CLI re-wrote its breadcrumb on every switch), not the old one."""
    from hermes_cli import gateway_chat_commands, terminal_breadcrumbs
    from hermes_cli.gateway_chat_view import GatewayChatView
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TMUX_PANE", "%9")
    monkeypatch.setattr(terminal_breadcrumbs.os, "ttyname", lambda fd: (_ for _ in ()).throw(OSError()))

    class Peer:
        async def rpc(self, method, **params):
            if method == "runtime.describe":
                return {"session_create": {"sources": ["cli"], "parameters": ["cwd", "source"]}}
            if method == "session.info":
                return {"launch_request": {"source": "cli"}, "cwd": str(tmp_path)}
            if method == "session.mutate":
                return {"status": "applied", "branched_session_id": "moved"}
            return {"stored_session_id": "moved", "execution_generation": 0, "info": {}}

    view = GatewayChatView(Peer(), {"stored_session_id": "old", "execution_generation": 0, "info": {}}, quiet=True)
    terminal_breadcrumbs.write_breadcrumb("old")

    async def apply(client, original, operation, payload, confirm=None):
        return await client.rpc("session.mutate")

    monkeypatch.setattr(view.mutations, "apply", apply)
    if command == "/new":
        await gateway_chat_commands._new(view, "")
    else:
        assert await view.command(command) is True
    assert terminal_breadcrumbs.read_breadcrumb()["session_id"] == "moved"
