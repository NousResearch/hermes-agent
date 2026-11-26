"""A multiplexed gateway serves every profile's inbound A2A on ONE listener.

Each served profile gets its own A2A adapter, and before this every secondary bound the same
default port: the first won, the rest died ``bind_failed`` and retried forever. Now a secondary on
the default port mounts at ``/p/<profile>/`` on the listener the process already holds (the
shared-ingress convention), answers with its own adapter and its own credentials, and an explicit
per-profile port still gets its own listener.
"""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request

import pytest

from plugins.platforms.a2a import listeners, protocol


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_is_free(port: int) -> bool:
    """Nothing listens there any more (a refused connect; TIME_WAIT leftovers do not count)."""
    with socket.socket() as s:
        try:
            s.connect(("127.0.0.1", port))
        except ConnectionRefusedError:
            return True
    return False


@pytest.fixture
def shared_port(monkeypatch):
    """The module default port, pointed at a free one (the live gateway may hold 9900)."""
    from plugins.platforms.a2a import adapter as adapter_mod
    for var in ("A2A_PORT", "A2A_BEARER_TOKEN", "A2A_PEER_TOKENS", "A2A_HOST", "A2A_PUBLIC_URL"):
        monkeypatch.delenv(var, raising=False)
    port = _free_port()
    monkeypatch.setattr(adapter_mod, "_DEFAULT_PORT", port)
    return port


def _adapter(profile: str | None = None, extra: dict | None = None, seen: list | None = None):
    """An adapter whose agent echoes; ``profile`` marks it the way the multiplex runner marks a
    secondary before connect (``set_owner_profile``)."""
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    adapter = A2AAdapter(PlatformConfig(enabled=True, extra=dict(extra or {})))
    if profile:
        adapter.set_owner_profile(profile)
    tag = profile or "default"

    async def handle_message(event):
        if seen is not None:
            seen.append(tag)
        await adapter.send(event.source.chat_id, f"{tag} says hi", metadata={"notify": True})

    adapter.handle_message = handle_message  # type: ignore[method-assign]
    adapter._message_handler = object()  # non-None so dispatch proceeds
    return adapter


def _get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _send(url: str, text: str, headers: dict | None = None):
    body = {"jsonrpc": "2.0", "id": "1", "method": "SendMessage",
            "params": {"message": {"role": "ROLE_USER", "parts": [{"text": text}], "messageId": "m-1"}}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _reply_text(resp: dict) -> str:
    task = resp["result"].get("task") or resp["result"]
    return protocol.extract_text(task["artifacts"][0])


class TestSharedListener:
    def test_secondaries_mount_on_the_default_listener(self, shared_port):
        """Default + two secondaries on the default port: all three connect, each path reaches only
        its own profile, and the root lists the mounted profiles."""
        seen: list = []
        root, flume, tolt = _adapter(seen=seen), _adapter("flume-pm", seen=seen), _adapter("tolt-pm", seen=seen)
        base = f"http://127.0.0.1:{shared_port}"

        async def run():
            for a in (root, flume, tolt):
                assert await a.connect() is True, a._fatal_error_message
            try:
                card = await asyncio.to_thread(_get, base + "/p/flume-pm/.well-known/agent-card.json")
                root_card = await asyncio.to_thread(_get, base + "/.well-known/agent-card.json")
                health = await asyncio.to_thread(_get, base + "/")
                sent = await asyncio.to_thread(_send, base + "/p/tolt-pm/", "hello tolt")
                unknown = await asyncio.to_thread(_get, base + "/p/nobody/.well-known/agent-card.json")
                return card, root_card, health, sent, unknown
            finally:
                for a in (tolt, flume, root):
                    await a.disconnect()

        card, root_card, health, sent, unknown = asyncio.run(run())
        assert card[0] == 200 and card[1]["url"] == f"{base}/p/flume-pm/"
        assert root_card[0] == 200 and root_card[1]["url"] == f"{base}/"
        assert health[1]["profiles"] == [
            {"profile": "flume-pm", "url": f"{base}/p/flume-pm/"},
            {"profile": "tolt-pm", "url": f"{base}/p/tolt-pm/"},
        ]
        assert sent[0] == 200 and "tolt-pm says hi" in _reply_text(sent[1])
        assert seen == ["tolt-pm"]  # never the default's (or flume's) session
        assert unknown[0] == 404
        assert _port_is_free(shared_port)  # the last mount out closed the socket

    def test_secondary_first_and_listener_outlives_its_binder(self, shared_port):
        """No default A2A adapter: the first secondary binds, the root is a 404, and the listener
        stays up for the others when the binder disconnects."""
        first, second = _adapter("first-pm"), _adapter("second-pm")
        base = f"http://127.0.0.1:{shared_port}"

        async def run():
            assert await first.connect() is True
            assert await second.connect() is True
            root = await asyncio.to_thread(_get, base + "/")
            await first.disconnect()
            gone = await asyncio.to_thread(_get, base + "/p/first-pm/health")
            alive = await asyncio.to_thread(_send, base + "/p/second-pm/", "still there?")
            await second.disconnect()
            return root, gone, alive

        root, gone, alive = asyncio.run(run())
        assert root[0] == 404
        assert gone[0] == 404
        assert alive[0] == 200 and "second-pm says hi" in _reply_text(alive[1])
        assert _port_is_free(shared_port)

    def test_each_mount_authenticates_with_its_own_credentials(self, shared_port, monkeypatch):
        """Shared socket, separate security: the default requires its token; a token-less secondary
        on the same loopback listener stays localhost-only and needs none."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", "root-token")
        root = _adapter()
        monkeypatch.delenv("A2A_BEARER_TOKEN")
        open_pm = _adapter("open-pm")
        base = f"http://127.0.0.1:{shared_port}"

        async def run():
            assert await root.connect() is True
            assert await open_pm.connect() is True
            try:
                return (await asyncio.to_thread(_send, base + "/", "no token"),
                        await asyncio.to_thread(_send, base + "/", "token", {"Authorization": "Bearer root-token"}),
                        await asyncio.to_thread(_send, base + "/p/open-pm/", "loopback"))
            finally:
                await open_pm.disconnect()
                await root.disconnect()

        anon_root, authed_root, local_open = asyncio.run(run())
        assert anon_root[0] == 401
        assert authed_root[0] == 200
        assert local_open[0] == 200 and "open-pm says hi" in _reply_text(local_open[1])

    def test_explicit_port_keeps_its_own_listener_root(self, shared_port):
        """A secondary whose operator gave it its own port binds that port's root, as before."""
        own_port = _free_port()
        own = _adapter("own-port-pm", extra={"port": own_port})

        async def run():
            assert await own.connect() is True
            try:
                return await asyncio.to_thread(_get, f"http://127.0.0.1:{own_port}/.well-known/agent-card.json")
            finally:
                await own.disconnect()

        status, card = asyncio.run(run())
        assert status == 200 and card["url"] == f"http://127.0.0.1:{own_port}/"
        assert _port_is_free(shared_port)  # never touched the shared port

    def test_duplicate_mount_fails_and_leaves_the_first_serving(self, shared_port):
        """Two roots (or two adapters for one profile) on one port: the second fails bind_failed,
        retryable, and the first keeps serving."""
        first, second = _adapter(), _adapter()
        base = f"http://127.0.0.1:{shared_port}"

        async def run():
            assert await first.connect() is True
            try:
                assert await second.connect() is False
                return await asyncio.to_thread(_get, base + "/health")
            finally:
                await second.disconnect()
                await first.disconnect()

        status, _ = asyncio.run(run())
        assert status == 200
        assert second._fatal_error_code == "bind_failed" and second._fatal_error_retryable
        assert _port_is_free(shared_port)

    def test_port_held_by_another_process_still_fails_bind(self, shared_port):
        with socket.socket() as squatter:
            squatter.bind(("127.0.0.1", shared_port))
            squatter.listen(1)
            adapter = _adapter("late-pm")
            assert asyncio.run(adapter.connect()) is False
            assert adapter._fatal_error_code == "bind_failed"


class TestResolve:
    @pytest.mark.parametrize("raw,mount,path", [
        ("/", "", "/"),
        ("/.well-known/agent-card.json", "", "/.well-known/agent-card.json"),
        ("/p/flume-pm", "flume-pm", "/"),
        ("/p/flume-pm/", "flume-pm", "/"),
        ("/p/flume-pm/.well-known/agent-card.json?x=1", "flume-pm", "/.well-known/agent-card.json?x=1"),
        ("/p/flume-pm/sub/agent/", "flume-pm", "/sub/agent/"),
        ("/p/", "", "/p/"),
    ])
    def test_paths(self, raw, mount, path):
        listener = listeners.Listener(host="127.0.0.1", port=1, server=None,
                                      mounts={"": "root", "flume-pm": "flume"})
        adapter, routed = listeners.resolve(listener, raw)
        assert adapter == listener.mounts[mount]
        assert routed == path

    def test_reserved_served_agent_path(self, shared_port):
        adapter = _adapter(extra={"agents": {"p": {"profile": "x"}, "ok": {"profile": "y"}}})
        assert "p" not in adapter._agents and "ok" in adapter._agents


class TestCallByProfileName:
    def test_a2a_call_resolves_a_served_profile_by_name(self, shared_port, monkeypatch):
        """``a2a_call(agent="<profile>")`` reaches a profile this gateway serves with no
        ``a2a_agents`` entry; the caller never resolves itself; configured peers still win."""
        from plugins.platforms.a2a import tools
        target = _adapter("target-pm")
        base = f"http://127.0.0.1:{shared_port}"

        async def run():
            assert await target.connect() is True
            try:
                monkeypatch.setattr(tools, "_configured_peers", dict)
                found = tools._resolve_peer("target-pm")
                reply = await asyncio.to_thread(tools.a2a_call, {"agent": "target-pm", "message": "ping"})
                monkeypatch.setattr("hermes_cli.profiles.current_profile_name", lambda default=None: "target-pm")
                self_call = tools._resolve_peer("target-pm")
                monkeypatch.setattr(tools, "_configured_peers",
                                    lambda: {"target-pm": {"url": "http://elsewhere.test/"}})
                configured = tools._resolve_peer("target-pm")
                return found, reply, self_call, configured
            finally:
                await target.disconnect()

        found, reply, self_call, configured = asyncio.run(run())
        assert found["url"] == f"{base}/p/target-pm/"
        assert "target-pm says hi" in reply
        assert self_call is None
        assert configured["url"] == "http://elsewhere.test/"
        assert listeners.local_url("target-pm") is None


class TestCallByProfileNameWithToken:
    def test_a_by_name_call_presents_the_callers_bearer_token(self, shared_port, monkeypatch):
        """With ``A2A_BEARER_TOKEN`` set the mount refuses anonymous callers, so a call by profile
        name carries the caller's own token; with no token configured the call stays anonymous."""
        from plugins.platforms.a2a import tools
        monkeypatch.setattr(tools, "_configured_peers", dict)
        monkeypatch.setenv("A2A_BEARER_TOKEN", "fleet-token")
        monkeypatch.setenv("A2A_ALLOW_ALL_USERS", "true")
        target = _adapter("target-pm")

        async def run():
            assert await target.connect() is True
            try:
                peer = tools._resolve_peer("target-pm")
                reply = await asyncio.to_thread(tools.a2a_call, {"agent": "target-pm", "message": "ping"})
                monkeypatch.delenv("A2A_BEARER_TOKEN")
                anonymous = tools._resolve_peer("target-pm")
                return peer, reply, anonymous
            finally:
                await target.disconnect()

        peer, reply, anonymous = asyncio.run(run())
        assert peer["auth"] == {"type": "bearer", "token": "fleet-token"}
        assert "target-pm says hi" in reply
        assert anonymous["auth"] == {}


class TestPerHomePluginCopies:
    def test_two_profile_copies_of_the_plugin_share_one_listener(self, shared_port):
        """The loader imports the plugin once per profile home (``hermes_plugins.<slug>__home_<digest>``),
        so ``adapter.py`` globals are per profile. The registry is reached by its canonical name: two
        copies still meet on one listener instead of the second dying ``bind_failed``."""
        import importlib.util
        import pathlib
        import sys
        import types

        from gateway.config import PlatformConfig

        plugin_dir = pathlib.Path(listeners.__file__).parent
        ns_pkg = types.ModuleType("hermes_plugins_test")  # the loader's namespace parent, as it builds it
        ns_pkg.__path__ = []
        sys.modules["hermes_plugins_test"] = ns_pkg
        copies = []
        for name in ("hermes_plugins_test.a2a_home_one", "hermes_plugins_test.a2a_home_two"):
            spec = importlib.util.spec_from_file_location(
                name, plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)])
            pkg = importlib.util.module_from_spec(spec)
            sys.modules[name] = pkg
            spec.loader.exec_module(pkg)
            adapter_mod = importlib.import_module(f"{name}.adapter")
            adapter_mod._DEFAULT_PORT = shared_port
            copies.append(adapter_mod)
        try:
            one_mod, two_mod = copies
            assert one_mod is not two_mod and one_mod._listeners is two_mod._listeners is listeners
            root = one_mod.A2AAdapter(PlatformConfig(enabled=True, extra={}))
            secondary = two_mod.A2AAdapter(PlatformConfig(enabled=True, extra={}))
            secondary.set_owner_profile("copy-pm")
            for a in (root, secondary):
                a._message_handler = None

            async def run():
                assert await root.connect() is True
                assert await secondary.connect() is True
                try:
                    return await asyncio.to_thread(
                        _get, f"http://127.0.0.1:{shared_port}/p/copy-pm/.well-known/agent-card.json")
                finally:
                    await secondary.disconnect()
                    await root.disconnect()

            status, card = asyncio.run(run())
            assert status == 200 and card["url"].endswith("/p/copy-pm/")
        finally:
            for name in list(sys.modules):
                if name.startswith("hermes_plugins_test"):
                    del sys.modules[name]
