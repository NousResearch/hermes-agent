"""``resolve-profile`` gives a client the verdict ``discover_gateway_endpoint`` gives, over the real socket.

Each case runs a real control server, asks it ``resolve-profile`` the way a client following the
documented rendezvous would (the profile's own socket, else its multiplexer root's), and compares
the answer with Python discovery of the same home on the same live socket.
"""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from gateway.control_resolve import resolve_profile_handler
from gateway.control_socket import GatewayControlServer
from gateway.runtime_bootstrap import TicketStore
from hermes_cli.gateway_client import private_control_query
from hermes_cli.gateway_runtime import discover_gateway_endpoint, ensure_gateway_runtime

pytestmark = pytest.mark.platforms("linux", "macos")


def _descriptor(*homes: Path, instance="instance-a", state="ready", **extra) -> dict:
    return {"runtime_protocol": 1, "state": state, "instance_id": instance, "authority_epoch": 2,
            "api_origin": "http://127.0.0.1:8765", "supervisor": "none", "code_sha": "c0de" * 10,
            "capabilities": ["session-authority-v1"],
            "served_profiles": [{"home": str(h), "profile_id": str(h)} for h in homes], **extra}


@pytest.fixture()
def layout(tmp_path):
    root = tmp_path / "r" / ".hermes"
    secondary = root / "profiles" / "work"
    secondary.mkdir(parents=True, mode=0o700)
    (root / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    solo = tmp_path / "solo"
    solo.mkdir(mode=0o700)
    return solo.resolve(), root.resolve(), secondary.resolve()


def _server(home: Path, descriptor: dict) -> GatewayControlServer:
    server = GatewayControlServer(home, verb_handlers={"identify": lambda: dict(descriptor)})
    server.ticket_store = TicketStore(descriptor["instance_id"], frozenset(p["home"] for p in descriptor["served_profiles"]))
    server.resolve_handler = resolve_profile_handler(lambda: dict(descriptor))
    return server


def _ask(socket_home: Path, verb: str, params: dict) -> dict:
    line = json.dumps({"protocol": 1, "id": 1, "verb": verb, "params": params}).encode() + b"\n"
    return json.loads(private_control_query(socket_home, line))


def _discovered(verdict) -> dict:
    endpoint = None
    if verdict.endpoint is not None:
        endpoint = {**asdict(verdict.endpoint), "capabilities": sorted(verdict.endpoint.capabilities)}
    return {"state": verdict.state, "reason_code": verdict.reason_code, "detail": verdict.detail, "endpoint": endpoint}


def _resolved(answer: dict) -> dict:
    assert answer["ok"] is True, answer
    result = answer["result"]
    endpoint = result["endpoint"]
    if endpoint is not None:
        endpoint = {k: v for k, v in endpoint.items() if k not in {"ws_url", "ws_protocol", "ticket_protocol_prefix"}}
    return {"state": result["state"], "reason_code": result["reason_code"], "detail": result["detail"], "endpoint": endpoint}


def _parity(server_home: Path, descriptor: dict, client_home: Path, *, before=None):
    """``(discovery verdict, resolve result)`` for *client_home* against one live server."""
    server = _server(server_home, descriptor)

    async def scenario():
        assert await server.start()
        try:
            if before is not None:
                before()
            verdict = await asyncio.to_thread(discover_gateway_endpoint, client_home)
            answer = await asyncio.to_thread(_ask, server_home, "resolve-profile", {"home": str(client_home)})
            return verdict, answer
        finally:
            await server.stop()

    verdict, answer = asyncio.run(scenario())
    assert _resolved(answer) == _discovered(verdict)
    return verdict, answer["result"]


def test_a_running_owner_resolves_as_discovery_finds_it(layout):
    solo, _root, _secondary = layout
    verdict, result = _parity(solo, _descriptor(solo), solo)
    assert verdict.state == "ready" and result["control"] == "self" and result["attachable"] is True
    assert result["endpoint"]["ws_url"] == "ws://127.0.0.1:8765/api/ws"
    assert result["update_fenced"] is False and result["message"] is None


def test_a_served_secondary_is_answered_by_the_host_socket(layout):
    _solo, root, secondary = layout
    verdict, result = _parity(root, _descriptor(root, secondary), secondary)
    assert verdict.endpoint is not None and verdict.endpoint.control_home == str(root)
    assert result["control"] == "host" and result["endpoint"]["profile_id"] == str(secondary)


@pytest.mark.parametrize("descriptor_extra, expected", [
    ({}, ("absent", None)),
    ({"parked_profiles": {"work": "duplicate Telegram token"}}, ("inaccessible", "profile_parked")),
])
def test_an_unserved_or_parked_secondary_matches_discovery(layout, descriptor_extra, expected):
    _solo, root, secondary = layout
    verdict, result = _parity(root, _descriptor(root, **descriptor_extra), secondary)
    assert (verdict.state, verdict.reason_code) == expected
    assert result["control"] == "host" and result["attachable"] is False


@pytest.mark.parametrize("state, drain_reason, attachable", [
    ("starting", None, False), ("draining", "restart_wait", True), ("draining", "idle_exit", False)])
def test_owner_states_match_discovery(layout, state, drain_reason, attachable):
    solo, _root, _secondary = layout
    extra = {"drain_reason": drain_reason} if drain_reason else {}
    verdict, result = _parity(solo, _descriptor(solo, state=state, **extra), solo)
    assert verdict.state == state and result["attachable"] is attachable


def test_a_secondary_whose_own_owner_is_starting_matches_discovery(layout):
    """A standalone secondary that holds its lock but has not published a socket yet."""
    from gateway.runtime_ownership import ProfileOwnership
    _solo, root, secondary = layout
    owner = ProfileOwnership()
    owner.reserve([secondary])
    try:
        verdict, result = _parity(root, _descriptor(root), secondary)
    finally:
        owner.close()
    assert verdict.state == "starting" and result["control"] == "host"


def test_an_absent_or_stale_owner_has_no_socket_to_ask(layout):
    """Nothing answers: discovery reports the lock verdict and a client's dial fails as \"no gateway\"."""
    solo, _root, _secondary = layout
    assert discover_gateway_endpoint(solo).state == "absent"
    with pytest.raises(FileNotFoundError):
        _ask(solo, "resolve-profile", {"home": str(solo)})
    import socket
    from gateway.control_socket import resolve_server_socket_path
    bind_path, pointer = resolve_server_socket_path(solo)  # where a gateway for this home listens
    if pointer is not None:
        bind_path.parent.mkdir(mode=0o700, exist_ok=True)
        pointer.write_text(str(bind_path), encoding="utf-8")
        pointer.chmod(0o600)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
            stale.bind(str(bind_path))  # the node a crashed gateway left behind: nobody listens
        bind_path.chmod(0o600)
        assert discover_gateway_endpoint(solo).state == "absent"
        with pytest.raises(ConnectionRefusedError):
            _ask(solo, "resolve-profile", {"home": str(solo)})
    finally:
        bind_path.unlink(missing_ok=True)
        if pointer is not None:
            bind_path.parent.rmdir()


def test_update_fenced_is_exactly_when_ensure_does_not_return_discovery(layout):
    """``attachable and not update_fenced`` is the case where ``ensure`` hands back the discovery
    verdict unchanged, which is what lets a client skip the start command."""
    solo, _root, _secondary = layout
    marker = solo / ".hermes-update-in-progress"
    for fenced in (False, True):
        if fenced:
            marker.touch()  # an update still writing its marker fences for its grace period
        server = _server(solo, _descriptor(solo))

        async def scenario():
            assert await server.start()
            try:
                answer = await asyncio.to_thread(_ask, solo, "resolve-profile", {"home": str(solo)})
                discovered = await asyncio.to_thread(discover_gateway_endpoint, solo)
                ensured = await asyncio.to_thread(ensure_gateway_runtime, solo, timeout=5)
                return answer["result"], discovered, ensured
            finally:
                await server.stop()

        result, discovered, ensured = asyncio.run(scenario())
        assert result["update_fenced"] is fenced and result["attachable"] is True
        assert (ensured == discovered) is (not fenced)
        if fenced:
            assert (ensured.state, ensured.reason_code) == ("draining", "update_paused")


def test_a_restarted_owner_refuses_the_old_instance_and_resolves_the_new_one(layout):
    solo, _root, _secondary = layout
    before = _descriptor(solo, instance="instance-a")
    after = _descriptor(solo, instance="instance-b")
    _first, old = _parity(solo, before, solo)
    server = _server(solo, after)

    async def scenario():
        assert await server.start()
        try:
            refused = await asyncio.to_thread(_ask, solo, "session-ticket", {
                "profile_id": old["endpoint"]["profile_id"], "instance_id": old["endpoint"]["instance_id"],
                "purpose": "interactive"})
            renewed = await asyncio.to_thread(_ask, solo, "resolve-profile", {"home": str(solo)})
            discovered = await asyncio.to_thread(discover_gateway_endpoint, solo)
            return refused, renewed, discovered
        finally:
            await server.stop()

    refused, renewed, discovered = asyncio.run(scenario())
    assert refused == {"ok": False, "error": "PermissionError: stale runtime instance", "protocol": 1, "id": 1}
    assert _resolved(renewed) == _discovered(discovered)
    assert renewed["result"]["endpoint"]["instance_id"] == "instance-b"


def test_a_symlinked_spelling_of_the_home_resolves_to_the_served_profile_id(layout, tmp_path):
    """The client sends its home as it knows it; ``profile_id`` comes back in the owner's spelling,
    which is what ``session-ticket`` compares byte for byte."""
    solo, _root, _secondary = layout
    alias = tmp_path / "alias"
    alias.symlink_to(solo)
    _verdict, result = _parity(solo, _descriptor(solo), alias)
    assert result["endpoint"]["profile_id"] == str(solo)


def _tree(top: Path) -> dict:
    """Every node under *top*: kind, mode and content (or link target), without following links."""
    import os
    import stat as st
    nodes = {}
    for dirpath, dirnames, filenames in os.walk(top):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            info = path.lstat()
            body = (os.readlink(path) if st.S_ISLNK(info.st_mode)
                    else path.read_bytes() if st.S_ISREG(info.st_mode) else None)
            nodes[str(path)] = (st.S_IFMT(info.st_mode), st.S_IMODE(info.st_mode), body)
    return nodes


def _resolve_line(server_home: Path, home: str) -> dict:
    """``resolve-profile`` straight through the dispatcher, as from a peer with no credentials."""
    server = GatewayControlServer(server_home)
    server.resolve_handler = resolve_profile_handler(lambda: _descriptor(server_home))
    return json.loads(server.handle_request_line(json.dumps(
        {"protocol": 1, "id": 1, "verb": "resolve-profile", "params": {"home": home}}).encode(), None))


@pytest.fixture()
def foreign(tmp_path):
    """A home outside every root: a same-user owner whose upgraded lock is still 0644."""
    owner = tmp_path / "foreign" / "owner"
    owner.mkdir(parents=True, mode=0o700)
    (owner / "gateway.lock").write_text('{"pid": 1}')
    (owner / "gateway.lock").chmod(0o644)
    return owner


def _escaping_profile(root: Path, foreign: Path) -> str:
    """``<root>/profiles/evil``, a symlink that leads out of the root to an overlay of *foreign*."""
    overlay = foreign.parent / "overlay"
    overlay.mkdir(mode=0o700)
    (overlay / "gateway.lock").symlink_to(foreign / "gateway.lock")
    (root / "profiles" / "evil").symlink_to(overlay)
    return str(root / "profiles" / "evil")


def _foreign_overlay(root: Path, foreign: Path) -> str:
    overlay = foreign.parent / "overlay"
    overlay.mkdir(mode=0o700)
    (overlay / "gateway.lock").symlink_to(foreign / "gateway.lock")
    return str(overlay)


def _overlay_profile(root: Path, foreign: Path) -> str:
    """An in-root profile whose ``gateway.lock`` is a link to the foreign owner's lock."""
    (root / "profiles" / "work" / "gateway.lock").symlink_to(foreign / "gateway.lock")
    return str(root / "profiles" / "work")


@pytest.mark.parametrize("make_home", [_foreign_overlay, _escaping_profile, _overlay_profile],
                         ids=["foreign-overlay", "in-root-symlink-escaping-root", "in-root-lock-linking-out"])
def test_a_home_outside_the_root_is_refused_before_anything_there_is_touched(layout, foreign, make_home):
    """Containment is decided before any overlay lock is followed: a request finally answered
    ``profile_mismatch`` must not have narrowed (or read through) a lock outside the root."""
    _solo, root, _secondary = layout
    home = make_home(root, foreign)
    before = _tree(foreign.parent)
    answer = _resolve_line(root, home)
    assert answer["ok"] is True, answer
    assert (answer["result"]["reason_code"], answer["result"]["control"]) == ("profile_mismatch", None)
    assert _tree(foreign.parent) == before
    assert (foreign / "gateway.lock").stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize("spell", [
    lambda h: f"{h}/../{h.name}", lambda h: f"{h}/.", lambda h: f"{h.parent}//{h.name}",
    lambda h: f"/{h}", lambda h: f"{h}/", lambda h: h.name,
], ids=["dot-dot", "dot", "duplicate-slash", "leading-double-slash", "trailing-slash", "relative"])
def test_a_walking_spelling_is_invalid_and_reads_nothing(layout, spell):
    """Only a plain absolute spelling is checked against the root; anything else is refused as a
    bad parameter before any path is resolved."""
    solo, _root, _secondary = layout
    (solo / "gateway.lock").write_text("{}")
    (solo / "gateway.lock").chmod(0o644)
    before = _tree(solo)
    answer = _resolve_line(solo, spell(solo))
    assert answer == {"ok": False, "error": "PermissionError: invalid resolve parameters", "protocol": 1, "id": 1}
    assert _tree(solo) == before


def test_an_allowed_home_changes_exactly_what_discovery_changes(layout):
    """Inside the root the lookup does what discovery does on main and nothing more: an unserved
    secondary's upgraded 0644 lock is narrowed to 0600 by both, an in-root overlay is followed."""
    _solo, root, secondary = layout
    other = root / "profiles" / "other"
    other.mkdir(mode=0o700)
    for home in (secondary, other):
        (home / "gateway.lock").write_text("{}")
        (home / "gateway.lock").chmod(0o644)
    discovered = discover_gateway_endpoint(other)  # no socket anywhere: the lock verdict
    answer = _resolve_line(root, str(secondary))
    assert (answer["result"]["state"], discovered.state) == ("absent", "absent")
    assert _tree(secondary) == {k.replace(str(other), str(secondary)): v for k, v in _tree(other).items()}
    overlay = root / "profiles" / "overlay"
    overlay.mkdir(mode=0o700)
    (overlay / "gateway.lock").symlink_to(secondary / "gateway.lock")
    followed = _resolve_line(root, str(overlay))["result"]
    assert (followed["control"], followed["state"]) == ("host", "absent")


def test_a_server_without_the_handler_keeps_the_unknown_verb_answer(layout):
    solo, _root, _secondary = layout
    server = GatewayControlServer(solo)
    line = server.handle_request_line(json.dumps(
        {"protocol": 1, "id": 1, "verb": "resolve-profile", "params": {"home": str(solo)}}).encode())
    assert json.loads(line) == {"ok": False, "error": "unknown verb: 'resolve-profile'", "protocol": 1,
                                "supported_verbs": ["identify", "status"], "id": 1}


def test_the_gateway_bootstrap_installs_the_verb_over_its_identify_descriptor(tmp_path, monkeypatch):
    """The listener ``start_gateway`` brings up answers ``resolve-profile`` from the runtime
    descriptor its ``identify`` verb answers with."""
    from types import SimpleNamespace
    from gateway.run_bootstrap import _start_gateway_start_control_socket

    home = tmp_path / "boot"
    home.mkdir(mode=0o700)
    home = home.resolve()
    monkeypatch.setenv("HERMES_HOME", str(home))
    descriptor = _descriptor(home, instance="instance-boot")
    runner = SimpleNamespace(session_runtime_descriptor=descriptor, _draining=False,
                             session_ticket_store=TicketStore("instance-boot", frozenset({str(home)})),
                             config=SimpleNamespace(multiplex_profiles=False))

    async def scenario():
        server = await _start_gateway_start_control_socket(runner)
        assert server is not None
        try:
            answer = await asyncio.to_thread(_ask, home, "resolve-profile", {"home": str(home)})
            verdict = await asyncio.to_thread(discover_gateway_endpoint, home)
            return answer, verdict
        finally:
            await server.stop()

    answer, verdict = asyncio.run(scenario())
    assert _resolved(answer) == _discovered(verdict)
    assert answer["result"]["endpoint"]["instance_id"] == "instance-boot"


def test_install_code_sha_leaves_the_gateways_boot_identity_alone(layout, monkeypatch):
    """``identify``'s ``code_sha`` is the code this gateway booted from: reading the installed commit
    for ``install_code_sha`` must not drop the cached identity it is answered from."""
    from hermes_cli import version_info
    solo, _root, _secondary = layout
    boot = version_info.VersionInfo("1.0.0", "1.0.0", None, "b" * 40, None, "git")
    monkeypatch.setattr(version_info, "_cached_version_info", boot)
    monkeypatch.setattr(version_info, "_stamp_or_none", lambda: None)
    monkeypatch.setattr(version_info, "_resolve_repo_dir", lambda: solo)
    monkeypatch.setattr(version_info, "_run_git", lambda repo, *args: "a" * 40)
    _verdict, result = _parity(solo, _descriptor(solo), solo)
    assert result["install_code_sha"] == "a" * 40
    assert version_info.get_code_identity()["sha"] == "b" * 40


@pytest.mark.parametrize("stamp, repo, expected", [
    ("s" * 40, "repo", "s" * 40),   # a packaged install's stamp wins
    (None, "repo", "a" * 40),       # a source checkout answers from git
    (None, None, None),             # neither: no installed commit to name
])
def test_install_code_sha_and_read_checkout_sha_share_one_lookup(monkeypatch, stamp, repo, expected):
    """The gateway's ``install_code_sha`` and the CLI's ``read_checkout_sha`` come from one reader,
    so they cannot drift apart on which commit is installed."""
    from gateway import control_resolve
    from hermes_cli import version_info
    monkeypatch.setattr(version_info, "_stamp_or_none",
                        lambda: version_info.VersionInfo("1.0.0", "1.0.0", None, stamp, None, "stamp") if stamp else None)
    monkeypatch.setattr(version_info, "_resolve_repo_dir", lambda: repo)
    monkeypatch.setattr(version_info, "_run_git", lambda repo_dir, *args: "a" * 40)
    assert control_resolve._install_code_sha() == version_info.read_checkout_sha() == expected
    monkeypatch.setattr(version_info, "installed_commit_sha", lambda: "shared")
    assert control_resolve._install_code_sha() == "shared"
