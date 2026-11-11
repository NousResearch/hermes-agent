"""``hermes profile create`` through the gateway: the create runs in the caller's profile, refusals keep
their type, an older or unreachable gateway falls back only when no URL names it, and two creates of
one name in one process end in one profile and a clean "already exists"."""
from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import profiles
from hermes_cli.profile_provision import ProfileCreateRequest, provision_profile


@pytest.fixture()
def root(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    (home / "config.yaml").write_text("model:\n  default: m-root\n  provider: openrouter\n")
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-root\n")
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, quiet=True: {"copied": []})
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: None)
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda canon: None)
    return home


def test_clone_config_copies_the_callers_profile_and_reports_it(root):
    """``--clone`` sends ``clone_config`` with no ``clone_from``: the gateway copies the profile it runs
    in, and ``report`` returns what the CLI prints."""
    from tui_gateway import server
    (root / "SOUL.md").write_text("I am the root's soul.\n")  # only a clone copies it
    clone = server._methods["profiles.create"](
        1, {"name": "copy", "no_alias": True, "clone_config": True, "report": True})["result"]
    assert "root's soul" in (Path(clone["path"]) / "SOUL.md").read_text()
    assert clone["receipt"]["source_label"] == "default"
    assert clone["receipt"]["path"] == clone["path"]


def test_a_refused_create_answers_with_its_type(root, monkeypatch):
    from hermes_cli import profile_provision
    from tui_gateway import server
    profiles.create_profile("taken", no_alias=True)
    error = server._methods["profiles.create"](1, {"name": "taken", "no_alias": True})["error"]
    assert error["code"] == 4062
    assert error["data"]["type"] == "FileExistsError"

    def leaky(request):  # a parse error quoting the source's config line
        raise ValueError("bad config.yaml line: api_key: sk-proj-abcdefghijklmnopqrstuvwx1234")
    monkeypatch.setattr(profile_provision, "provision_profile", leaky)
    error = server._methods["profiles.create"](1, {"name": "other", "no_alias": True})["error"]
    assert "abcdefghijklmnop" not in error["data"]["message"] and "abcdefghijklmnop" not in error["message"]


class _Client:
    def __init__(self, outcome):
        self.outcome = outcome

    async def rpc(self, method, *, _timeout=None, **params):
        assert method == "profiles.create" and params["report"] is True
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _gateway(monkeypatch, outcome, *, reachable=True):
    from hermes_cli import gateway_client, gateway_runtime

    @contextlib.asynccontextmanager
    async def connect(endpoint=None):
        if not reachable:
            raise ConnectionRefusedError("refused")
        yield _Client(outcome)
    monkeypatch.setattr(gateway_client, "connect_gateway", connect)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                        lambda home: SimpleNamespace(state="ready", endpoint=object()))


def test_typed_refusals_and_fallbacks(root, monkeypatch):
    from hermes_cli.gateway_client import GatewayRPCError
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    request = ProfileCreateRequest(name="x")
    _gateway(monkeypatch, GatewayRPCError("bad", 4062, {"type": "FileExistsError", "message": "exists"}))
    with pytest.raises(FileExistsError, match="exists"):
        create_through_gateway(request)
    # An older local gateway (unknown method / unknown field): create here.
    for code in (-32601, 4000):
        _gateway(monkeypatch, GatewayRPCError("old", code))
        assert create_through_gateway(request) is None
    _gateway(monkeypatch, None, reachable=False)
    assert create_through_gateway(request) is None
    # A gateway named by URL is never replaced by a local create.
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://gw.invalid/api/ws?token=secret-in-url")
    _gateway(monkeypatch, GatewayRPCError("old", -32601))
    with pytest.raises(GatewayCreateError, match="predates"):
        create_through_gateway(request)
    _gateway(monkeypatch, None, reachable=False)
    with pytest.raises(GatewayCreateError) as unreachable:
        create_through_gateway(request)
    assert "secret-in-url" not in str(unreachable.value)
    _gateway(monkeypatch, TimeoutError())
    with pytest.raises(GatewayCreateError, match="check `hermes profile list`"):
        create_through_gateway(request)


def test_the_core_create_holds_the_process_lock(root, monkeypatch):
    """``create_profile`` stages in ``.<name>.staging-<pid>``: two in-process creates of one name
    (two terminals through one gateway) shared it and the loser died with a raw OSError. The core
    create runs under one process lock, released before seeding / the alias."""
    from hermes_cli import profile_provision
    real, held = profiles.create_profile, []

    def create_profile(*a, **k):
        held.append(profile_provision._CREATE_LOCK.locked())
        return real(*a, **k)
    monkeypatch.setattr(profiles, "create_profile", create_profile)
    provision_profile(ProfileCreateRequest(name="twin", clone_config=True, alias=False))
    assert held == [True]
    assert not profile_provision._CREATE_LOCK.locked()
    with pytest.raises(FileExistsError):
        provision_profile(ProfileCreateRequest(name="twin", clone_config=True, alias=False))


def test_gateway_answers_that_create_nothing_here_or_report_the_unknown(root, monkeypatch):
    from hermes_cli.gateway_client import GatewayRPCError
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    request = ProfileCreateRequest(name="x")
    _gateway(monkeypatch, GatewayRPCError("backend is retiring", 5035))  # retiring: nothing was made
    assert create_through_gateway(request) is None
    _gateway(monkeypatch, {"ok": True, "path": "/p/x"})  # made it, but no receipt: never create here too
    with pytest.raises(GatewayCreateError, match="predates this version"):
        create_through_gateway(request)
    _gateway(monkeypatch, {"ok": True, "receipt": {"path": "/p/x", "mirrored": {"env": True}}})
    made = create_through_gateway(request)
    assert made["path"] == "/p/x" and made["cloned_plugins"] == [] and made["served_now"] is None
    _gateway(monkeypatch, KeyboardInterrupt())
    with pytest.raises(GatewayCreateError, match="may still finish"):
        create_through_gateway(request)
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "wss://me:pw@gw.invalid:8443/api/ws?token=t0k")
    _gateway(monkeypatch, None, reachable=False)
    with pytest.raises(GatewayCreateError) as unreachable:
        create_through_gateway(request)
    assert "wss://gw.invalid:8443" in str(unreachable.value)
    assert "pw" not in str(unreachable.value) and "t0k" not in str(unreachable.value)
    _gateway(monkeypatch, TimeoutError())  # recovery is checked where the profile was made
    with pytest.raises(GatewayCreateError, match="on the gateway's machine"):
        create_through_gateway(request)
    # Rejected by urlsplit (fullwidth colon; bad port) or hostless (authority lands in the path): unnamed.
    _gateway(monkeypatch, None, reachable=False)
    for bad in ("ws://me:pw@gw\uff1a8443/api/ws", "ws://me:pw@gw:99999/api/ws", "wss:me:pw@gw.invalid/api/ws"):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", bad)
        with pytest.raises(GatewayCreateError) as malformed:
            create_through_gateway(request)
        assert "pw" not in str(malformed.value) and "HERMES_TUI_GATEWAY_URL" in str(malformed.value)
