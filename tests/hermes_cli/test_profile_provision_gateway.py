"""``hermes profile create`` through the gateway: the create runs in the caller's profile, refusals keep
their type and full text, an older or unreachable gateway falls back only when no URL names it, an
answer that can't be confirmed is never a local create, and two creates of one name in one process end
in one profile and a clean "already exists"."""
from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

from typing import Optional

import pytest

from hermes_cli import profiles
from hermes_cli.profile_provision import ProfileCreateReceipt, ProfileCreateRequest, provision_profile


@pytest.fixture()
def root(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    (home / "config.yaml").write_text("model:\n  default: m-root\n  provider: openrouter\n")
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-root\n")
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, quiet=True, problems=None: {"copied": []})
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: None)
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda canon: None)
    return home


def _raise(exc):
    def fail(*_a, **_k):
        raise exc
    return fail


def test_the_contract_carries_the_cli_receipt_itself():
    """One definition: the contract's ``receipt`` is the dataclass the CLI builds and reads back."""
    from typing import Optional
    from hermes_agent.tui_gateway.contracts.profiles_vault_complete_foreign_subagents import ProfilesCreateResult
    assert ProfilesCreateResult.model_fields["receipt"].annotation == Optional[ProfileCreateReceipt]


def test_clone_config_copies_the_callers_profile_and_reports_it(root):
    """``--clone`` sends ``clone_config`` with no ``clone_from``: the gateway copies the profile it runs
    in, and ``report`` returns what the CLI prints."""
    from hermes_agent.tui_gateway import server
    (root / "SOUL.md").write_text("I am the root's soul.\n")  # only a clone copies it
    clone = server._methods["profiles.create"](
        1, {"name": "copy", "no_alias": True, "clone_config": True, "report": True})["result"]
    assert "root's soul" in (Path(clone["path"]) / "SOUL.md").read_text()
    made = ProfileCreateReceipt.from_wire(clone["receipt"])
    assert made is not None and made.source_label == "default" and made.path == clone["path"]
    fresh = server._methods["profiles.create"](1, {"name": "fresh", "no_alias": True, "report": True})["result"]
    assert ProfileCreateReceipt.from_wire(fresh["receipt"]).skills_synced == 0  # the stub copied none


def test_a_quiet_create_reports_why_seeding_failed(root, monkeypatch):
    def failing(path, quiet=True, problems=None):
        problems.append("⚠ Skill seeding timed out (60s)")
    monkeypatch.setattr(profiles, "seed_profile_skills", failing)
    request = ProfileCreateRequest(name="slow", alias=False)
    assert "Skill seeding timed out (60s)" in ProfileCreateReceipt.of(request, provision_profile(request)).warnings


def test_a_refused_create_answers_with_its_type_and_whole_text(root, monkeypatch):
    from hermes_cli import profile_provision
    from hermes_cli.gateway_client import rpc_error
    from hermes_cli.profile_provision import refusal_from
    from hermes_agent.tui_gateway import server
    profiles.create_profile("taken", no_alias=True)
    error = server._methods["profiles.create"](1, {"name": "taken", "no_alias": True})["error"]
    assert error["code"] == 4062
    assert error["data"]["type"] == "FileExistsError"

    long_text = "Profile name 'bad name' is invalid. " + "Use lowercase letters, digits, '-' and '_'. " * 8 + "END"
    monkeypatch.setattr(profile_provision, "provision_profile", _raise(ValueError(long_text)))
    error = server._methods["profiles.create"](1, {"name": "other", "no_alias": True})["error"]
    assert error["message"] == long_text  # Desktop shows it whole
    assert str(refusal_from(rpc_error(error).data)) == long_text  # and the CLI gets it whole too

    huge = "Profile name '" + "n" * 4000 + "' is invalid. Then run `hermes profile create ok`."
    monkeypatch.setattr(profile_provision, "provision_profile", _raise(ValueError(huge)))
    error = server._methods["profiles.create"](1, {"name": "other", "no_alias": True})["error"]
    assert error["message"] == huge
    kept = str(refusal_from(rpc_error(error).data))
    assert kept.endswith(" … [truncated]") and len(kept) == 3200 and kept.startswith("Profile name 'nnn")

    secret = "sk-proj-" + "X" * 40
    monkeypatch.setattr(profile_provision, "provision_profile",
                        _raise(ValueError("bad config.yaml line: api_key: " + secret)))
    error = server._methods["profiles.create"](1, {"name": "other", "no_alias": True})["error"]
    assert secret not in error["message"] and secret not in str(error["data"])


class _Client:
    def __init__(self, outcome, sent, instance):
        self.outcome, self.sent, self.instance = outcome, sent, instance

    async def rpc(self, method, *, _timeout=None, **params):
        if method == "runtime.describe":  # *instance*: an exception, or the id this gateway reports
            if isinstance(self.instance, BaseException):
                raise self.instance
            return {"instance_id": self.instance, "profile_id": "default"}
        assert method == "profiles.create" and params["report"] is True and params["no_alias"] is True
        self.sent.append(params)
        outcome = self.outcome(params) if callable(self.outcome) else self.outcome  # a callable: side effects
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _gateway(monkeypatch, outcome, *, reachable=True, instance="this-home") -> list[dict]:
    """A gateway answering *outcome* (a callable gets the params) and reporting *instance* from
    ``runtime.describe``; this home's discovered owner is ``this-home``. Returns the create params sent."""
    from hermes_cli import gateway_client, gateway_runtime
    sent: list[dict] = []

    @contextlib.asynccontextmanager
    async def connect(endpoint=None):
        if not reachable:
            raise ConnectionRefusedError("refused")
        yield _Client(outcome, sent, instance)
    monkeypatch.setattr(gateway_client, "connect_gateway", connect)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                        lambda home: SimpleNamespace(state="ready", endpoint=SimpleNamespace(instance_id="this-home")))
    return sent


def _receipt(**over):
    return {"path": "/p/x", "display_path": "~/p/x", "launch_label": "default", "source_label": "default",
            "skills_synced": 3, "skills_opted_out": False, "alias_path": None, "alias_collision": None,
            "mirrored": {"env": True}, "warnings": [], "cloned_plugins": [], "memory_provider": None,
            "channels_left_behind": [], "shared_channels": [], "served_now": None, **over}


def test_typed_refusals_and_fallbacks(root, monkeypatch):
    from hermes_cli.gateway_client import GatewayRPCError
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    request = ProfileCreateRequest(name="x", alias=False)
    _gateway(monkeypatch, GatewayRPCError("bad", 4062, {"type": "FileExistsError", "message_parts": ["exists"]}))
    with pytest.raises(FileExistsError, match="exists"):
        create_through_gateway(request)
    # An older local gateway (unknown method / unknown field): create here.
    for code in (-32601, 4000):
        _gateway(monkeypatch, GatewayRPCError("old", code))
        assert create_through_gateway(request) is None
    _gateway(monkeypatch, GatewayRPCError("unknown_method"))  # legacy owners omit the code
    assert create_through_gateway(request) is None
    _gateway(monkeypatch, None, reachable=False)
    assert create_through_gateway(request) is None
    draining = GatewayRPCError("runtime_draining", 4001, {"reason": "runtime_draining"})
    _gateway(monkeypatch, draining)  # refused before routing: nothing made
    assert create_through_gateway(request) is None
    # A gateway named by URL is never replaced by a local create.
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://gw.invalid/api/ws")
    _gateway(monkeypatch, draining)
    with pytest.raises(GatewayCreateError, match="is restarting; retry"):
        create_through_gateway(request)
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


def test_creates_of_one_name_share_a_lock_others_do_not(root, monkeypatch):
    """``create_profile`` stages in ``.<name>.staging-<pid>``: two in-process creates of one name
    (two terminals through one gateway) shared it and the loser died with a raw OSError. The core
    create of a name runs under that name's lock; another name's create does not wait for it."""
    from hermes_cli import profile_provision
    real, held = profiles.create_profile, []

    def create_profile(*a, **k):
        held.append((profile_provision._create_lock("Twin").locked(),
                     profile_provision._create_lock("other").locked()))
        return real(*a, **k)
    monkeypatch.setattr(profiles, "create_profile", create_profile)
    provision_profile(ProfileCreateRequest(name="twin", clone_config=True, alias=False))
    assert held == [(True, False)]
    assert not profile_provision._create_lock("twin").locked()
    with pytest.raises(FileExistsError):
        provision_profile(ProfileCreateRequest(name="twin", clone_config=True, alias=False))


def test_gateway_answers_that_create_nothing_here_or_report_the_unknown(root, monkeypatch):
    from hermes_cli.gateway_client import GatewayRPCError
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    request = ProfileCreateRequest(name="x", alias=False)
    _gateway(monkeypatch, GatewayRPCError("backend is retiring", 5035))  # retiring: nothing was made
    assert create_through_gateway(request) is None
    # Made it, without a complete receipt (older gateway, missing fields, odd shapes): never here too.
    for answer in ({"ok": True, "path": "/p/x"}, {"ok": True, "receipt": {"path": "/p/x", "mirrored": {}}},
                   ["not", "a", "dict"], {"receipt": _receipt(warnings="x")}):
        _gateway(monkeypatch, answer)
        with pytest.raises(GatewayCreateError, match="may have created 'x'"):
            create_through_gateway(request)
    # The handler failed without a typed refusal: it may have failed after the create.
    _gateway(monkeypatch, GatewayRPCError("request_failed", -32000))
    with pytest.raises(GatewayCreateError, match="may have created 'x'"):
        create_through_gateway(request)
    # Anything else after the request went out is an unknown outcome too, never a traceback.
    _gateway(monkeypatch, RuntimeError("decoder broke"))
    with pytest.raises(GatewayCreateError, match="may have created 'x'"):
        create_through_gateway(request)
    password = "hunter2" + "pass"
    _gateway(monkeypatch, {"ok": True, "receipt": _receipt(
        warnings=["model inherit: https://me:" + password + "@proxy.invalid/x refused"])})
    made = create_through_gateway(request)
    assert made.path == "/p/x" and password not in " ".join(made.warnings)
    _gateway(monkeypatch, KeyboardInterrupt())
    with pytest.raises(GatewayCreateError, match="may have created"):
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
    # Rejected by urlsplit (fullwidth colon; bad port, incl. a password an unencoded "/" made the port),
    # hostless, or not a ws URL: never named, never dialed.
    _gateway(monkeypatch, AssertionError("dialed"), reachable=False)
    for bad in ("ws://me:pw@gw\uff1a8443/api/ws", "ws://me:pw@gw:99999/api/ws", "wss:me:pw@gw.invalid/api/ws",
                "https://me:pw@gw.invalid/api/ws", "ws://me:pw/cd@gw.invalid/api/ws"):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", bad)
        with pytest.raises(GatewayCreateError, match="nothing was created") as malformed:
            create_through_gateway(request)
        assert "pw" not in str(malformed.value) and "HERMES_TUI_GATEWAY_URL" in str(malformed.value)
    # Dialable, but not showable safely (an "@" past the authority: a token like an email, or a password
    # an unencoded "/" pushed into the path): dialed as connect_gateway would, named by the variable only.
    for odd in ("wss://gw.invalid/api/ws?token=owner@example.com", "ws://me:12/pw@gw.invalid/api/ws"):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", odd)
        sent = _gateway(monkeypatch, {"ok": True, "receipt": _receipt()})
        assert create_through_gateway(request).path == "/p/x" and len(sent) == 1
        _gateway(monkeypatch, TimeoutError())
        with pytest.raises(GatewayCreateError) as lost:
            create_through_gateway(request)
        assert "HERMES_TUI_GATEWAY_URL gateway" in str(lost.value)
        assert not any(s in str(lost.value) for s in ("owner@", "example", "12", "pw"))


def _make_profile(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text("{}\n")


def _creating(path: Path, reported: Optional[Path] = None):
    """A gateway that makes the profile *path* during the request and reports *reported* (default *path*)."""
    def answer(params):
        _make_profile(path)
        return {"ok": True, "receipt": _receipt(path=str(reported or path))}
    return answer


def test_the_wrapper_is_written_here_only_when_this_create_made_the_local_profile(root, monkeypatch):
    """Written when ``-p x``'s directory here was absent before the request and the gateway made it at that
    path, whatever the URL says: a loopback tunnel or container can serve another home, and another
    machine with the same home layout reports the same path."""
    import shutil
    from hermes_cli.gateway_client import GatewayRPCError
    from hermes_cli.profile_provision_gateway import create_through_gateway
    wrote = []
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: wrote.append(name) or Path("/bin") / name)
    request = ProfileCreateRequest(name="x")
    here = root / "profiles" / "x"
    _gateway(monkeypatch, {"ok": True, "receipt": _receipt(path=str(here))})
    assert create_through_gateway(request).alias_path is None  # this path, but nothing there: not ours
    _gateway(monkeypatch, _creating(here))
    assert create_through_gateway(request).alias_path == "/bin/x"  # a discovered local owner made it here
    shutil.rmtree(here)
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "wss://gw.invalid/api/ws")
    assert create_through_gateway(request).alias_path == "/bin/x"  # any URL, as long as it landed here
    # Another machine, same layout: local x already existed, the gateway reports its own x at that path.
    _gateway(monkeypatch, {"ok": True, "receipt": _receipt(path=str(here))})
    assert create_through_gateway(request).alias_path is None
    # ...or local x is a shell a deleted profile left (no identity), which the other machine never touches.
    shutil.rmtree(here)
    here.mkdir()
    assert create_through_gateway(request).alias_path is None
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://127.0.0.1:9/api/ws?token=t")
    _gateway(monkeypatch, _creating(root.parent / "container-home" / "profiles" / "x"))
    assert create_through_gateway(ProfileCreateRequest(name="y")).alias_path is None  # loopback, another home
    # Another machine, same layout, while a local create of x runs: every path check passes, but the URL's
    # gateway is not this home's owner (another instance, or one that won't say), so no wrapper.
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "wss://gw.invalid/api/ws")
    for instance in ("elsewhere", None, GatewayRPCError("request_failed", 5000)):
        shutil.rmtree(here, ignore_errors=True)
        _gateway(monkeypatch, _creating(here), instance=instance)
        assert create_through_gateway(request).alias_path is None
    # This home's own gateway by URL: written (the loopback case above).
    shutil.rmtree(here)
    _gateway(monkeypatch, _creating(here))
    assert create_through_gateway(request).alias_path == "/bin/x"
    wrote.pop()
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://127.0.0.1:9/api/ws?token=t")
    # Local x appeared during the request (a concurrent local create), but the gateway made its own elsewhere.
    shutil.rmtree(here)
    _gateway(monkeypatch, _creating(here, reported=root.parent / "container-home" / "profiles" / "x"))
    assert create_through_gateway(request).alias_path is None
    assert wrote == ["x", "x"]
    # A deleted profile's leftover shell is replaced by the create: this create made x, so its command is written.
    shutil.rmtree(here)
    here.mkdir()
    _gateway(monkeypatch, _creating(here))
    assert create_through_gateway(request).alias_path == "/bin/x"
    assert wrote == ["x", "x", "x"]

    shutil.rmtree(here)
    _gateway(monkeypatch, _creating(here))
    monkeypatch.setattr(profiles, "check_alias_collision", _raise(RuntimeError("memory provider lookup failed")))
    made = create_through_gateway(request)  # the profile exists: a wrapper failure is a warning
    assert made.alias_path is None and made.warnings == ["alias wrapper: memory provider lookup failed"]


def test_a_url_gateway_creates_in_this_terminals_profile(root, monkeypatch):
    """A URL's connection carries no profile-bound ticket: the terminal's profile is named, so ``-p work``
    copies work, not the gateway's launch profile. A profile it can't serve refuses before any create."""
    from hermes_cli.gateway_client import rpc_error
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    from hermes_agent.tui_gateway import ws_legacy_fallback
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "wss://gw.invalid/api/ws")
    request = ProfileCreateRequest(name="x", alias=False, clone_config=True)
    sent = _gateway(monkeypatch, {"ok": True, "receipt": _receipt()})
    create_through_gateway(request)
    work = root / "profiles" / "work"
    work.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(work))
    create_through_gateway(request)
    assert [params["profile"] for params in sent] == ["default", "work"]
    _gateway(monkeypatch, rpc_error(ws_legacy_fallback._unavailable({"id": 1})["error"]))
    with pytest.raises(GatewayCreateError, match=r"cannot serve this terminal's profile \(work\); nothing was created"):
        create_through_gateway(request)
    import hermes_constants
    with monkeypatch.context() as patched:
        patched.setattr(hermes_constants, "profile_name_for_home", lambda home: None)  # no name resolves
        with pytest.raises(GatewayCreateError, match="not a named profile.*nothing was created"):
            create_through_gateway(request)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL")
    monkeypatch.setenv("HERMES_HOME", str(root))
    sent = _gateway(monkeypatch, {"ok": True, "receipt": _receipt()})
    create_through_gateway(request)
    assert "profile" not in sent[0]  # a local owner's ticket already names it


def test_refusals_the_cli_never_reports_as_maybe_created(root, monkeypatch):
    """A ``ValueError`` subclass from the create (a broken source config), a missing name and a typed
    refusal with no text are refusals: nothing was made, so never "may have created"."""
    from hermes_cli.gateway_client import GatewayRPCError, rpc_error
    from hermes_cli.profile_provision import refusal
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    from hermes_agent.tui_gateway import server
    request = ProfileCreateRequest(name="x", alias=False)
    import json
    try:
        json.loads("{")
    except json.JSONDecodeError as exc:
        _gateway(monkeypatch, rpc_error({"code": 4062, "message": str(exc), "data": refusal(exc)[1]}))
    with pytest.raises(ValueError, match="Expecting property name"):
        create_through_gateway(request)
    blank = server._methods["profiles.create"](1, {"name": "  "})["error"]
    _gateway(monkeypatch, rpc_error(blank))
    with pytest.raises(ValueError, match="profile name cannot be empty"):
        create_through_gateway(request)
    _gateway(monkeypatch, GatewayRPCError("bad", 4062, {"type": "FileNotFoundError"}))  # typed, no text
    with pytest.raises(FileNotFoundError, match="nothing was created"):
        create_through_gateway(request)
    token = "ghp_" + "A" * 36  # a bounded reason code the client keeps, credential-shaped
    _gateway(monkeypatch, GatewayRPCError(token, -32000))
    with pytest.raises(GatewayCreateError) as untyped:
        create_through_gateway(request)
    assert token not in str(untyped.value)


@pytest.mark.parametrize("bad", [{"skills_synced": "3"}, {"skills_synced": True}, {"skills_opted_out": None},
                                 {"cloned_plugins": [1]}, {"served_now": "false"}, {"display_path": None},
                                 {"mirrored": {"env": "yes"}}, {"alias_path": 5}, {"path": ""}])
def test_a_receipt_with_a_wrong_type_is_an_unknown_outcome(root, monkeypatch, bad):
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    _gateway(monkeypatch, {"ok": True, "receipt": _receipt(**bad)})
    with pytest.raises(GatewayCreateError, match="may have created 'x'"):
        create_through_gateway(ProfileCreateRequest(name="x", alias=False))


def test_a_newer_gateways_extra_receipt_fields_are_dropped(root, monkeypatch):
    from hermes_cli.profile_provision_gateway import create_through_gateway
    _gateway(monkeypatch, {"ok": True, "receipt": _receipt(mirrored={"env": True, "skills": True}, extra=[1])})
    made = create_through_gateway(ProfileCreateRequest(name="x", alias=False))
    assert made.mirrored == {"env": True, "skills": True} and not hasattr(made, "extra")


@pytest.mark.parametrize("where", ["create", "receipt"])
def test_a_config_outage_mid_create_is_an_unknown_outcome(root, monkeypatch, where):
    """``ConfigBackendUnavailable`` is a ``SystemExit``: past ``except Exception`` it reached the
    dispatcher's 4064 ("profile unavailable"), read as nothing created although the profile existed."""
    from hermes_cli import profile_provision
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from hermes_cli.gateway_client import rpc_error
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    from hermes_agent.tui_gateway.server import handle_request  # the real dispatcher (rebound onto server)
    if where == "create":
        monkeypatch.setattr(profile_provision, "provision_profile", _raise(ConfigBackendUnavailable("backend down")))
    else:  # the profile is published; reading what the CLI prints fails
        monkeypatch.setattr(ProfileCreateReceipt, "of", _raise(ConfigBackendUnavailable("backend down")))
    error = handle_request({"jsonrpc": "2.0", "id": 1, "method": "profiles.create",
                            "params": {"name": "w2", "no_alias": True, "report": True}})["error"]
    assert error["code"] == 5062
    _gateway(monkeypatch, rpc_error(error))
    with pytest.raises(GatewayCreateError, match="may have created 'w2'"):
        create_through_gateway(ProfileCreateRequest(name="w2", alias=False))
    _gateway(monkeypatch, rpc_error(error))
    with pytest.raises(GatewayCreateError, match="backend down"):  # the reason, not "request_failed"
        create_through_gateway(ProfileCreateRequest(name="w2", alias=False))
    _gateway(monkeypatch, rpc_error({"code": 4064, "message": "profile unavailable"}))  # never "nothing made"
    with pytest.raises(GatewayCreateError, match="may have created 'w2'"):
        create_through_gateway(ProfileCreateRequest(name="w2", alias=False))


@pytest.mark.parametrize("reason", ["profile_mismatch", "profile_unavailable", "runtime_draining", "unknown_method"])
def test_a_handler_failure_named_like_a_routing_refusal_is_an_unknown_outcome(root, monkeypatch, reason):
    """Routing refusals are matched by code AND ``data.reason``: a handler failure (5062) whose text or
    reason reads like one, after the profile was made, is never "nothing was created", "retry" or a
    local create, wherever the gateway is."""
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from hermes_cli.gateway_client import rpc_error
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    from hermes_agent.tui_gateway.server import handle_request
    monkeypatch.setattr(ProfileCreateReceipt, "of", _raise(ConfigBackendUnavailable(reason)))
    real = handle_request({"jsonrpc": "2.0", "id": 1, "method": "profiles.create",
                           "params": {"name": "w3", "no_alias": True, "report": True}})["error"]
    assert real["code"] == 5062 and (root / "profiles" / "w3").is_dir()
    worst = {"code": 5062, "message": reason, "data": {"reason": reason}}  # the reason survives rpc_error
    assert str(rpc_error(worst)) == reason
    for error in (real, worst):
        for url in ("", "ws://gw.invalid/api/ws"):
            monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", url)
            _gateway(monkeypatch, rpc_error(error))
            with pytest.raises(GatewayCreateError, match="may have created 'w3'"):
                create_through_gateway(ProfileCreateRequest(name="w3", alias=False))


def test_a_create_with_no_gateway_url_loads_no_gateway_platforms_or_pydantic():
    """Every ``hermes profile create`` runs this module: naming a URL needs ``gateway.platforms`` (~225
    modules) and reading a gateway's receipt needs pydantic, so neither loads on the way to a local create."""
    import subprocess
    import sys
    probe = ("import sys; from hermes_cli.profile_provision_gateway import _where; "
             "from hermes_cli.profile_provision import ProfileCreateReceipt; _where(''); "
             "print([m for m in ('gateway.platforms', 'pydantic') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[2])
    assert out.stdout.strip() == "[]"
