"""``hermes profile create`` through the gateway: the create runs in the caller's profile, refusals keep
their type and full text, an older or unreachable gateway falls back only when no URL names it, an
answer that can't be confirmed is never a local create, and two creates of one name in one process end
in one profile and a clean "already exists"."""
from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import profiles
from hermes_cli.profile_provision import RECEIPT_FIELDS, CreateReceipt, ProfileCreateRequest, provision_profile


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


def test_the_receipt_matches_its_contract():
    from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import ProfileCreateReceipt
    assert tuple(ProfileCreateReceipt.model_fields) == RECEIPT_FIELDS


def test_clone_config_copies_the_callers_profile_and_reports_it(root):
    """``--clone`` sends ``clone_config`` with no ``clone_from``: the gateway copies the profile it runs
    in, and ``report`` returns what the CLI prints."""
    from tui_gateway import server
    (root / "SOUL.md").write_text("I am the root's soul.\n")  # only a clone copies it
    clone = server._methods["profiles.create"](
        1, {"name": "copy", "no_alias": True, "clone_config": True, "report": True})["result"]
    assert "root's soul" in (Path(clone["path"]) / "SOUL.md").read_text()
    made = CreateReceipt.from_wire(clone["receipt"])
    assert made is not None and made.source_label == "default" and made.path == clone["path"]


def test_a_quiet_create_reports_why_seeding_failed(root, monkeypatch):
    def failing(path, quiet=True, problems=None):
        problems.append("⚠ Skill seeding timed out (60s)")
    monkeypatch.setattr(profiles, "seed_profile_skills", failing)
    request = ProfileCreateRequest(name="slow", alias=False)
    assert "Skill seeding timed out (60s)" in CreateReceipt.of(request, provision_profile(request)).warnings


def test_a_refused_create_answers_with_its_type_and_whole_text(root, monkeypatch):
    from hermes_cli import profile_provision
    from hermes_cli.gateway_client import rpc_error
    from hermes_cli.profile_provision import refusal_from
    from tui_gateway import server
    profiles.create_profile("taken", no_alias=True)
    error = server._methods["profiles.create"](1, {"name": "taken", "no_alias": True})["error"]
    assert error["code"] == 4062
    assert error["data"]["type"] == "FileExistsError"

    long_text = "Profile name 'bad name' is invalid. " + "Use lowercase letters, digits, '-' and '_'. " * 8 + "END"
    monkeypatch.setattr(profile_provision, "provision_profile", _raise(ValueError(long_text)))
    error = server._methods["profiles.create"](1, {"name": "other", "no_alias": True})["error"]
    assert error["message"] == long_text  # Desktop shows it whole
    assert str(refusal_from(rpc_error(error).data)) == long_text  # and the CLI gets it whole too

    secret = "sk-proj-" + "X" * 40
    monkeypatch.setattr(profile_provision, "provision_profile",
                        _raise(ValueError("bad config.yaml line: api_key: " + secret)))
    error = server._methods["profiles.create"](1, {"name": "other", "no_alias": True})["error"]
    assert secret not in error["message"] and secret not in str(error["data"])


class _Client:
    def __init__(self, outcome):
        self.outcome = outcome

    async def rpc(self, method, *, _timeout=None, **params):
        assert method == "profiles.create" and params["report"] is True and params["no_alias"] is True
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


def _receipt(**over):
    return {**dict.fromkeys(RECEIPT_FIELDS), "path": "/p/x", "display_path": "~/p/x", "launch_label": "default",
            "source_label": "default", "mirrored": {"env": True}, "warnings": [], "cloned_plugins": [],
            "channels_left_behind": [], "shared_channels": [], **over}


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
    # Rejected by urlsplit (fullwidth colon; bad port), hostless, a password an unencoded "/" pushed into
    # the path, or not a ws URL: never named, never dialed.
    for bad in ("ws://me:pw@gw\uff1a8443/api/ws", "ws://me:pw@gw:99999/api/ws", "wss:me:pw@gw.invalid/api/ws",
                "https://me:pw@gw.invalid/api/ws", "ws://me:pw/cd@gw.invalid/api/ws", "ws://me:12/pw@gw.invalid/api/ws"):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", bad)
        with pytest.raises(GatewayCreateError) as malformed:
            create_through_gateway(request)
        assert "pw" not in str(malformed.value) and "HERMES_TUI_GATEWAY_URL" in str(malformed.value)


def test_the_wrapper_is_written_here_only_for_this_machines_profile(root, monkeypatch):
    """Written when the gateway's profile is the one ``-p x`` selects here (same directory, exists),
    whatever the URL says: a loopback tunnel or container can serve another home."""
    from hermes_cli.profile_provision_gateway import create_through_gateway
    wrote = []
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: wrote.append(name) or Path("/bin") / name)
    request = ProfileCreateRequest(name="x")
    here = root / "profiles" / "x"
    _gateway(monkeypatch, {"ok": True, "receipt": _receipt(path=str(here))})
    assert create_through_gateway(request).alias_path is None  # this path, but nothing there: not ours
    here.mkdir(parents=True)
    assert create_through_gateway(request).alias_path == "/bin/x"  # a discovered local owner made it here
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "wss://gw.invalid/api/ws")
    assert create_through_gateway(request).alias_path == "/bin/x"  # any URL, as long as it landed here
    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://127.0.0.1:9/api/ws?token=t")
    _gateway(monkeypatch, {"ok": True, "receipt": _receipt(path="/elsewhere/.hermes/profiles/x")})
    assert create_through_gateway(request).alias_path is None  # loopback, but another home: no wrapper
    assert wrote == ["x", "x"]

    _gateway(monkeypatch, {"ok": True, "receipt": _receipt(path=str(here))})
    monkeypatch.setattr(profiles, "check_alias_collision", _raise(RuntimeError("memory provider lookup failed")))
    made = create_through_gateway(request)  # the profile exists: a wrapper failure is a warning
    assert made.alias_path is None and made.warnings == ["alias wrapper: memory provider lookup failed"]


def test_refusals_the_cli_never_reports_as_maybe_created(root, monkeypatch):
    """A ``ValueError`` subclass from the create (a broken source config), a missing name and an
    unavailable profile are refusals: nothing was made, so never "may have created"."""
    from hermes_cli.gateway_client import GatewayRPCError, rpc_error
    from hermes_cli.profile_provision import refusal
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    from tui_gateway import server
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
    _gateway(monkeypatch, GatewayRPCError("profile_unavailable", 4064))
    with pytest.raises(GatewayCreateError, match="nothing was created"):
        create_through_gateway(request)
    token = "ghp_" + "A" * 36  # a bounded reason code the client keeps, credential-shaped
    _gateway(monkeypatch, GatewayRPCError(token, -32000))
    with pytest.raises(GatewayCreateError) as untyped:
        create_through_gateway(request)
    assert token not in str(untyped.value)


@pytest.mark.parametrize("bad", [{"skills_seeded": "broken"}, {"skills_seeded": {"copied": 1}},
                                 {"cloned_plugins": [1]}, {"served_now": "false"}, {"display_path": None},
                                 {"mirrored": {"env": "yes"}}, {"alias_path": 5}])
def test_a_receipt_with_a_wrong_type_is_an_unknown_outcome(root, monkeypatch, bad):
    from hermes_cli.profile_provision_gateway import GatewayCreateError, create_through_gateway
    _gateway(monkeypatch, {"ok": True, "receipt": _receipt(**bad)})
    with pytest.raises(GatewayCreateError, match="may have created 'x'"):
        create_through_gateway(ProfileCreateRequest(name="x", alias=False))
