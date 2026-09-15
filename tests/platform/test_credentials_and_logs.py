"""Credential entry and runtime log reading.

Credential entry is the one path in NOVA that writes a secret, and it was deliberately
absent until an administrator asked for it. Most of what follows is about the things that
had to be true before it could exist, so if any of them regress the suite says so loudly:

* ``.env`` stays on the materialiser's ``NEVER_WRITE`` list, so ``apply`` still cannot
  overwrite a credential;
* only names the tenant's own declaration asks for may be written, because ``.env`` is
  loaded into the environment of the process that runs the agent;
* no value comes back out — not through a read, not through the audit log.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from nova.apply import apply_bundle
from nova.audit import AuditLog
from nova.control import ControlAPI
from nova.control.auth import Principal
from nova.credentials import check_writable, slots_for_agent
from nova.errors import SpecError
from nova.runtime import get_runtime
from nova.spec import load_bundle

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "nova" / "examples" / "acme"
ADMIN = Principal(name="ops", role="admin")
VIEWER = Principal(name="watcher", role="viewer")

SECRET = "123:A-VERY-DISTINCTIVE-SECRET-VALUE"


@pytest.fixture
def live(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    shutil.copytree(EXAMPLE, root)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NOVA_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))

    bundle = load_bundle(root)
    runtime = get_runtime("hermes", home=home, tenant_id=bundle.tenant_id)
    audit = AuditLog.for_home(home, tenant_id=bundle.tenant_id, actor="test")
    apply_bundle(bundle, runtime, audit=audit, dry_run=False)
    api = ControlAPI(bundle, runtime, audit=audit)
    return {"api": api, "home": home, "root": root, "bundle": bundle, "runtime": runtime}


def _write(api, path, principal, payload):
    return api.write(f"/platform/v1{path}", principal, payload)


def _env(home: Path, agent: str = "operations") -> Path:
    return home / "profiles" / agent / ".env"


def _audit_text(home: Path) -> str:
    return (home / "nova" / "audit.jsonl").read_text()


# -- the allowlist -------------------------------------------------------------


def test_only_declared_credentials_are_writable(live):
    """The control that stops "set a token" becoming "run my code".

    `.env` is loaded into the environment of the process that runs the agent, so a write
    path accepting any name could set LD_PRELOAD, PYTHONPATH or BASH_ENV.
    """
    for hostile in ("LD_PRELOAD", "PYTHONPATH", "PATH", "BASH_ENV", "NODE_OPTIONS"):
        response = _write(live["api"], "/agents/operations/credentials", ADMIN,
                          {"values": {hostile: "/tmp/evil"}})
        assert response.status == 400, f"{hostile} was accepted"
        assert hostile in response.body["error"]["message"]
    assert not _env(live["home"]).exists(), "a refused write created the file anyway"


def test_the_allowlist_comes_from_the_tenants_own_declaration(live):
    names = {s.name for s in slots_for_agent(live["bundle"], "operations")}
    # Telegram grants this agent, so its manifest's variables are writable.
    assert "TELEGRAM_BOT_TOKEN" in names
    # The deployment's model credential is writable.
    assert "ACME_LLM_KEY" in names
    # A platform the tenant has not connected is not.
    assert "SLACK_BOT_TOKEN" not in names


def test_revoking_a_channel_grant_revokes_its_credentials(live, tmp_path):
    """The allowlist is derived per request, so it follows the declaration rather than
    lagging behind it."""
    import yaml

    # Re-point the grant rather than emptying it: a connection granting nobody is refused
    # at load, because "empty" could mean no agent or every agent and one of those is a
    # channel into the whole workforce.
    path = live["root"] / "channels.yaml"
    document = yaml.safe_load(path.read_text())
    for channel in document.get("channels", []):
        channel["allowed_agents"] = ["customer-support"]
        for route in channel.get("routes", []) or []:
            route["agent"] = "customer-support"
    path.write_text(yaml.safe_dump(document))

    revoked = load_bundle(live["root"])
    assert "TELEGRAM_BOT_TOKEN" not in {s.name for s in slots_for_agent(revoked, "operations")}
    with pytest.raises(SpecError, match="TELEGRAM_BOT_TOKEN"):
        check_writable(revoked, "operations", ["TELEGRAM_BOT_TOKEN"])


# -- writing -------------------------------------------------------------------


def test_a_credential_is_written_owner_only(live):
    response = _write(live["api"], "/agents/operations/credentials", ADMIN,
                      {"values": {"TELEGRAM_BOT_TOKEN": SECRET}})
    assert response.status == 200, response.body
    assert response.body["changed"] == ["TELEGRAM_BOT_TOKEN"]

    env = _env(live["home"])
    assert SECRET in env.read_text()
    assert oct(env.stat().st_mode & 0o777) == "0o600", "a credential file was group or world readable"


def test_a_value_never_comes_back_out(live):
    _write(live["api"], "/agents/operations/credentials", ADMIN,
           {"values": {"TELEGRAM_BOT_TOKEN": SECRET}})

    read = json.dumps(live["api"].handle("/platform/v1/agents/operations/credentials").body)
    assert SECRET not in read
    assert '"set": true' in read.lower().replace("'", '"')

    # And not through any other read either — the whole surface is checked rather than the
    # one route that obviously carries it.
    for route in ("/agents", "/agents/operations/config", "/channels", "/policy", "/health",
                  "/agents/operations/activity"):
        body = json.dumps(live["api"].handle(f"/platform/v1{route}").body)
        assert SECRET not in body, f"{route} returned a credential value"


def test_a_value_never_reaches_the_audit_log(live):
    _write(live["api"], "/agents/operations/credentials", ADMIN,
           {"values": {"TELEGRAM_BOT_TOKEN": SECRET}})
    text = _audit_text(live["home"])
    assert SECRET not in text, "the audit log recorded a credential value"
    # But the act itself is recorded, with the human who did it.
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    creds = [r for r in records if r.get("kind") == "agent.credentials_changed"]
    assert [r["phase"] for r in creds] == ["intent", "committed"]
    assert {r["actor"] for r in creds} == {"ops"}
    assert "TELEGRAM_BOT_TOKEN" in json.dumps(creds), "the audit does not say what changed"


def test_apply_cannot_overwrite_a_credential(live):
    """`.env` stays on NEVER_WRITE. Credential entry did not weaken that — it added a
    separate path, so a configuration push still cannot clobber a secret."""
    _write(live["api"], "/agents/operations/credentials", ADMIN,
           {"values": {"TELEGRAM_BOT_TOKEN": SECRET}})
    apply_bundle(
        load_bundle(live["root"]), live["runtime"],
        audit=AuditLog.for_home(live["home"], tenant_id=live["bundle"].tenant_id, actor="t"),
        dry_run=False,
    )
    assert SECRET in _env(live["home"]).read_text()

    from nova.runtime.hermes.materialize import NEVER_WRITE

    assert ".env" in NEVER_WRITE


def test_an_edit_preserves_the_rest_of_the_file(live):
    """The file is the operator's. A write through the control plane must not reformat it
    or drop the parts NOVA does not model."""
    env = _env(live["home"])
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text("# my notes\nUNMANAGED_THING=keep\n")
    _write(live["api"], "/agents/operations/credentials", ADMIN,
           {"values": {"TELEGRAM_BOT_TOKEN": SECRET}})
    text = env.read_text()
    assert "# my notes" in text
    assert "UNMANAGED_THING=keep" in text
    assert SECRET in text


def test_an_empty_value_clears_rather_than_stores_nothing(live):
    api = live["api"]
    _write(api, "/agents/operations/credentials", ADMIN, {"values": {"TELEGRAM_BOT_TOKEN": SECRET}})
    response = _write(api, "/agents/operations/credentials", ADMIN,
                      {"values": {"TELEGRAM_BOT_TOKEN": ""}})
    assert response.body["changed"] == ["TELEGRAM_BOT_TOKEN"]
    assert SECRET not in _env(live["home"]).read_text()
    slots = {c["name"]: c for c in api.handle("/platform/v1/agents/operations/credentials").body["credentials"]}
    assert slots["TELEGRAM_BOT_TOKEN"]["set"] is False


def test_a_viewer_may_not_write_a_credential(live):
    assert _write(live["api"], "/agents/operations/credentials", VIEWER,
                  {"values": {"TELEGRAM_BOT_TOKEN": SECRET}}).status == 403
    assert not _env(live["home"]).exists()


def test_a_malformed_credential_body_is_refused(live):
    api = live["api"]
    assert _write(api, "/agents/operations/credentials", ADMIN, {}).status == 400
    assert _write(api, "/agents/operations/credentials", ADMIN, {"values": {}}).status == 400
    assert _write(api, "/agents/operations/credentials", ADMIN,
                  {"values": {"TELEGRAM_BOT_TOKEN": 42}}).status == 400
    assert _write(api, "/agents/ghost/credentials", ADMIN,
                  {"values": {"TELEGRAM_BOT_TOKEN": "x"}}).status == 404


def test_a_value_with_awkward_characters_round_trips(live):
    """A token containing a `#` truncated at the comment marker is a failure nobody would
    think to look for."""
    from nova._env import read_env_file

    awkward = 'tok#en with "quotes" and spaces\\backslash'
    _write(live["api"], "/agents/operations/credentials", ADMIN,
           {"values": {"TELEGRAM_BOT_TOKEN": awkward}})
    env = _env(live["home"])
    assert "TELEGRAM_BOT_TOKEN" in read_env_file(env)

    import subprocess, sys

    # Parsed by something other than NOVA's own reader, so this tests the file rather than
    # a round-trip through one implementation's blind spots.
    value = subprocess.run(
        [sys.executable, "-c",
         "import sys,shlex;"
         "line=[l for l in open(sys.argv[1]) if l.startswith('TELEGRAM_BOT_TOKEN=')][0];"
         "print(shlex.split(line.strip().split('=',1)[1])[0], end='')",
         str(env)],
        capture_output=True, text=True,
    )
    assert value.stdout == awkward, f"the value did not survive quoting: {value.stdout!r}"


# -- logs ----------------------------------------------------------------------


def _log(home: Path, name: str, text: str) -> Path:
    directory = home / "profiles" / "operations" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text)
    return path


def test_listing_logs_says_what_exists_rather_than_guessing(live):
    body = live["api"].handle("/platform/v1/agents/operations/logs").body
    assert {s["stream"] for s in body["streams"]} == {"agent", "errors", "gateway"}
    assert all(s["present"] is False for s in body["streams"])

    _log(live["home"], "agent.log", "hello\n")
    body = live["api"].handle("/platform/v1/agents/operations/logs").body
    assert next(s for s in body["streams"] if s["stream"] == "agent")["present"] is True


def test_a_log_is_tailed_and_bounded(live):
    _log(live["home"], "agent.log", "\n".join(f"line {i}" for i in range(5000)))
    body = live["api"].handle("/platform/v1/agents/operations/logs",
                              {"stream": "agent", "lines": "10"}).body
    assert body["lines"][-1] == "line 4999"
    assert len(body["lines"]) == 10
    assert body["truncated"] is True


def test_a_caller_cannot_ask_for_an_unbounded_read(live):
    from nova.runtime.hermes.observe import MAX_BYTES, MAX_LINES

    _log(live["home"], "agent.log", "\n".join(f"line {i}" for i in range(50_000)))
    body = live["api"].handle("/platform/v1/agents/operations/logs",
                              {"stream": "agent", "lines": "9999999"}).body
    assert len(body["lines"]) <= MAX_LINES
    assert sum(len(line) for line in body["lines"]) <= MAX_BYTES


def test_a_log_with_no_newlines_still_returns_something(live):
    """One enormous line is what a crash dump written in a single call looks like. Dropping
    the partial first line after seeking would otherwise eat the whole file."""
    _log(live["home"], "errors.log", "x" * 400_000)
    body = live["api"].handle("/platform/v1/agents/operations/logs",
                              {"stream": "errors", "lines": "50"}).body
    assert sum(len(line) for line in body["lines"]) > 0
    assert body["truncated"] is True


def test_the_stream_name_cannot_escape_the_log_directory(live):
    """The stream comes from a URL. Resolving a caller-supplied name against a directory is
    how traversal happens even when every individual check looks fine."""
    for hostile in ("../../etc/passwd", "../.env", "/etc/passwd", "..", "agent.log"):
        response = live["api"].handle("/platform/v1/agents/operations/logs",
                                      {"stream": hostile})
        assert response.status == 400, f"{hostile} was accepted"


def test_logs_are_admin_only(live):
    """A log line can carry anything the runtime wrote — a prompt, a tool argument, part of
    a document. No sanitiser is attempted, so the gate is who may read."""
    assert VIEWER.may("/agents/operations/logs") is False
    assert ADMIN.may("/agents/operations/logs") is True


def test_activity_reports_only_what_the_runtime_recorded(live):
    body = live["api"].handle("/platform/v1/agents/operations/activity").body
    assert set(body) == {"agent_id", "tasks", "executions", "decisions", "logs"}
    # A fresh tenant has done nothing, and every section says so by being empty rather than
    # by being absent or invented.
    assert body["tasks"] == [] and body["executions"] == [] and body["decisions"] == []
    assert live["api"].handle("/platform/v1/agents/ghost/activity").status == 404
