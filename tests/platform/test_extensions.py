"""Granting an agent an MCP server or a runtime plugin.

Both are capabilities the runtime already has, so most of what follows is about the two
ways a control plane can lie about them.

**The durability lie.** ``nova apply`` rewrites the profile's ``config.yaml`` from the
bundle. A grant written into that file would save, persist, survive a reload, and vanish at
the next apply — the same trap as editing ``SOUL.md`` in place. So the tests below check
that a grant lands in the *bundle* and reaches the profile only through materialization.

**The readiness lie.** 54 of the 65 catalogue entries authenticate with OAuth, which needs
a browser consent NOVA cannot perform. A green tick beside one of those would be a claim
that the agent can call it. It cannot, and the warning says so.

The catalogue itself is the runtime's; these tests assert the shape NOVA puts around it,
not its contents, so a runtime upgrade that adds a server does not fail them.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

import nova.runtime.hermes  # noqa: F401 — registers the extension discovery
from nova.apply import apply_bundle
from nova.audit import AuditLog
from nova.control import ControlAPI
from nova.control.auth import Principal
from nova.errors import SpecError
from nova.extensions import catalogue
from nova.extensions import manage as extension_ops
from nova.runtime import get_runtime
from nova.runtime.hermes.materialize import build_config, extension_sections, warnings_for
from nova.spec import load_bundle
from nova.spec.agent import AgentSpec

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "nova" / "examples" / "acme"
ADMIN = Principal(name="ops", role="admin")
VIEWER = Principal(name="watcher", role="viewer")
AGENT = "customer-support"


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
    return {"api": ControlAPI(bundle, runtime, audit=audit), "root": root, "home": home}


def _agent_yaml(root: Path, agent_id: str = AGENT) -> dict:
    return yaml.safe_load((root / "agents" / f"{agent_id}.yaml").read_text())


def _profile_config(home: Path, agent_id: str = AGENT) -> dict:
    path = home / "profiles" / agent_id / "config.yaml"
    return yaml.safe_load(path.read_text()) if path.is_file() else {}


def _first(kind: str):
    """One catalogue entry of a given auth kind, or skip.

    Chosen from the live catalogue rather than named, so this file does not become a
    second copy of a registry the runtime owns.
    """
    entry = next((e for e in catalogue().mcp if e.auth == kind and not e.installs), None)
    if entry is None:
        pytest.skip(f"this runtime ships no {kind} MCP server")
    return entry


# -- the catalogue -------------------------------------------------------------


def test_the_catalogue_comes_from_the_runtime():
    known = catalogue()
    assert known.mcp, "the runtime ships an MCP catalogue and NOVA should be reading it"
    assert known.plugins, "the runtime ships plugins and NOVA should be reading them"


def test_every_mcp_entry_names_the_toolset_it_creates():
    # An MCP server is not a new governance surface: its tools land in `mcp-<name>`, which
    # the compiled policy decides like any other. Saying so is how the screen avoids
    # implying a gate that is not there.
    for entry in catalogue().mcp:
        assert entry.toolset == f"mcp-{entry.id}"


def test_channels_are_not_offered_as_plugins():
    # 22 of the discovered manifests are platforms. They have their own screen, and a
    # second switch for the same thing would eventually disagree with the first.
    offered = [e for e in catalogue().plugins if e.grantable]
    assert offered
    assert all(e.kind != "platform" for e in offered)


def test_a_bundled_backend_is_marked_as_already_loading():
    backend = next((e for e in catalogue().plugins if e.kind == "backend"), None)
    assert backend is not None
    # Offering "enable" for one of these would be a switch that reports a change and
    # changes nothing.
    assert backend.auto_loads is True


def test_no_credential_value_can_travel_with_a_catalogue_entry():
    for entry in catalogue().mcp:
        for row in entry.to_dict()["credentials"]:
            assert set(row) == {"name", "prompt", "required", "secret"}


# -- declaring -----------------------------------------------------------------


def test_a_grant_is_declared_in_the_bundle_not_the_profile(live):
    server = _first("oauth")
    response = live["api"].write(
        f"/platform/v1/agents/{AGENT}/mcp", ADMIN, {"server": server.id, "granted": True}
    )
    assert response.status == 200, response.body
    assert _agent_yaml(live["root"])["extensions"]["mcp"] == [server.id]


def test_applying_compiles_the_grant_into_the_profile(live):
    server = _first("oauth")
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": server.id, "granted": True})
    servers = _profile_config(live["home"]).get("mcp_servers") or {}
    assert server.id in servers
    assert servers[server.id]["enabled"] is True


def test_a_grant_survives_a_re_apply(live):
    """The property the whole design exists for.

    A grant written straight into ``config.yaml`` would pass every test above and fail
    this one, silently, the first time anybody applied the bundle again.
    """
    server = _first("oauth")
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": server.id, "granted": True})
    bundle = load_bundle(live["root"])
    runtime = get_runtime("hermes", home=live["home"], tenant_id=bundle.tenant_id)
    audit = AuditLog.for_home(live["home"], tenant_id=bundle.tenant_id, actor="test")
    apply_bundle(bundle, runtime, audit=audit)
    assert server.id in (_profile_config(live["home"]).get("mcp_servers") or {})


def test_revoking_removes_it_from_both(live):
    server = _first("oauth")
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": server.id, "granted": True})
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": server.id, "granted": False})
    assert "extensions" not in _agent_yaml(live["root"])
    assert not (_profile_config(live["home"]).get("mcp_servers") or {})


def test_the_policy_plugin_is_never_dropped_when_plugins_are_granted(live):
    """The most dangerous regression available here.

    Replacing ``plugins.enabled`` instead of merging into it would leave an agent whose
    enforcement plugin is present, correct, and never consulted — a governance control
    that passes review by inspection and enforces nothing.
    """
    live["api"].write(f"/platform/v1/agents/{AGENT}/plugins", ADMIN,
                      {"plugin": "google_meet", "state": "enable"})
    enabled = (_profile_config(live["home"]).get("plugins") or {}).get("enabled") or []
    assert "nova-policy" in enabled
    assert "google_meet" in enabled


def test_a_plugin_can_be_disabled(live):
    live["api"].write(f"/platform/v1/agents/{AGENT}/plugins", ADMIN,
                      {"plugin": "web/tavily", "state": "disable"})
    plugins = _profile_config(live["home"]).get("plugins") or {}
    assert plugins["disabled"] == ["web/tavily"]


def test_returning_a_plugin_to_default_clears_both_lists(live):
    for state in ("enable", "disable", "default"):
        live["api"].write(f"/platform/v1/agents/{AGENT}/plugins", ADMIN,
                          {"plugin": "google_meet", "state": state})
    assert "extensions" not in _agent_yaml(live["root"])


def test_enable_and_disable_cannot_both_hold(tmp_path):
    with pytest.raises(SpecError):
        AgentSpec.parse({
            "id": "x",
            "extensions": {"plugins": {"enable": ["a"], "disable": ["a"]}},
        })


# -- refusals ------------------------------------------------------------------


def test_a_server_outside_the_catalogue_is_refused(live):
    response = live["api"].write(
        f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
        {"server": "not-a-real-server", "granted": True},
    )
    assert response.status == 400
    assert "catalogue" in response.body["error"]["message"]
    assert "extensions" not in _agent_yaml(live["root"])


def test_an_arbitrary_command_cannot_be_smuggled_in_as_a_server(live):
    """The reason the catalogue is the allowlist.

    A stdio MCP server is a command line. Accepting one from a browser form would turn
    admin on the Control Centre into code execution on the NOVA host.
    """
    for attempt in ("/bin/sh -c 'curl evil'", "../../etc/passwd", "python -c pwn"):
        response = live["api"].write(
            f"/platform/v1/agents/{AGENT}/mcp", ADMIN, {"server": attempt, "granted": True}
        )
        assert response.status == 400, attempt
    assert "extensions" not in _agent_yaml(live["root"])


def test_a_plugin_the_runtime_does_not_ship_is_refused(live):
    response = live["api"].write(
        f"/platform/v1/agents/{AGENT}/plugins", ADMIN,
        {"plugin": "nope/nothing", "state": "enable"},
    )
    assert response.status == 400
    assert "extensions" not in _agent_yaml(live["root"])


def test_an_unknown_plugin_state_is_refused(live):
    response = live["api"].write(
        f"/platform/v1/agents/{AGENT}/plugins", ADMIN,
        {"plugin": "google_meet", "state": "sometimes"},
    )
    assert response.status == 400


def test_a_server_needing_a_host_install_is_not_installed_silently():
    """It clones a repository and runs a build. That is a host act, not a side effect."""
    entry = next((e for e in catalogue().mcp if e.installs), None)
    if entry is None:
        pytest.skip("this runtime ships no install-based MCP server")
    spec = AgentSpec.parse({"id": "x", "extensions": {"mcp": [entry.id]}})
    servers, _, problems = extension_sections(spec)
    assert entry.id not in servers
    assert any("hermes mcp install" in p["reason"] for p in problems)


def test_a_viewer_cannot_grant_anything(live):
    server = _first("oauth")
    for path, body in (
        (f"/platform/v1/agents/{AGENT}/mcp", {"server": server.id, "granted": True}),
        (f"/platform/v1/agents/{AGENT}/plugins", {"plugin": "google_meet", "state": "enable"}),
    ):
        assert live["api"].write(path, VIEWER, body).status == 403
    assert "extensions" not in _agent_yaml(live["root"])


# -- honesty -------------------------------------------------------------------


def test_an_oauth_grant_is_reported_as_not_yet_authorized():
    server = _first("oauth")
    spec = AgentSpec.parse({"id": "x", "extensions": {"mcp": [server.id]}})
    notes = warnings_for(spec)
    assert any("hermes mcp login" in note for note in notes)


def test_a_server_needing_no_auth_carries_no_such_warning():
    server = _first("none")
    spec = AgentSpec.parse({"id": "x", "extensions": {"mcp": [server.id]}})
    assert not any("mcp login" in note for note in warnings_for(spec))


def test_saved_and_applied_are_reported_apart(live):
    server = _first("oauth")
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": server.id, "granted": True})
    # The API holds the bundle it was constructed with; a fresh one sees the edit.
    bundle = load_bundle(live["root"])
    runtime = get_runtime("hermes", home=live["home"], tenant_id=bundle.tenant_id)
    api = ControlAPI(bundle, runtime,
                     audit=AuditLog.for_home(live["home"], tenant_id=bundle.tenant_id,
                                             actor="test"))
    body = api.handle(f"/platform/v1/agents/{AGENT}/extensions").body
    assert body["granted"]["mcp"] == [server.id]
    assert body["applied"]["known"] is True
    assert server.id in body["applied"]["mcp"]


def test_an_unapplied_agent_says_so(live):
    body = live["api"].handle(f"/platform/v1/agents/{AGENT}/extensions").body
    assert body["applied"]["known"] is False
    assert "not been applied" in body["applied"]["detail"]


def test_the_agent_view_marks_what_is_granted(live):
    server = _first("oauth")
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": server.id, "granted": True})
    bundle = load_bundle(live["root"])
    api = ControlAPI(bundle, get_runtime("hermes", home=live["home"],
                                         tenant_id=bundle.tenant_id))
    rows = api.handle(f"/platform/v1/agents/{AGENT}/extensions").body["mcp"]
    assert [row["id"] for row in rows if row["granted"]] == [server.id]


def test_an_unknown_agent_is_a_404(live):
    assert live["api"].handle("/platform/v1/agents/ghost/extensions").status == 404


# -- credentials ---------------------------------------------------------------


def test_an_api_key_server_contributes_a_writable_credential_slot(live):
    entry = next((e for e in catalogue().mcp if e.auth == "api_key" and e.credentials), None)
    if entry is None:
        pytest.skip("this runtime ships no api_key MCP server")
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": entry.id, "granted": True})
    bundle = load_bundle(live["root"])
    api = ControlAPI(bundle, get_runtime("hermes", home=live["home"],
                                         tenant_id=bundle.tenant_id))
    slots = api.handle(f"/platform/v1/agents/{AGENT}/credentials").body["credentials"]
    names = [s["name"] for s in slots if s["source"] == f"mcp:{entry.id}"]
    assert names == [var.name for var in entry.credentials]


def test_an_oauth_server_contributes_no_credential_slot(live):
    """Its authorization is a consent the runtime stores, not a variable.

    Offering a field would invite somebody to paste a token into a file nothing reads.
    """
    server = _first("oauth")
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": server.id, "granted": True})
    bundle = load_bundle(live["root"])
    api = ControlAPI(bundle, get_runtime("hermes", home=live["home"],
                                         tenant_id=bundle.tenant_id))
    slots = api.handle(f"/platform/v1/agents/{AGENT}/credentials").body["credentials"]
    assert not [s for s in slots if s["source"] == f"mcp:{server.id}"]


def test_revoking_a_server_revokes_the_right_to_write_its_key(live):
    from nova.credentials import writable_names

    entry = next((e for e in catalogue().mcp if e.auth == "api_key" and e.credentials), None)
    if entry is None:
        pytest.skip("this runtime ships no api_key MCP server")
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": entry.id, "granted": True})
    granted = writable_names(load_bundle(live["root"]), AGENT)
    assert entry.credentials[0].name in granted

    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": entry.id, "granted": False})
    revoked = writable_names(load_bundle(live["root"]), AGENT)
    assert entry.credentials[0].name not in revoked


# -- audit ---------------------------------------------------------------------


def test_a_grant_is_audited_intent_then_committed(live):
    server = _first("oauth")
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": server.id, "granted": True})
    lines = [json.loads(line) for line in
             (live["home"] / "nova" / "audit.jsonl").read_text().splitlines()]
    rows = [r for r in lines if r["kind"] == "agent.mcp_changed"]
    assert [r["phase"] for r in rows] == ["intent", "committed"]
    assert rows[0]["detail"]["server"] == server.id


def test_a_refused_grant_is_audited_as_failed(live):
    live["api"].write(f"/platform/v1/agents/{AGENT}/mcp", ADMIN,
                      {"server": "not-a-real-server", "granted": True})
    lines = [json.loads(line) for line in
             (live["home"] / "nova" / "audit.jsonl").read_text().splitlines()]
    rows = [r for r in lines if r["kind"] == "agent.mcp_changed"]
    assert rows[-1]["phase"] == "failed"


# -- nothing else moved --------------------------------------------------------


def test_an_agent_with_no_extensions_compiles_exactly_as_before():
    spec = AgentSpec.parse({"id": "x"})
    config = build_config(spec, policy=True, knowledge=False)
    assert "mcp_servers" not in config
    assert config["plugins"]["enabled"] == ["nova-policy"]
