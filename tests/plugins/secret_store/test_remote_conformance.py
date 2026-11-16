"""The installed secret store conformance suites against the remote store, and the vendored
connector fixtures against its client.

``TestRemotePinning`` runs ``hermes_conformance.secrets.SecretPinningConformance`` (the port: pin,
administration, levels, no value on any refusal or failure path) and ``TestRemoteView`` runs
``SecretViewConformance`` (the process-wide view hooks), both over real HTTP to the stub plane. The
fixture tests pin the vendored bytes to the manifest and ``SOURCE.json``, check that the names the
connector refuses are exactly the ones this client keeps local, and that the resolved views and the
agent's wire steps the connector specifies are what the client reads and sends.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import types
from pathlib import Path

import pytest

import hermes_secret_routing as routing
from agent.runtime_scope import bind_profile_scope, bind_state
from hermes_conformance.secrets import Backend, SecretPinningConformance, SecretViewConformance, ViewBackend
from hermes_scope import LOCAL_OWNER, Group, Principal, Scope
from plugins.config_backends.remote import client as config_client
from plugins.config_backends.remote import credentials as cred_mod
from plugins.secret_store import remote as remote_pkg
from plugins.secret_store.remote import backend as backend_mod
from plugins.secret_store.remote import client as secrets_client

from .stub_secrets_plane import INSTANCE, StubSecretsPlane, remote_env

VENDOR = Path(__file__).resolve().parents[3] / "plugins" / "secret_store" / "remote" / "vendor"
MANIFEST = VENDOR / "conformance" / "secrets-plane-fixtures.sha256"
# A literal pin: an edited or re-vendored fixture set is a reviewed bump here.
APPROVED_FIXTURE_DIGEST = "623b8b8f16cecbd5c284abccec290ec65b931714a394f208fb1f4193cd27aeb4"
PINNED_KEY = "ZZ_ADMINISTRATOR_PINNED_KEY"


@pytest.fixture
def plane(tmp_path, monkeypatch):
    root = tmp_path / "root"
    homes = {name: root / "profiles" / name for name in ("a", "b", "unpinned", "launch")}
    for home in homes.values():
        home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(homes["launch"]))  # none of the agents is the process's own
    with StubSecretsPlane() as p:
        for k, v in remote_env(p).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(backend_mod, "BOOT_RETRY_DELAYS", ())
        remote_pkg._reset_for_tests()
        routing.reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        yield p, homes
        remote_pkg._reset_for_tests()
        routing.reset_for_tests()
        cred_mod._IDP_CACHE.clear()


def remote_backend(plane, homes, monkeypatch, tmp_path) -> Backend:
    provider = remote_pkg.get_provider()

    def bind_member(agent):
        return bind_state(Scope("local", str(homes[agent]), Principal("member"), owner=LOCAL_OWNER, can_administer=False))

    def foreign_tenant(agent):
        return bind_state(Scope("acme", str(homes[agent]), Principal("u"), owner=Group("g"), can_administer=False))

    @contextlib.contextmanager
    def unbound():
        monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
        yield

    @contextlib.contextmanager
    def managed_key():
        managed = tmp_path / "managed"
        managed.mkdir(exist_ok=True)
        (managed / ".env").write_text(f"{PINNED_KEY}=set-by-the-administrator\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        try:
            yield PINNED_KEY
        finally:
            monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)

    @contextlib.contextmanager
    def fail_writes(kind):
        """``prepare``: the plane credential cannot be read; ``encode``: the body cannot be
        serialized; ``replace``: the plane fails the write itself."""
        if kind == "prepare":
            def unreadable(home):
                raise OSError("the credential store could not be read")
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(secrets_client, "plane_token", unreadable)
                yield
        elif kind == "encode":
            def dumps(*args, **kwargs):
                raise TypeError("the body cannot be serialized")
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(config_client, "json", types.SimpleNamespace(dumps=dumps, loads=json.loads))
                yield
        elif kind == "replace":
            plane.fail_status = 500
            try:
                yield
            finally:
                plane.fail_status = None
        else:
            raise LookupError(f"no such failure point: {kind!r}")

    return Backend(open=lambda agent: provider.open(homes[agent]),
                   bind=lambda agent: bind_profile_scope(homes[agent]),
                   bind_member=bind_member, foreign_tenant=foreign_tenant, unbound=unbound,
                   managed_key=managed_key, fail_writes=fail_writes, levels=provider.levels)


@pytest.fixture
def backend(plane, monkeypatch, tmp_path):
    p, homes = plane
    return remote_backend(p, homes, monkeypatch, tmp_path)


class TestRemotePinning(SecretPinningConformance):
    pass


@pytest.fixture
def view_backend(plane):
    p, homes = plane
    provider = remote_pkg.get_provider()
    home = homes["a"]
    provider.boot(home)

    @contextlib.contextmanager
    def failing():
        p.fail_status = 500
        try:
            yield
        finally:
            p.fail_status = None

    def restart():
        """Values apply at start: a change made elsewhere is read by the next process. A poll asks
        for that restart; this stands in for it by loading the home afresh in the same provider."""
        assert provider.poll_one(provider._states[provider._key(home)])
        with provider._lock:
            provider._states.clear()
            provider._restart_reason = None
            provider._restart_taken = False
        provider.boot(home)

    return ViewBackend(provider=provider, home=home,
                       seed_external=lambda env: p.set_level(p.profile("a"), env=env),
                       refresh=restart, fail_writes=failing)


class TestRemoteView(SecretViewConformance):
    pass


# --- the vendored connector fixtures ----------------------------------------------------------


def _fixtures(sub):
    return sorted((VENDOR / "conformance" / "secrets-plane" / sub).glob("*.json"))


def test_vendored_fixtures_match_manifest_and_source():
    source = json.loads((VENDOR / "SOURCE.json").read_text(encoding="utf-8-sig"))
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert digest == APPROVED_FIXTURE_DIGEST and source["fixtureDigest"] == APPROVED_FIXTURE_DIGEST
    listed = {}
    for line in MANIFEST.read_text(encoding="utf-8-sig").splitlines():
        sha, rel = line.split(None, 1)
        listed[rel.strip()] = sha
    for rel, sha in listed.items():
        assert hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == sha, rel
    on_disk = {str(p.relative_to(VENDOR)) for p in (VENDOR / "conformance").rglob("*.json")}
    assert on_disk == set(listed)


@pytest.mark.parametrize("case", json.loads((VENDOR / "conformance" / "secrets-plane" / "reserved-names.json")
                                            .read_text(encoding="utf-8-sig"))["cases"],
                         ids=lambda c: f"{c['kind']}:{c['name']}")
def test_the_names_the_plane_refuses_are_exactly_the_ones_this_client_keeps_local(case):
    provider = backend_mod.RemoteSecretStoreProvider()
    if case["kind"] == "env":
        kept = routing.stays_local(provider, case["name"])
    else:
        kept = case["name"] in provider.bootstrap_auth_records()
    assert kept is (case["expect"] == "reserved")


def test_this_clients_bootstrap_set_is_what_the_plane_reserves():
    reserved = json.loads((VENDOR / "conformance" / "secrets-reserved-names.json").read_text(encoding="utf-8-sig"))
    provider = backend_mod.RemoteSecretStoreProvider()
    planes = set(reserved["bootstrapEnvExact"]) | set(reserved["bootstrapContract"]["envExact"])
    assert provider.bootstrap_env_names() == planes
    assert set(reserved["bootstrapContract"]["authProviders"]) <= provider.bootstrap_auth_records()
    for prefix in reserved["bootstrapEnvPrefixes"]:
        assert provider.keeps_local(prefix + "ANYTHING")
    for name in reserved["globalEnvExact"]:
        assert routing.stays_local(provider, name)


@pytest.mark.parametrize("fixture", _fixtures("resolution"), ids=lambda f: f.stem)
def test_a_resolved_view_reads_as_the_connector_resolves_it(fixture, plane):
    case = json.loads(fixture.read_text(encoding="utf-8-sig"))
    p, homes = plane
    p.instances[INSTANCE]["owner"] = case["agent"]["owner"]
    levels = case["levels"]
    for doc, spec in ((p.tenant, levels.get("tenant")), (p.instances[INSTANCE]["level"], levels.get("instance")),
                      (p.profile("a"), levels.get("profile"))):
        if spec:
            p.set_level(doc, env=spec.get("env"), auth=spec.get("auth"))
    for subject, spec in (levels.get("owners") or {}).items():
        p.set_level(p.owner_level(subject), env=spec.get("env"), auth=spec.get("auth"))
    provider = remote_pkg.get_provider()
    expected = case["expected"]
    assert provider.env_view(homes["a"]) == {k: v["value"] for k, v in expected["env"].items()}
    assert provider.auth_view(homes["a"]) == {k: v["record"] for k, v in expected["auth"].items()}
    view = backend_mod.parse_view({**expected, "version": "v", "profileVersion": 0}, None)
    assert {k: meta[0] for k, meta in view.env_meta.items()} == {k: v["level"] for k, v in expected["env"].items()}


#: Connector fixtures whose steps the agent's client takes (``as: agent``); the others exercise
#: routes the agent never calls (a person's names-only reads and administration, the NAS editor).
AGENT_WIRE = {"self-get", "self-patch", "errors", "profile-cas", "auth-cas-per-provider",
              "env-requires-profile-version", "atomic-mixed-write"}
NOT_AGENT_WIRE = {"person-agent-get": "a person's names-only read of an agent",
                  "person-profile-patch": "a person administering an agent's profile",
                  "nas-instance-patch": "the NAS editor's instance-level write",
                  "owner-and-tenant-levels": "owner and tenant administration routes",
                  "level-cas": "owner and tenant level CAS, written by people"}


def test_every_wire_fixture_is_run_or_named_as_another_callers():
    names = {f.stem for f in [*_fixtures("wire"), *_fixtures("cas")]}
    assert names == AGENT_WIRE | set(NOT_AGENT_WIRE)


def _matches(expected, got) -> bool:
    if expected in ("<any>", "<etag>"):
        return isinstance(got, str) and (expected == "<any>" or bool(got))
    if isinstance(expected, dict):
        return isinstance(got, dict) and set(expected) == set(got) and all(_matches(v, got[k]) for k, v in expected.items())
    return expected == got


@pytest.mark.parametrize("fixture", [f for f in [*_fixtures("wire"), *_fixtures("cas")] if f.stem in AGENT_WIRE],
                         ids=lambda f: f.stem)
def test_the_agents_wire_steps_are_what_the_client_sends_and_reads(fixture, plane, monkeypatch):
    case = json.loads(fixture.read_text(encoding="utf-8-sig"))
    p, homes = plane
    previous_etag = None
    for step in case["steps"]:
        actor, path = step["as"], step["path"]
        if actor == "agent":
            profile = path.split("profile=", 1)[1]
            etag = previous_etag if step.get("ifNoneMatch") == "<previous.etag>" else None
            resp = secrets_client.request(step["method"], homes["a"], profile, etag=etag, body=step.get("body"))
        elif actor == "admin" and path == "/v1/secrets/tenant":
            p.set_level(p.tenant, env=step["body"].get("set"))
            continue
        elif actor == "none":
            resp = config_client.send(step["method"], p.url + path, token="", headers={})
        else:
            continue  # another caller's route (a person's read): not the agent's to send
        assert resp.status == step["expect"]["status"], (step, resp.body)
        if "body" in step["expect"]:
            assert _matches(step["expect"]["body"], resp.body), (step["expect"]["body"], resp.body)
        if resp.status == 200:
            assert resp.etag == resp.body["version"]
            previous_etag = resp.etag
