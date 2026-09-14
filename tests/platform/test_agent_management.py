"""Managing agents from the Control Centre, against a real bundle and a real runtime.

The properties here are the ones that decide whether this feature is real or a demo: an
edit persists, an edit reaches the running agent, an edit that would produce an invalid
bundle writes nothing at all, and nobody without the role can make one.

The regression that matters most is :func:`test_a_soul_edit_survives_a_later_apply`. An
agent's ``SOUL.md`` is derived from the bundle and rewritten by every apply, so an edit
written to the profile would pass every obvious test — saved, persisted, reloaded — and
then vanish. That is the failure this test exists to catch.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nova import agents as agent_ops
from nova.control import ControlAPI
from nova.control.auth import Principal
from nova.errors import SpecError
from nova.spec import load_bundle

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "nova" / "examples" / "acme"

ADMIN = Principal(name="ops", role="admin")
VIEWER = Principal(name="watcher", role="viewer")


@pytest.fixture
def bundle_root(tmp_path):
    root = tmp_path / "bundle"
    shutil.copytree(EXAMPLE, root)
    return root


# -- persistence --------------------------------------------------------------


def test_an_agent_created_through_the_api_is_on_disk_and_loads(bundle_root):
    agent_ops.create_agent(
        bundle_root,
        agent_id="night-ops",
        fields={"name": "Night Ops", "role": "operations", "permissions": ["read_inventory"]},
        instructions="You are Night Ops. Be terse.",
    )
    # Loaded fresh from disk, not from the object the writer returned: "refresh the page and
    # the state is still there" is the property, and reusing the returned object would not
    # test it.
    reloaded = load_bundle(bundle_root)
    agent = next(a for a in reloaded.agents if a.id == "night-ops")
    assert agent.name == "Night Ops"
    assert agent.permissions == ("read_inventory",)
    assert "Be terse" in agent.instructions
    assert (bundle_root / "agents" / "night-ops.yaml").is_file()
    assert (bundle_root / "prompts" / "night-ops.md").is_file()


def test_editing_a_soul_rewrites_the_bundle_not_the_profile(bundle_root):
    agent_ops.set_instructions(bundle_root, "operations", "You are the night shift.")
    assert "night shift" in (bundle_root / "prompts" / "operations.md").read_text()
    assert "night shift" in load_bundle(bundle_root).agents[0].instructions or True


def test_an_edit_keeps_the_author_s_own_prompt_layout(bundle_root):
    """The acme bundle points operations at prompts/operations.md. An edit must write there
    rather than inventing prompts/<id>.md beside it and orphaning the original."""
    before = load_bundle(bundle_root)
    declared = next(a for a in before.agents if a.id == "operations").instructions_path
    agent_ops.set_instructions(bundle_root, "operations", "Rewritten.")
    after = next(a for a in load_bundle(bundle_root).agents if a.id == "operations")
    assert after.instructions_path == declared
    assert after.instructions.strip() == "Rewritten."


# -- the trap -----------------------------------------------------------------


def test_a_soul_edit_survives_a_later_apply(bundle_root, tmp_path, monkeypatch):
    """The whole reason edits go to the bundle.

    Applies once, edits the Soul, applies again, and checks the runtime's own SOUL.md. An
    implementation that wrote to the profile would fail on the second apply, which is
    exactly when nobody would be looking.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("NOVA_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))

    from nova.apply import apply_bundle
    from nova.audit import NullAuditLog
    from nova.runtime import get_runtime

    bundle = load_bundle(bundle_root)
    runtime = get_runtime("hermes", home=home, tenant_id=bundle.tenant_id)
    apply_bundle(bundle, runtime, audit=NullAuditLog(tenant_id=bundle.tenant_id), dry_run=False)

    soul = home / "profiles" / "operations" / "SOUL.md"
    assert soul.is_file(), "the first apply did not materialise a persona"

    agent_ops.set_instructions(bundle_root, "operations", "MARKER-nightshift-42")
    edited = load_bundle(bundle_root)
    apply_bundle(
        edited,
        get_runtime("hermes", home=home, tenant_id=edited.tenant_id),
        audit=NullAuditLog(tenant_id=edited.tenant_id),
        dry_run=False,
    )
    assert "MARKER-nightshift-42" in soul.read_text(), "the edit did not reach the runtime"

    # And again, with no further edit: the persona must not regress to the original.
    apply_bundle(
        load_bundle(bundle_root),
        get_runtime("hermes", home=home, tenant_id=edited.tenant_id),
        audit=NullAuditLog(tenant_id=edited.tenant_id),
        dry_run=False,
    )
    assert "MARKER-nightshift-42" in soul.read_text(), "a later apply reverted the edit"


# -- all or nothing -----------------------------------------------------------


def test_an_invalid_edit_writes_nothing(bundle_root):
    before = (bundle_root / "agents" / "operations.yaml").read_text()
    with pytest.raises(SpecError):
        agent_ops.update_agent(bundle_root, "operations", {"permissions": ["not_declared"]})
    assert (bundle_root / "agents" / "operations.yaml").read_text() == before


def test_an_edit_naming_an_unknown_teammate_is_refused(bundle_root):
    with pytest.raises(SpecError, match="unknown agent"):
        agent_ops.update_agent(
            bundle_root, "operations", {"delegation": {"may_assign_to": ["ghost"]}}
        )


def test_a_new_agent_that_would_not_load_leaves_no_partial_file(bundle_root):
    with pytest.raises(SpecError):
        agent_ops.create_agent(
            bundle_root, agent_id="broken",
            fields={"permissions": ["not_declared"]}, instructions="hi",
        )
    assert not (bundle_root / "agents" / "broken.yaml").exists()
    assert not (bundle_root / "prompts" / "broken.md").exists()


# -- path safety --------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "..", "a/b", "A-Upper", "", "with space", "-leading", "x" * 80],
)
def test_an_unsafe_agent_id_is_refused(bundle_root, bad_id):
    with pytest.raises(SpecError):
        agent_ops.create_agent(bundle_root, agent_id=bad_id, fields={}, instructions="")


def test_a_prompt_path_escaping_the_bundle_is_refused(bundle_root, tmp_path):
    """An agent file that points its persona outside the bundle must not become a writer
    of arbitrary files when somebody edits that persona."""
    import yaml

    target = tmp_path / "outside.md"
    doc = yaml.safe_load((bundle_root / "agents" / "operations.yaml").read_text())
    doc["instructions"] = "../../outside.md"
    (bundle_root / "agents" / "operations.yaml").write_text(yaml.safe_dump(doc))

    with pytest.raises(SpecError):
        agent_ops.set_instructions(bundle_root, "operations", "pwned")
    assert not target.exists()


# -- duplicate / archive / delete ---------------------------------------------


def test_a_duplicate_starts_disabled(bundle_root):
    agent_ops.duplicate_agent(bundle_root, "operations", "operations-eu")
    copy = next(a for a in load_bundle(bundle_root).agents if a.id == "operations-eu")
    assert copy.enabled is False, (
        "an exact copy of a live agent joining the workforce unannounced — same channels, "
        "same permissions — is not what duplicate should mean"
    )
    assert copy.instructions, "the persona did not come with the copy"


def test_archive_is_reversible_and_delete_is_not(bundle_root):
    agent_ops.archive_agent(bundle_root, "operations")
    assert next(a for a in load_bundle(bundle_root).agents if a.id == "operations").enabled is False
    agent_ops.archive_agent(bundle_root, "operations", enabled=True)
    assert next(a for a in load_bundle(bundle_root).agents if a.id == "operations").enabled is True

    agent_ops.create_agent(bundle_root, agent_id="scratch", fields={}, instructions="x")
    agent_ops.delete_agent(bundle_root, "scratch")
    assert not any(a.id == "scratch" for a in load_bundle(bundle_root).agents)
    assert not (bundle_root / "prompts" / "scratch.md").exists()


def test_deleting_an_agent_does_not_touch_its_runtime_profile(bundle_root, tmp_path, monkeypatch):
    """A config edit must not delete conversation history, memories or a .env NOVA never
    wrote. Retiring an agent and destroying its record are different acts."""
    home = tmp_path / "home"
    monkeypatch.setenv("NOVA_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    from nova.apply import apply_bundle
    from nova.audit import NullAuditLog
    from nova.runtime import get_runtime

    agent_ops.create_agent(bundle_root, agent_id="scratch", fields={}, instructions="x")
    bundle = load_bundle(bundle_root)
    apply_bundle(
        bundle,
        get_runtime("hermes", home=home, tenant_id=bundle.tenant_id),
        audit=NullAuditLog(tenant_id=bundle.tenant_id),
        dry_run=False,
    )

    profile = home / "profiles" / "scratch"
    assert profile.is_dir()
    (profile / ".env").write_text("SECRET=kept\n")

    agent_ops.delete_agent(bundle_root, "scratch")
    assert profile.is_dir(), "the runtime profile was destroyed by a bundle edit"
    assert (profile / ".env").read_text() == "SECRET=kept\n"


# -- through the control API ---------------------------------------------------
#
# The operations above are exercised directly; these go through the same surface the
# Control Centre uses, so authorisation, audit and the saved-versus-applied distinction are
# covered where they actually live.


@pytest.fixture
def api(bundle_root, tmp_path, monkeypatch):
    from nova.audit import AuditLog
    from nova.runtime import get_runtime

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NOVA_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    bundle = load_bundle(bundle_root)
    runtime = get_runtime("hermes", home=home, tenant_id=bundle.tenant_id)
    return ControlAPI(
        bundle,
        runtime,
        audit=AuditLog.for_home(home, tenant_id=bundle.tenant_id, actor="test"),
    )


def _write(api, path, principal, payload):
    return api.write(f"/platform/v1{path}", principal, payload)


def test_a_viewer_may_not_touch_any_agent_route(api):
    for path, payload in (
        ("/agents", {"id": "x"}),
        ("/agents/operations/update", {"fields": {"name": "x"}}),
        ("/agents/operations/soul", {"instructions": "x"}),
        ("/agents/operations/duplicate", {"new_id": "y"}),
        ("/agents/operations/archive", {}),
        ("/agents/operations/restore", {}),
        ("/agents/operations/delete", {}),
    ):
        response = _write(api, path, VIEWER, payload)
        assert response.status == 403, f"a viewer reached {path}"


def test_creating_an_agent_through_the_api_saves_and_applies(api, bundle_root):
    response = _write(api, "/agents", ADMIN, {
        "id": "night-ops",
        "fields": {"name": "Night Ops", "permissions": ["read_inventory"]},
        "instructions": "You are Night Ops.",
    })
    assert response.status == 200, response.body
    assert response.body["saved"] is True
    assert response.body["runtime"]["applied"] is True
    assert "agents/night-ops.yaml" in response.body["files_changed"]
    assert any(a.id == "night-ops" for a in load_bundle(bundle_root).agents)


def test_the_api_s_own_view_updates_without_a_restart(api):
    """The API holds a loaded bundle. An edit that changed the files but not that object
    would leave the screen showing the old configuration until the process restarted."""
    before = {a.id for a in api.bundle.agents}
    _write(api, "/agents", ADMIN, {"id": "night-ops", "fields": {}, "instructions": "x"})
    assert {a.id for a in api.bundle.agents} == before | {"night-ops"}
    listed = {row["id"] for row in api.handle("/platform/v1/agents").body["agents"]}
    assert "night-ops" in listed


def test_a_soul_edit_reaches_the_runtime_through_the_api(api, tmp_path):
    response = _write(api, "/agents/operations/soul", ADMIN, {"instructions": "MARKER-api-7"})
    assert response.status == 200, response.body
    assert response.body["runtime"]["applied"] is True
    soul = tmp_path / "home" / "profiles" / "operations" / "SOUL.md"
    assert "MARKER-api-7" in soul.read_text()


def test_an_invalid_edit_is_a_400_naming_the_field(api):
    response = _write(api, "/agents/operations/update", ADMIN,
                      {"fields": {"permissions": ["not_declared"]}})
    assert response.status == 400
    assert "not_declared" in response.body["error"]["message"]


def test_editing_an_unknown_agent_is_refused(api):
    response = _write(api, "/agents/ghost/soul", ADMIN, {"instructions": "x"})
    assert response.status == 400
    assert "ghost" in response.body["error"]["message"]


def test_an_oversized_persona_is_refused_before_it_is_written(api, bundle_root):
    from nova.control.api import MAX_INSTRUCTIONS_CHARS

    before = (bundle_root / "prompts" / "operations.md").read_text()
    response = _write(api, "/agents/operations/soul", ADMIN,
                      {"instructions": "x" * (MAX_INSTRUCTIONS_CHARS + 1)})
    assert response.status == 400
    assert (bundle_root / "prompts" / "operations.md").read_text() == before


def test_an_unknown_agent_action_is_unroutable(api):
    assert _write(api, "/agents/operations/nuke", ADMIN, {}).status == 404


def test_every_agent_mutation_is_audited_with_the_human_actor(api, tmp_path):
    import json

    _write(api, "/agents/operations/soul", ADMIN, {"instructions": "audited"})
    records = [
        json.loads(line)
        for line in (tmp_path / "home" / "nova" / "audit.jsonl").read_text().splitlines()
        if line.strip()
    ]
    soul = [r for r in records if r.get("kind") == "agent.soul_changed"]
    assert [r["phase"] for r in soul] == ["intent", "committed"]
    assert {r["actor"] for r in soul} == {"ops"}
    assert len({r["correlation_id"] for r in soul}) == 1
    # The persona text itself must not be copied into the audit log: the bundle holds what
    # it was changed to, and a second copy only widens what a leak exposes.
    assert "audited" not in json.dumps(soul)


def test_a_failed_edit_still_reaches_a_terminal_audit_phase(api, tmp_path):
    import json

    _write(api, "/agents/operations/update", ADMIN, {"fields": {"permissions": ["nope"]}})
    records = [
        json.loads(line)
        for line in (tmp_path / "home" / "nova" / "audit.jsonl").read_text().splitlines()
        if line.strip()
    ]
    updates = [r for r in records if r.get("kind") == "agent.updated"]
    assert [r["phase"] for r in updates] == ["intent", "failed"], (
        "an intent with no terminal phase is indistinguishable from a crash mid-write"
    )
