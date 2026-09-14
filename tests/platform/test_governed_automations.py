"""Governed automations: the compiler, and the writes that go through it.

Phase 11 withheld *create* because ``cron.jobs.create_job`` takes a free-text prompt, and
a control plane calling it would hand an agent a recurring instruction no policy ever
reviewed. Phase 12 adds create by making an automation a spec that compiles against its
tenant first. These tests are mostly about what the compiler **refuses**.

The load-bearing claim, stated precisely so the tests can hold it honest: an automation
may not reach past its owning agent. Its work runs inside that agent's profile, so the
agent's compiled policy is the runtime ceiling; the compiler's job is to stop a
declaration asking for more than the agent was granted.
"""

from __future__ import annotations

import json

import pytest

from nova.audit.log import AuditLog
from nova.automations.compile import compile_automation
from nova.runtime.hermes.automations import validate_schedule
from nova.control.auth import Principal
from nova.errors import SpecError
from nova.spec.automation import AutomationSpec, load_automations
from nova.spec.bundle import load_bundle


EXAMPLE = "nova/examples/acme"


@pytest.fixture
def bundle():
    return load_bundle(EXAMPLE)


def _spec(**overrides) -> AutomationSpec:
    base = {
        "id": "nightly-recon",
        "title": "Nightly reconciliation",
        "agent": "operations",
        "schedule": "every day at 07:00",
        "objective": "Reconcile yesterday's ledger against the bank export.",
    }
    return AutomationSpec.parse({**base, **overrides})


# ---------------------------------------------------------------------------
# Spec shape
# ---------------------------------------------------------------------------

def test_a_minimal_automation_parses():
    spec = _spec()
    assert spec.id == "nightly-recon"
    assert spec.agent == "operations"
    assert spec.enabled is True


def test_an_objective_is_required():
    """Refused by the shared field reader before the spec's own validate() runs — the
    framework's message is more specific, so it is the one asserted."""
    with pytest.raises(SpecError, match="objective: is required"):
        _spec(objective="   ")
    with pytest.raises(SpecError, match="objective: is required"):
        AutomationSpec.parse(
            {"id": "x", "title": "X", "agent": "operations", "schedule": "every day at 07:00"}
        )


def test_a_schedule_is_required():
    with pytest.raises(SpecError, match="schedule: is required"):
        _spec(schedule="")


def test_an_enormous_objective_is_refused():
    """A recurring instruction thousands of characters long is a procedure, and belongs
    in the knowledge base with the objective referencing it."""
    with pytest.raises(SpecError, match="over the"):
        _spec(objective="x" * 5000)


def test_unknown_keys_are_refused():
    """A typo'd field must fail loudly, not be silently ignored — an automation with a
    misspelled ``permissions`` would run with none and look governed."""
    with pytest.raises(SpecError):
        AutomationSpec.parse(
            {
                "id": "x", "title": "X", "agent": "operations",
                "schedule": "every day at 07:00", "objective": "do it",
                "permisions": ["read_inventory"],  # deliberate typo
            }
        )


def test_duplicate_grants_are_refused():
    """Caught by the shared list reader on the bundle path. The spec's own duplicate
    check still matters for a spec built in code rather than parsed."""
    with pytest.raises(SpecError, match="more than once"):
        _spec(permissions=["read_inventory", "read_inventory"])

    from dataclasses import replace

    with pytest.raises(SpecError, match="duplicate entries in permissions"):
        replace(_spec(), permissions=("a", "a")).validate()


# ---------------------------------------------------------------------------
# Compilation against the tenant
# ---------------------------------------------------------------------------

def test_a_valid_automation_compiles(bundle):
    compiled = compile_automation(_spec(permissions=["read_inventory"]), bundle)
    assert compiled.agent_id == "operations"
    assert compiled.tenant_id == "acme"
    assert compiled.digest.startswith("sha256:")
    # Exactly what the runtime will be called with — nothing added at the call site.
    assert compiled.job_kwargs["prompt"] == _spec().objective
    assert compiled.job_kwargs["schedule"] == "every day at 07:00"
    assert compiled.job_kwargs["paused"] is False
    # deliver=local: passing an origin would make the runtime default to routing output
    # back through a channel this automation was never reviewed for.
    assert compiled.job_kwargs["deliver"] == "local"


def test_an_unknown_agent_is_refused(bundle):
    with pytest.raises(SpecError, match="does not have"):
        compile_automation(_spec(agent="ghost"), bundle)


def test_a_permission_the_agent_lacks_is_refused(bundle):
    """The escalation path this exists to close: asking for something the tenant never
    granted, on a timer."""
    with pytest.raises(SpecError, match="does not hold"):
        compile_automation(_spec(permissions=["crm_refund"]), bundle)


def test_knowledge_the_agent_cannot_read_is_refused(bundle):
    with pytest.raises(SpecError, match="may not read"):
        compile_automation(_spec(knowledge=["board-minutes"]), bundle)


def test_a_channel_that_does_not_grant_the_agent_is_refused(bundle):
    with pytest.raises(SpecError, match="do not grant"):
        compile_automation(_spec(channels=["some-other-channel"]), bundle)


def test_an_invalid_schedule_is_refused_with_the_runtimes_own_message(bundle):
    """NOVA does not implement a second schedule grammar: a phrase this accepts is one
    the thing that decides when jobs run accepts, because it is the same parser.

    The parser is *injected* — the compiler may not name the runtime itself, a boundary
    ``tests/platform/test_boundaries.py`` enforces and which caught the first version of
    this code importing ``cron.jobs`` directly.
    """
    with pytest.raises(SpecError, match="is not valid"):
        compile_automation(
            _spec(schedule="whenever I feel like it"), bundle,
            validate_schedule=validate_schedule,
        )


def test_without_a_validator_the_schedule_is_left_to_the_adapter(bundle):
    """Bundle load has no runtime, so it decides everything it can from the bundle alone
    and leaves the phrase to be checked where a parser exists."""
    compiled = compile_automation(_spec(schedule="whenever I feel like it"), bundle)
    assert compiled.job_kwargs["schedule"] == "whenever I feel like it"


def test_the_runtimes_parser_is_what_validates(bundle):
    assert validate_schedule("every 30 minutes")["kind"] == "interval"
    assert validate_schedule("every day at 07:00")["kind"] == "cron"
    with pytest.raises(SpecError):
        validate_schedule("at some point")


def test_a_disabled_automation_is_created_paused(bundle):
    """Created paused rather than created-then-paused, so it is never briefly live."""
    compiled = compile_automation(_spec(enabled=False), bundle)
    assert compiled.job_kwargs["paused"] is True
    assert compiled.job_kwargs["paused_reason"]


def test_paused_reason_is_omitted_when_enabled(bundle):
    """The runtime refuses ``paused_reason`` without ``paused=True``; an empty string is
    not the same as absent. Found by the runtime rejecting the first version of this."""
    compiled = compile_automation(_spec(enabled=True), bundle)
    assert "paused_reason" not in compiled.job_kwargs


def test_the_digest_is_tenant_salted(bundle):
    """Identical declarations in two tenants are not the same reviewed artifact."""
    from dataclasses import replace

    a = compile_automation(_spec(), bundle)
    other = replace(bundle, organization=replace(bundle.organization, tenant_id="globex"))
    b = compile_automation(_spec(), other)
    assert a.digest != b.digest


def test_the_digest_changes_when_the_objective_changes(bundle):
    a = compile_automation(_spec(), bundle)
    b = compile_automation(_spec(objective="Do something else entirely."), bundle)
    assert a.digest != b.digest


# ---------------------------------------------------------------------------
# Bundle integration
# ---------------------------------------------------------------------------

def test_a_bundle_with_a_bad_automation_does_not_load(tmp_path):
    """Caught by ``nova validate``, not at the moment someone tries to schedule it."""
    import shutil

    root = tmp_path / "acme"
    shutil.copytree(EXAMPLE, root)
    (root / "automations").mkdir()
    (root / "automations" / "bad.yaml").write_text(
        "id: bad\ntitle: Bad\nagent: operations\n"
        "schedule: every day at 07:00\nobjective: x\npermissions: [crm_refund]\n"
    )
    with pytest.raises(SpecError, match="does not hold"):
        load_bundle(root)


def test_a_bundle_with_a_good_automation_loads(tmp_path):
    import shutil

    root = tmp_path / "acme"
    shutil.copytree(EXAMPLE, root)
    (root / "automations").mkdir()
    (root / "automations" / "ok.yaml").write_text(
        "id: nightly\ntitle: Nightly\nagent: operations\n"
        "schedule: every day at 07:00\nobjective: Reconcile the ledger.\n"
        "permissions: [read_inventory]\n"
    )
    loaded = load_bundle(root)
    assert [a.id for a in loaded.automations] == ["nightly"]


def test_duplicate_automation_ids_are_refused(tmp_path):
    import shutil

    root = tmp_path / "acme"
    shutil.copytree(EXAMPLE, root)
    (root / "automations").mkdir()
    for name in ("a.yaml", "b.yaml"):
        (root / "automations" / name).write_text(
            "id: same\ntitle: T\nagent: operations\n"
            "schedule: every day at 07:00\nobjective: x\n"
        )
    with pytest.raises(SpecError, match="declared twice"):
        load_automations(root)


def test_no_automations_directory_is_valid(bundle):
    assert bundle.automations == ()


# ---------------------------------------------------------------------------
# Control API: create
# ---------------------------------------------------------------------------

def _api(tmp_path, bundle, runtime):
    from nova.control.api import ControlAPI

    api = ControlAPI.__new__(ControlAPI)
    api.bundle = bundle
    api.runtime = runtime
    api.audit = AuditLog.for_home(tmp_path, tenant_id="acme", actor="nova-control")
    return api


class _Runtime:
    """A runtime that records what it was asked to do."""

    name = "stub"

    class capabilities:  # noqa: N801 — mirrors the real attribute shape
        scheduling = True

    def __init__(self, home):
        self.state_location = home
        self.created = []

    def validate_schedule(self, schedule: str) -> None:
        """The real adapter's parser, so these exercise the same refusals production does."""
        from nova.runtime.hermes.automations import validate_schedule

        validate_schedule(schedule)

    def create_automation(self, agent_id, compiled):
        from nova.runtime.base import AutomationView

        self.created.append((agent_id, compiled))
        return AutomationView(
            automation_id="job123", name=compiled.spec.title, agent_id=agent_id,
            schedule_display=compiled.spec.schedule, enabled=compiled.spec.enabled,
            created_at="2026-01-01T00:00:00Z",
        )


def test_create_compiles_before_the_runtime_sees_anything(tmp_path, bundle):
    runtime = _Runtime(tmp_path)
    api = _api(tmp_path, bundle, runtime)

    response = api.write(
        "/platform/v1/automations",
        Principal(name="alice", role="admin", via="token"),
        _spec(permissions=["read_inventory"]).to_dict(),
    )
    assert response.status == 201
    assert response.body["applied"] is True
    # The runtime received a compiled object, never a raw prompt.
    agent_id, compiled = runtime.created[0]
    assert agent_id == "operations"
    assert compiled.digest.startswith("sha256:")


def test_an_ungranted_permission_never_reaches_the_runtime(tmp_path, bundle):
    runtime = _Runtime(tmp_path)
    api = _api(tmp_path, bundle, runtime)

    response = api.write(
        "/platform/v1/automations",
        Principal(name="alice", role="admin", via="token"),
        _spec(permissions=["crm_refund"]).to_dict(),
    )
    assert response.status == 400
    assert "does not hold" in response.body["error"]["message"]
    assert runtime.created == [], "a refused declaration reached the scheduler"


def test_a_viewer_cannot_create(tmp_path, bundle):
    runtime = _Runtime(tmp_path)
    api = _api(tmp_path, bundle, runtime)

    response = api.write(
        "/platform/v1/automations",
        Principal(name="carol", role="viewer", via="token"),
        _spec().to_dict(),
    )
    assert response.status == 403
    assert runtime.created == []


@pytest.mark.parametrize("action", ["pause", "resume", "delete"])
def test_a_viewer_cannot_pause_resume_or_delete(tmp_path, bundle, action):
    runtime = _Runtime(tmp_path)
    api = _api(tmp_path, bundle, runtime)

    response = api.write(
        "/platform/v1/automations/job123/decide",
        Principal(name="carol", role="viewer", via="token"),
        {"action": action},
    )
    assert response.status == 403


def test_create_writes_intent_then_committed(tmp_path, bundle):
    runtime = _Runtime(tmp_path)
    api = _api(tmp_path, bundle, runtime)
    api.write(
        "/platform/v1/automations",
        Principal(name="alice", role="admin", via="token"),
        _spec(reason="SOX control 4.2").to_dict(),
    )

    events = [e for e in api.audit.read() if e.kind == "automation.declared"]
    assert [e.phase for e in events] == ["intent", "committed"]
    assert {e.actor for e in events} == {"alice"}
    # The intent carries the reviewed declaration, so a later reader can see what was
    # approved even if the commit never landed.
    assert events[0].detail["reason"] == "SOX control 4.2"
    assert events[0].digest == events[1].digest
    assert events[1].detail["runtime_job_id"] == "job123"


def test_a_refused_declaration_records_nothing(tmp_path, bundle):
    """Refused at compile time is refused before the write-ahead record: there was no
    intent to act on something the tenant could never have granted."""
    runtime = _Runtime(tmp_path)
    api = _api(tmp_path, bundle, runtime)
    api.write(
        "/platform/v1/automations",
        Principal(name="alice", role="admin", via="token"),
        _spec(permissions=["crm_refund"]).to_dict(),
    )
    assert [e for e in api.audit.read() if e.kind == "automation.declared"] == []


def test_a_runtime_failure_still_closes_the_intent(tmp_path, bundle):
    """Every intent reaches a terminal phase. Without this an exception leaves an
    ``intent`` with no ``committed`` and no ``failed``, and the log reads as though the
    act might have happened."""
    class _Exploding(_Runtime):
        def create_automation(self, agent_id, compiled):
            raise RuntimeError("cron store is read-only")

    runtime = _Exploding(tmp_path)
    api = _api(tmp_path, bundle, runtime)
    response = api.write(
        "/platform/v1/automations",
        Principal(name="alice", role="admin", via="token"),
        _spec().to_dict(),
    )
    assert response.status == 502
    events = [e for e in api.audit.read() if e.kind == "automation.declared"]
    assert [e.phase for e in events] == ["intent", "failed"]
    assert "read-only" in events[1].error


def test_the_registry_records_provenance(tmp_path, bundle):
    from nova.automations import registry

    runtime = _Runtime(tmp_path)
    api = _api(tmp_path, bundle, runtime)
    api.write(
        "/platform/v1/automations",
        Principal(name="alice", role="admin", via="token"),
        _spec(permissions=["read_inventory"], reason="quarter close").to_dict(),
    )

    entry = registry.governance(tmp_path, "job123", tenant_id="acme")
    assert entry is not None
    assert entry["declared_by"] == "alice"
    assert entry["reason"] == "quarter close"
    assert entry["permissions"] == ["read_inventory"]


def test_the_registry_is_tenant_checked(tmp_path, bundle):
    from nova.automations import registry

    runtime = _Runtime(tmp_path)
    api = _api(tmp_path, bundle, runtime)
    api.write(
        "/platform/v1/automations",
        Principal(name="alice", role="admin", via="token"),
        _spec().to_dict(),
    )
    assert registry.governance(tmp_path, "job123", tenant_id="globex") is None
    assert registry.governance(tmp_path, "job123", tenant_id="acme") is not None


def test_a_damaged_registry_does_not_break_reads(tmp_path, bundle):
    from nova.automations import registry

    path = registry.registry_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json")
    assert registry.governance(tmp_path, "anything") is None


# ---------------------------------------------------------------------------
# Credential secrecy
# ---------------------------------------------------------------------------

def test_no_credential_reaches_a_response(tmp_path, bundle):
    """The compiled form and the API response carry a schedule and an instruction —
    never a secret. Asserted rather than assumed because the objective is free text and
    an operator could paste one in."""
    runtime = _Runtime(tmp_path)
    api = _api(tmp_path, bundle, runtime)
    response = api.write(
        "/platform/v1/automations",
        Principal(name="alice", role="admin", via="token"),
        _spec().to_dict(),
    )
    payload = json.dumps(response.body)
    for marker in ("TELEGRAM_BOT_TOKEN", "ACME_LLM_KEY", "token_sha256", "api_key"):
        assert marker not in payload
