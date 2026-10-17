"""Tests for the enterprise core: resource model, store invariants, audit."""

from __future__ import annotations

import pytest

from enterprise.audit import AuditLog
from enterprise.contracts import DriverRegistry
from enterprise.errors import (
    ConflictError,
    DriverError,
    NotFoundError,
    ScopeError,
    ValidationError,
)
from enterprise.resources import (
    Kind,
    NamespacePhase,
    Resource,
    ResourceMeta,
    validate_name,
)
from enterprise.store import ResourceStore


@pytest.fixture()
def store(tmp_path):
    s = ResourceStore(tmp_path / "occ.db")
    yield s
    s.close()


def mk(kind: Kind, name: str, namespace=None, spec=None, status=None) -> Resource:
    return Resource(
        meta=ResourceMeta(kind=kind.value, name=name, namespace=namespace),
        spec=spec or {},
        status=status or {},
    )


def ready_ns(store: ResourceStore, name: str = "acme") -> None:
    store.create(mk(Kind.NAMESPACE, name))
    store.update_status(Kind.NAMESPACE, name, None,
                        {"phase": NamespacePhase.READY.value})


# ---------------------------------------------------------------------------
# Names + scope
# ---------------------------------------------------------------------------

class TestNamesAndScope:
    def test_valid_dns_label(self):
        assert validate_name("acme-prod-1") == "acme-prod-1"

    @pytest.mark.parametrize("bad", ["", "UPPER", "-lead", "trail-", "a" * 64,
                                     "under_score", "dot.name"])
    def test_invalid_names(self, bad):
        with pytest.raises(ValidationError):
            validate_name(bad)

    def test_namespace_scoped_kind_requires_namespace(self):
        with pytest.raises(ValidationError):
            mk(Kind.AGENT, "bot", spec={"harness": "h", "configuration": "c"}).validate()

    def test_installation_kind_rejects_namespace(self):
        with pytest.raises(ValidationError):
            mk(Kind.HARNESS, "hermes", namespace="acme",
               spec={"version": "1.0", "image": "img"}).validate()

    def test_restriction_valid_in_both_scopes(self):
        rule = {"rule": {"deny": ["deploy:Agent"]}}
        mk(Kind.RESTRICTION, "no-deploys", spec=rule).validate()
        mk(Kind.RESTRICTION, "no-deploys", namespace="acme", spec=rule).validate()


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------

class TestSpecValidation:
    def test_configuration_rejects_embedded_secret(self):
        res = mk(Kind.CONFIGURATION, "cfg", namespace="acme",
                 spec={"config": {"model": {"api_key": "sk-123"}}})
        with pytest.raises(ValidationError, match="secret-like"):
            res.validate()

    def test_configuration_allows_secret_reference_shape(self):
        res = mk(Kind.CONFIGURATION, "cfg", namespace="acme",
                 spec={"config": {"model": {"api_key": {"secretRef": "llm-key"}}}})
        res.validate()

    def test_secret_must_not_carry_value(self):
        res = mk(Kind.SECRET, "llm-key", namespace="acme",
                 spec={"broker": "vault", "key": "prod/llm", "value": "sk-x"})
        with pytest.raises(ValidationError, match="must not contain secret values"):
            res.validate()

    def test_sandbox_allowlist_requires_entries(self):
        res = mk(Kind.SANDBOX_POLICY, "locked", namespace="acme",
                 spec={"network": "egress-allowlist"})
        with pytest.raises(ValidationError, match="egressAllow"):
            res.validate()

    def test_restriction_cannot_be_empty(self):
        res = mk(Kind.RESTRICTION, "noop", spec={"rule": {"deny": []}})
        with pytest.raises(ValidationError, match="narrow, never grant"):
            res.validate()


# ---------------------------------------------------------------------------
# Store invariants
# ---------------------------------------------------------------------------

class TestStore:
    def test_create_get_roundtrip(self, store):
        ready_ns(store)
        store.create(mk(Kind.CONFIGURATION, "cfg", namespace="acme",
                        spec={"config": {"model": "hermes-4"}}))
        got = store.get(Kind.CONFIGURATION, "cfg", "acme")
        assert got.spec["config"]["model"] == "hermes-4"

    def test_duplicate_identity_conflicts(self, store):
        ready_ns(store)
        res = mk(Kind.CHANNEL, "tg", namespace="acme", spec={"platform": "telegram"})
        store.create(res)
        with pytest.raises(ConflictError):
            store.create(mk(Kind.CHANNEL, "tg", namespace="acme",
                            spec={"platform": "telegram"}))

    def test_namespaced_resource_requires_existing_namespace(self, store):
        with pytest.raises(ScopeError, match="missing Namespace"):
            store.create(mk(Kind.CHANNEL, "tg", namespace="ghost",
                            spec={"platform": "telegram"}))

    def test_terminating_namespace_admits_nothing(self, store):
        ready_ns(store)
        store.update_status(Kind.NAMESPACE, "acme", None,
                            {"phase": NamespacePhase.TERMINATING.value})
        with pytest.raises(ScopeError, match="terminating"):
            store.create(mk(Kind.CHANNEL, "tg", namespace="acme",
                            spec={"platform": "telegram"}))

    def test_cross_namespace_reference_denied(self, store):
        ready_ns(store, "acme")
        ready_ns(store, "globex")
        store.create(mk(Kind.HARNESS, "hermes",
                        spec={"version": "1.0", "image": "ghcr.io/x"}))
        # configuration lives in globex, agent in acme -> must fail
        store.create(mk(Kind.CONFIGURATION, "cfg", namespace="globex",
                        spec={"config": {}}))
        with pytest.raises(ScopeError, match="cross-namespace"):
            store.create(mk(Kind.AGENT, "bot", namespace="acme",
                            spec={"harness": "hermes", "configuration": "cfg"}))

    def test_agent_requires_existing_harness(self, store):
        ready_ns(store)
        store.create(mk(Kind.CONFIGURATION, "cfg", namespace="acme",
                        spec={"config": {}}))
        with pytest.raises(NotFoundError):
            store.create(mk(Kind.AGENT, "bot", namespace="acme",
                            spec={"harness": "ghost", "configuration": "cfg"}))

    def test_generation_conflict(self, store):
        ready_ns(store)
        store.create(mk(Kind.CONFIGURATION, "cfg", namespace="acme",
                        spec={"config": {"a": 1}}))
        first = store.get(Kind.CONFIGURATION, "cfg", "acme")
        second = store.get(Kind.CONFIGURATION, "cfg", "acme")
        first.spec["config"]["a"] = 2
        store.update_spec(first)
        second.spec["config"]["a"] = 3
        with pytest.raises(ConflictError, match="generation conflict"):
            store.update_spec(second)

    def test_revision_spec_immutable(self, store):
        ready_ns(store)
        rev = mk(Kind.AGENT_REVISION, "bot-rev-1", namespace="acme", spec={
            "agent": "bot", "agentUid": "u1", "workloadIdentity": "wi-bot",
            "harness": {"name": "hermes", "version": "1.0"},
            "configuration": {"model": "hermes-4"},
            "computeDriver": "kubernetes", "sandboxDriver": "k8s-baseline",
        })
        store.create(rev)
        rev2 = store.get(Kind.AGENT_REVISION, "bot-rev-1", "acme")
        rev2.spec["configuration"] = {"model": "other"}
        with pytest.raises(ValidationError, match="immutable"):
            store.update_spec(rev2)
        # status updates remain allowed
        store.update_status(Kind.AGENT_REVISION, "bot-rev-1", "acme",
                            {"phase": "Active"})

    def test_delete_blocked_by_dependents(self, store):
        ready_ns(store)
        store.create(mk(Kind.HARNESS, "hermes",
                        spec={"version": "1.0", "image": "img"}))
        store.create(mk(Kind.CONFIGURATION, "cfg", namespace="acme",
                        spec={"config": {}}))
        store.create(mk(Kind.AGENT, "bot", namespace="acme",
                        spec={"harness": "hermes", "configuration": "cfg"}))
        with pytest.raises(ConflictError, match="referenced by"):
            store.delete(Kind.CONFIGURATION, "cfg", "acme")
        with pytest.raises(ConflictError, match="referenced by"):
            store.delete(Kind.HARNESS, "hermes")

    def test_namespace_delete_requires_drain(self, store):
        ready_ns(store)
        store.create(mk(Kind.CHANNEL, "tg", namespace="acme",
                        spec={"platform": "telegram"}))
        with pytest.raises(ConflictError, match="drain"):
            store.delete(Kind.NAMESPACE, "acme")
        store.delete(Kind.CHANNEL, "tg", "acme")
        store.delete(Kind.NAMESPACE, "acme")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class TestAudit:
    def test_record_and_query(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        log.record(actor="kevin@acme.com", actor_kind="principal",
                   action="hermes.agents.deploy", outcome="allow",
                   kind="Agent", namespace="acme", resource="bot")
        rows = log.query(kind="Agent", namespace="acme")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "allow"
        log.close()

    def test_refuses_secretlike_detail(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        with pytest.raises(ValueError, match="never carry secrets"):
            log.record(actor="x", actor_kind="principal", action="a",
                       outcome="allow", detail={"api_key": "sk-live-123"})
        log.close()

    def test_rejects_unknown_outcome(self, tmp_path):
        log = AuditLog(tmp_path / "audit.db")
        with pytest.raises(ValueError):
            log.record(actor="x", actor_kind="principal", action="a",
                       outcome="maybe")
        log.close()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_single_selection_per_capability(self):
        reg = DriverRegistry()

        class FakeDriver:
            name = "kubernetes"

        reg.select("compute", FakeDriver())
        with pytest.raises(ValueError, match="already has a selected"):
            class Other:
                name = "nomad"
            reg.select("compute", Other())

    def test_unselected_capability_fails_closed(self):
        reg = DriverRegistry()
        with pytest.raises(DriverError):
            reg.get("compute")
