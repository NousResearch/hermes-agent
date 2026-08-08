import pytest

from agent.executive import build_objective_services


def enabled_config():
    return {"goals": {"evidence_pack": {"enabled": True}}}


def disabled_config():
    return {"goals": {"evidence_pack": {"enabled": False}}}


def assert_disabled_normal(services):
    assert services.evidence_pack_engine is None
    assert services.evidence_pack_status == "disabled"
    assert services.evidence_pack_degrade_reason is None
    assert services.evidence_pack_error_type is None
    assert services.evidence_pack_enabled is False


def assert_invalid_config(services):
    assert services.evidence_pack_engine is None
    assert services.evidence_pack_status == "disabled"
    assert services.evidence_pack_degrade_reason == "invalid_config"
    assert services.evidence_pack_error_type is None
    assert services.evidence_pack_enabled is False


class RecordingFactory:
    def __init__(self, engine=None, exc=None):
        self.engine = object() if engine is None else engine
        self.exc = exc
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.engine


def test_empty_config_is_disabled_normal():
    assert_disabled_normal(build_objective_services(session_id="s1", config={}))
    assert_disabled_normal(build_objective_services(session_id="s1", config=None))


def test_missing_goals_is_disabled_normal():
    services = build_objective_services(session_id="s1", config={"other": {}})

    assert_disabled_normal(services)


def test_missing_evidence_pack_is_disabled_normal():
    services = build_objective_services(session_id="s1", config={"goals": {}})

    assert_disabled_normal(services)


def test_enabled_false_is_disabled_normal():
    services = build_objective_services(session_id="s1", config=disabled_config())

    assert_disabled_normal(services)


def test_enabled_true_with_factory_makes_evidence_pack_available():
    engine = object()
    factory = RecordingFactory(engine=engine)

    services = build_objective_services(
        session_id="s1",
        config=enabled_config(),
        evidence_pack_engine_factory=factory,
    )

    assert services.evidence_pack_engine is engine
    assert services.evidence_pack_status == "available"
    assert services.evidence_pack_degrade_reason is None
    assert services.evidence_pack_error_type is None
    assert services.evidence_pack_enabled is True


def test_factory_is_not_called_when_disabled():
    factory = RecordingFactory()

    build_objective_services(
        session_id="s1",
        config=disabled_config(),
        evidence_pack_engine_factory=factory,
    )

    assert factory.calls == []


def test_factory_is_called_exactly_once_with_expected_dependencies():
    config = enabled_config()
    sources = object()
    storage = object()
    audit_sink = object()
    factory = RecordingFactory()

    build_objective_services(
        session_id="  preserved session  ",
        config=config,
        sources=sources,
        storage=storage,
        audit_sink=audit_sink,
        evidence_pack_engine_factory=factory,
    )

    assert factory.calls == [
        {
            "session_id": "  preserved session  ",
            "config": config,
            "sources": sources,
            "storage": storage,
            "audit_sink": audit_sink,
        }
    ]


def test_dependencies_are_preserved_by_identity():
    sources = object()
    storage = object()
    audit_sink = object()

    services = build_objective_services(
        session_id="s1",
        config={},
        sources=sources,
        storage=storage,
        audit_sink=audit_sink,
    )

    assert services.sources is sources
    assert services.storage is storage
    assert services.audit_sink is audit_sink


def test_two_session_ids_produce_independent_services():
    first = build_objective_services(session_id="s1", config={})
    second = build_objective_services(session_id="s2", config={})

    assert first is not second
    assert first.session_id == "s1"
    assert second.session_id == "s2"


def test_enabled_true_without_factory_degrades():
    services = build_objective_services(session_id="s1", config=enabled_config())

    assert services.evidence_pack_engine is None
    assert services.evidence_pack_status == "degraded"
    assert services.evidence_pack_degrade_reason == "factory_missing"
    assert services.evidence_pack_error_type is None
    assert services.evidence_pack_enabled is False


def test_factory_exception_degrades_and_only_keeps_exception_type():
    factory = RecordingFactory(exc=RuntimeError("sensitive detail"))

    services = build_objective_services(
        session_id="s1",
        config=enabled_config(),
        evidence_pack_engine_factory=factory,
    )

    assert services.evidence_pack_engine is None
    assert services.evidence_pack_status == "degraded"
    assert services.evidence_pack_degrade_reason == "factory_error"
    assert services.evidence_pack_error_type == "RuntimeError"


@pytest.mark.parametrize("session_id", ["", "   ", None, 123])
def test_invalid_session_id_values_are_rejected(session_id):
    with pytest.raises(ValueError, match="session_id is required"):
        build_objective_services(session_id=session_id, config={})


def test_non_mapping_root_is_invalid_config_and_does_not_call_factory():
    factory = RecordingFactory()

    for config in ([], 1, "enabled"):
        services = build_objective_services(
            session_id="s1",
            config=config,
            evidence_pack_engine_factory=factory,
        )
        assert_invalid_config(services)

    assert factory.calls == []


    factory = RecordingFactory()

    services = build_objective_services(
        session_id="s1",
        config={"goals": []},
        evidence_pack_engine_factory=factory,
    )

    assert_invalid_config(services)
    assert factory.calls == []


def test_invalid_evidence_pack_shape_degrades_without_calling_factory():
    factory = RecordingFactory()

    services = build_objective_services(
        session_id="s1",
        config={"goals": {"evidence_pack": []}},
        evidence_pack_engine_factory=factory,
    )

    assert_invalid_config(services)
    assert factory.calls == []


@pytest.mark.parametrize("enabled", ["yes", 1, 0, [], {}])
def test_non_bool_enabled_values_are_invalid_config_and_do_not_call_factory(enabled):
    factory = RecordingFactory()

    services = build_objective_services(
        session_id="s1",
        config={"goals": {"evidence_pack": {"enabled": enabled}}},
        evidence_pack_engine_factory=factory,
    )

    assert_invalid_config(services)
    assert factory.calls == []


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_factory_does_not_catch_base_exceptions(exc_type):
    factory = RecordingFactory(exc=exc_type())

    with pytest.raises(exc_type):
        build_objective_services(
            session_id="s1",
            config=enabled_config(),
            evidence_pack_engine_factory=factory,
        )


def test_each_call_returns_a_new_instance():
    first = build_objective_services(session_id="s1", config={})
    second = build_objective_services(session_id="s1", config={})

    assert first is not second


# ─────────────────────────────────────────────────────────────────────
# B1-E4: Storage unavailable degrade path
# ─────────────────────────────────────────────────────────────────────


def test_storage_unavailable_exception_is_caught_as_storage_unavailable_reason():
    """The typed ``EvidencePackStorageUnavailable`` exception surfaces as
    ``storage_unavailable`` / class name, NOT as the generic
    ``factory_error`` envelope.
    """
    from agent.executive.services import EvidencePackStorageUnavailable

    class TypedFactory:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            raise EvidencePackStorageUnavailable("no SessionDB on hand")

    factory = TypedFactory()

    services = build_objective_services(
        session_id="s1",
        config=enabled_config(),
        evidence_pack_engine_factory=factory,
    )

    assert services.evidence_pack_engine is None
    assert services.evidence_pack_status == "degraded"
    assert services.evidence_pack_degrade_reason == "storage_unavailable"
    assert services.evidence_pack_error_type == "EvidencePackStorageUnavailable"
    assert services.evidence_pack_enabled is False


def test_storage_unavailable_is_caught_before_generic_exception():
    """A factory that raises ``EvidencePackStorageUnavailable`` (a
    ``RuntimeError`` subclass) MUST be matched by the typed handler
    rather than the generic ``Exception`` catch-all — otherwise the
    more specific ``storage_unavailable`` reason would be lost.
    """
    from agent.executive.services import EvidencePackStorageUnavailable

    raised = []

    class TypedFactory:
        def __call__(self, **kwargs):
            raised.append("called")
            raise EvidencePackStorageUnavailable("unavailable")

    services = build_objective_services(
        session_id="s1",
        config=enabled_config(),
        evidence_pack_engine_factory=TypedFactory(),
    )

    assert raised == ["called"]
    assert services.evidence_pack_degrade_reason == "storage_unavailable"
    assert services.evidence_pack_error_type == "EvidencePackStorageUnavailable"


def test_evidence_pack_degrade_reason_literal_includes_storage_unavailable():
    """The ``EvidencePackDegradeReason`` Literal must list the new reason."""
    from typing import get_args

    from agent.executive.services import EvidencePackDegradeReason

    assert "storage_unavailable" in get_args(EvidencePackDegradeReason)
    # And the prior reasons must still be present.
    assert "factory_error" in get_args(EvidencePackDegradeReason)
    assert "factory_missing" in get_args(EvidencePackDegradeReason)
    assert "invalid_config" in get_args(EvidencePackDegradeReason)


def test_keyboard_interrupt_and_system_exit_still_propagate():
    """Base exceptions must NOT be caught — even by the typed handler."""
    from agent.executive.services import EvidencePackStorageUnavailable

    for exc_cls in (KeyboardInterrupt, SystemExit):
        class RaiseFactory:
            def __call__(self, **kwargs):
                raise exc_cls()

        with pytest.raises(exc_cls):
            build_objective_services(
                session_id="s1",
                config=enabled_config(),
                evidence_pack_engine_factory=RaiseFactory(),
            )

    # And a typed storage exception must NOT be confused for the base
    # classes — it is a RuntimeError and must be caught by the typed
    # handler. Sanity check on the inheritance.
    assert issubclass(EvidencePackStorageUnavailable, RuntimeError)


def test_dependencies_are_preserved_by_identity_on_storage_failure():
    """Even when storage is unavailable, the borrowed dependencies are
    preserved by identity on the returned ``ObjectiveServices``.
    """
    from agent.executive.services import EvidencePackStorageUnavailable

    sources = object()
    storage = object()
    audit_sink = object()

    class TypedFactory:
        def __call__(self, **kwargs):
            raise EvidencePackStorageUnavailable()

    services = build_objective_services(
        session_id="s1",
        config=enabled_config(),
        sources=sources,
        storage=storage,
        audit_sink=audit_sink,
        evidence_pack_engine_factory=TypedFactory(),
    )

    assert services.sources is sources
    assert services.storage is storage
    assert services.audit_sink is audit_sink


def test_typed_storage_exception_is_owned_by_services_module():
    """The typed exception lives in ``agent.executive.services``."""
    import agent.executive.services as services_module
    from agent.executive.services import EvidencePackStorageUnavailable

    assert services_module.EvidencePackStorageUnavailable is EvidencePackStorageUnavailable
    assert EvidencePackStorageUnavailable.__module__ == "agent.executive.services"
