"""The admin doors answer by the backend's declared ``physical_admin`` capability."""

from __future__ import annotations

from pathlib import Path

from persistence.core.admin import PHYSICAL_ADMIN, AdminUnsupported, HealthCheckUnavailable, ProfileRepairAdmin, StoreAdmin


def test_both_doors_answer_by_the_declared_capability(backend, backend_name, scope_a):
    store_answers = [backend.open_store_admin(scope_a), backend.open_store_admin(scope_a, read_only=False),
                     backend.open_store_admin(Path(scope_a.hermes_home_key) / "state.db")]
    repair_answers = [backend.open_profile_repair(scope_a, read_only=True), backend.open_profile_repair(scope_a, read_only=False)]
    if backend.capabilities.physical_admin:
        # CONTRACT: a declaring backend answers every target form with a handle, never unsupported.
        assert all(isinstance(answer, StoreAdmin) for answer in store_answers)
        assert all(isinstance(answer, ProfileRepairAdmin) for answer in repair_answers)
    else:
        # CONTRACT: any other backend answers both doors with the capability and its own kind.
        unsupported = AdminUnsupported(PHYSICAL_ADMIN, backend_name)
        assert store_answers == [unsupported] * 3
        assert repair_answers == [unsupported] * 2


def test_the_unsupported_answer_is_a_value_no_handle_protocol_accepts():
    answer = AdminUnsupported(PHYSICAL_ADMIN, "in_memory")
    assert (answer.capability, answer.backend_kind) == ("physical_admin", "in_memory")
    assert answer == AdminUnsupported("physical_admin", "in_memory")
    assert hash(answer) == hash(AdminUnsupported("physical_admin", "in_memory"))
    assert not isinstance(answer, StoreAdmin) and not isinstance(answer, ProfileRepairAdmin)


def test_an_unavailable_health_check_is_a_reason_that_is_neither_healthy_nor_damage():
    answer = HealthCheckUnavailable("could not make the probe copy (database or disk is full)")
    # CONTRACT: ``opens_cleanly`` answers a probe that could not run with this ``str`` subclass, so a caller that
    # prints reasons keeps working while one that branches tells it from None (healthy) and from damage text.
    assert isinstance(answer, str) and answer == "could not make the probe copy (database or disk is full)"
    assert not isinstance("database disk image is malformed", HealthCheckUnavailable)
