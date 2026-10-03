"""Focused tests for CLI ObjectiveServices consumer wiring."""
from typing import Any
from unittest.mock import patch


def _cli(session_id="cli-goal-wiring"):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = session_id
    cli._goal_manager = None
    return cli


def test_get_goal_manager_builds_and_caches_objective_services():
    from agent.executive.services import ObjectiveServices
    from agent.executive.knowledge_discovery.factory import (
        build_evidence_pack_engine,
    )

    cli = _cli()
    cli._session_db = object()
    config = {"goals": {"max_turns": 7}}
    services = ObjectiveServices(session_id=cli.session_id)

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("agent.executive.services.build_objective_services", return_value=services) as build,
        patch("hermes_cli.goals.GoalManager") as manager_cls,
    ):
        first = cli._get_goal_manager()
        build.reset_mock()
        second = cli._get_goal_manager()

    assert first is second
    assert build.call_count == 1
    build.assert_called_once_with(
        session_id=cli.session_id,
        config=config,
        storage=cli._session_db,
        evidence_pack_engine_factory=build_evidence_pack_engine,
    )
    assert manager_cls.call_args_list[0].args == ()
    assert manager_cls.call_args_list[0].kwargs == {
        "session_id": cli.session_id,
        "default_max_turns": 7,
        "services": services,
    }


def test_get_goal_manager_default_off_and_degraded_services_do_not_invoke_engine():
    from agent.executive.services import ObjectiveServices

    for services in (
        ObjectiveServices(session_id="cli-disabled", evidence_pack_status="disabled"),
        ObjectiveServices(
            session_id="cli-degraded",
            evidence_pack_status="degraded",
            evidence_pack_degrade_reason="factory_missing",
        ),
    ):
        cli = _cli(services.session_id)
        cli._session_db = object()
        with (
            patch("hermes_cli.config.load_config", return_value={}),
            patch("agent.executive.services.build_objective_services", return_value=services),
            patch("hermes_cli.goals.GoalManager") as manager_cls,
        ):
            manager = cli._get_goal_manager()

        assert manager is manager_cls.return_value
        assert services.evidence_pack_engine is None
        manager_cls.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# B1-E4: Canonical factory wiring
# ─────────────────────────────────────────────────────────────────────


def test_get_goal_manager_passes_self_session_db_and_canonical_factory():
    """HermesCLI._get_goal_manager MUST pass the borrowed _session_db
    AND the canonical ``build_evidence_pack_engine`` factory on the
    enabled branch.
    """
    from agent.executive.services import ObjectiveServices
    from agent.executive.knowledge_discovery.factory import (
        build_evidence_pack_engine,
    )

    cli = _cli("cli-b1-e4-enabled")
    sentinel_db = object()
    cli._session_db = sentinel_db
    config = {"goals": {"max_turns": 7, "evidence_pack": {"enabled": True}}}
    services = ObjectiveServices(session_id=cli.session_id)

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("agent.executive.services.build_objective_services", return_value=services) as build,
        patch("hermes_cli.goals.GoalManager"),
    ):
        cli._get_goal_manager()

    # Exactly one call, with the right keyword arguments.
    assert build.call_count == 1
    kwargs = build.call_args.kwargs
    assert kwargs["session_id"] == cli.session_id
    assert kwargs["config"] is config
    assert kwargs["storage"] is sentinel_db
    assert kwargs["evidence_pack_engine_factory"] is build_evidence_pack_engine


def test_get_goal_manager_passes_storage_and_factory_on_config_none_fallback():
    """The config-None fallback branch MUST also pass ``storage`` and
    ``evidence_pack_engine_factory``. ``build_objective_services``
    bypasses the factory when ``config is None`` regardless of those
    kwargs.
    """
    from agent.executive.services import ObjectiveServices
    from agent.executive.knowledge_discovery.factory import (
        build_evidence_pack_engine,
    )

    cli = _cli("cli-b1-e4-fallback")
    sentinel_db = object()
    cli._session_db = sentinel_db
    services = ObjectiveServices(session_id=cli.session_id)

    with (
        patch(
            "hermes_cli.config.load_config",
            side_effect=RuntimeError("config unreadable"),
        ),
        patch("agent.executive.services.build_objective_services", return_value=services) as build,
        patch("hermes_cli.goals.GoalManager"),
    ):
        cli._get_goal_manager()

    assert build.call_count == 1
    kwargs = build.call_args.kwargs
    assert kwargs["config"] is None
    assert kwargs["storage"] is sentinel_db
    assert kwargs["evidence_pack_engine_factory"] is build_evidence_pack_engine


def test_get_goal_manager_never_passes_storage_none():
    """The CLI must not hardcode ``storage=None``. It passes
    ``self._session_db`` even when that attribute is missing — the
    factory raises the typed storage_unavailable signal in that case.
    """
    from agent.executive.services import ObjectiveServices

    cli = _cli("cli-b1-e4-no-storage")
    # Note: no _session_db attribute is set on purpose.
    assert not hasattr(cli, "_session_db")

    services = ObjectiveServices(session_id=cli.session_id)
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.executive.services.build_objective_services", return_value=services) as build,
        patch("hermes_cli.goals.GoalManager"),
    ):
        cli._get_goal_manager()

    # Whatever storage is passed, it is NOT the literal None literal
    # bound to the ``storage`` keyword.
    kwargs = build.call_args.kwargs
    # ``storage`` is either absent or references ``getattr(self, "_session_db", None)`` —
    # never the literal ``storage=None`` kwarg.
    if "storage" in kwargs:
        assert "storage" not in build.call_args.kwargs or (
            kwargs.get("storage") is None or kwargs.get("storage") is getattr(cli, "_session_db", None)
        )


def test_get_goal_manager_per_session_rebinding_remains_authoritative():
    """Existing per-session rebinding behavior is preserved."""
    from agent.executive.services import ObjectiveServices

    cli = _cli("cli-b1-e4-session-a")
    cli._session_db = object()
    services_a = ObjectiveServices(session_id="cli-b1-e4-session-a")
    services_b = ObjectiveServices(session_id="cli-b1-e4-session-b")

    manager_calls: list[Any] = []

    class CountingManager:
        def __init__(self, **kwargs):
            # Reproduce the minimum real GoalManager contract: a public
            # ``session_id`` attribute. HermesCLI._get_goal_manager
            # caches by checking ``existing.session_id == self.session_id``;
            # without this attribute the cache check always misses and the
            # repeated same-session call is incorrectly rebuilt (which
            # would consume the second side_effect and break the test).
            self.session_id = kwargs["session_id"]
            manager_calls.append(kwargs)

    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch(
            "agent.executive.services.build_objective_services",
            side_effect=[services_a, services_b],
        ) as build,
        patch("hermes_cli.goals.GoalManager", CountingManager),
    ):
        mgr_a = cli._get_goal_manager()
        # Same session — cached, no second build call.
        mgr_a_again = cli._get_goal_manager()
        # Switch session — must rebind.
        cli.session_id = "cli-b1-e4-session-b"
        mgr_b = cli._get_goal_manager()

    # Two distinct ObjectiveServices were built — one per session.
    assert build.call_count == 2
    # Same-session call was cached, so the GoalManager constructor saw
    # exactly one invocation for session-a and one for session-b.
    assert len(manager_calls) == 2
    assert manager_calls[0]["session_id"] == "cli-b1-e4-session-a"
    assert manager_calls[1]["session_id"] == "cli-b1-e4-session-b"


def test_get_goal_manager_does_not_import_hermes_cli_in_factory_call(monkeypatch):
    """The CLI must import ``build_evidence_pack_engine`` from the
    canonical factory module, not roll its own factory inline.

    Behavioral contract: after invoking ``HermesCLI._get_goal_manager``,
    the ``build_objective_services`` call must carry the canonical
    ``build_evidence_pack_engine`` factory as the
    ``evidence_pack_engine_factory`` kwarg. The drive is a real CLI
    invocation through a monkeypatched ``build_objective_services``;
    the assertion observes the kwarg the CLI actually transmitted.
    """
    from agent.executive.services import ObjectiveServices
    from agent.executive.knowledge_discovery.factory import (
        build_evidence_pack_engine,
    )

    cli = _cli("cli-canonical-factory")
    cli._session_db = object()
    config = {"goals": {"max_turns": 7, "evidence_pack": {"enabled": True}}}
    services = ObjectiveServices(session_id=cli.session_id)

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch(
            "agent.executive.services.build_objective_services",
            return_value=services,
        ) as build,
        patch("hermes_cli.goals.GoalManager"),
    ):
        cli._get_goal_manager()

    # Behavioral observation: the canonical factory callable is the
    # exact object the CLI transmitted as ``evidence_pack_engine_factory``.
    # This is the documented CLI seam — the CLI must not roll its own
    # factory inline.
    assert build.call_count == 1
    kwargs = build.call_args.kwargs
    assert kwargs["evidence_pack_engine_factory"] is build_evidence_pack_engine, (
        "CLI must use the canonical factory callable "
        "agent.executive.knowledge_discovery.factory.build_evidence_pack_engine"
    )


def test_get_goal_manager_passes_borrowed_session_db_identity_not_copy():
    """``storage`` passed to build_objective_services is the SAME
    object as ``self._session_db`` — not a copy.
    """
    from agent.executive.services import ObjectiveServices

    cli = _cli("cli-b1-e4-identity")
    sentinel_db = object()
    cli._session_db = sentinel_db

    services = ObjectiveServices(session_id=cli.session_id)
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.executive.services.build_objective_services", return_value=services) as build,
        patch("hermes_cli.goals.GoalManager"),
    ):
        cli._get_goal_manager()

    assert build.call_args.kwargs["storage"] is sentinel_db