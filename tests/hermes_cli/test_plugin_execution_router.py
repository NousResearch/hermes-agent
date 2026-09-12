from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest
import yaml

from agent.execution_router import (
    ExecutionKind,
    ExecutionRouteDecisionV1,
    ExecutionRouterProviderDescriptorV1,
)
from hermes_cli.plugin_capabilities import (
    EXECUTION_ROUTING_CAPABILITY,
    ExecutionRouterConsentState,
    execution_router_consent_status,
    record_consent,
    record_execution_router_consent,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _provider(plugin_id="router-plugin", plugin_version="1.2.3", provider_id="router-a"):
    class Provider:
        descriptor = ExecutionRouterProviderDescriptorV1(
            plugin_id=plugin_id,
            plugin_version=plugin_version,
            provider_id=provider_id,
            contract_version="1.0",
            supported_execution_kinds=tuple(ExecutionKind),
        )

        def resolve_execution_route(self, request, cancellation):
            return ExecutionRouteDecisionV1.pass_through(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
            )

    return Provider()


def _context(manager=None, version="1.2.3"):
    manager = manager or PluginManager()
    manifest = PluginManifest(
        name="router-plugin",
        version=version,
        source="user",
        key="router-plugin",
        capabilities=[EXECUTION_ROUTING_CAPABILITY],
    )
    return PluginContext(manifest, manager), manager


def test_generic_grant_only_records_pending_without_callable_or_ledger(hermes_home):
    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    ctx, manager = _context()

    result = ctx.register_execution_router(_provider())

    assert result.status == "consent_required"
    assert manager.get_execution_router_registration() is None
    assert manager._ownership_ledger == {}
    status = execution_router_consent_status("router-plugin")
    assert status.state is ExecutionRouterConsentState.PENDING
    assert status.subject.provider_id == "router-a"


def test_exact_consent_activates_only_on_subsequent_registration(hermes_home):
    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    first, _ = _context()
    pending = first.register_execution_router(_provider()).subject
    record_execution_router_consent(pending)

    second, manager = _context()
    result = second.register_execution_router(_provider())

    assert result.status == "active"
    assert manager.get_execution_router_registration().provider.descriptor.provider_id == "router-a"
    assert execution_router_consent_status("router-plugin").state is ExecutionRouterConsentState.CONSENTED
    assert manager._ownership_ledger["router-plugin"][0].kind == "execution_router"


def test_exact_router_consent_is_stale_when_live_generic_grant_is_revoked(hermes_home):
    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    ctx, _ = _context()
    subject = ctx.register_execution_router(_provider()).subject
    record_execution_router_consent(subject)
    config_path = hermes_home / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["plugins"]["entries"]["router-plugin"]["granted_capabilities"] = []
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    status = execution_router_consent_status("router-plugin", subject)

    assert status.state is ExecutionRouterConsentState.STALE
    from hermes_cli.plugins_cmd import _run_execution_router_consent
    console = MagicMock()
    with patch("hermes_cli.plugins_cmd._is_tty", return_value=True), patch(
        "hermes_cli.plugins_cmd._ask_yes", return_value=True
    ) as ask:
        assert _run_execution_router_consent(console, "router-plugin") is False
    ask.assert_not_called()


def test_cli_rejects_pending_router_subject_when_current_manifest_version_changed(hermes_home):
    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    ctx, _ = _context()
    ctx.register_execution_router(_provider())
    from hermes_cli.plugins_cmd import _run_execution_router_consent

    console = MagicMock()
    current_entry = ("router-plugin", "1.2.4", "", "user", hermes_home, "router-plugin")
    with patch("hermes_cli.plugins_cmd._find_plugin_entry", return_value=current_entry), patch(
        "hermes_cli.plugins_cmd._declared_capabilities_for_key",
        return_value=[EXECUTION_ROUTING_CAPABILITY],
    ), patch("hermes_cli.plugins_cmd._is_tty", return_value=True), patch(
        "hermes_cli.plugins_cmd._ask_yes", return_value=True
    ) as ask:
        assert _run_execution_router_consent(console, "router-plugin") is False
    ask.assert_not_called()


def test_single_slot_conflict_and_generation_safe_unload(hermes_home):
    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    first, manager = _context()
    subject = first.register_execution_router(_provider()).subject
    record_execution_router_consent(subject)
    active = first.register_execution_router(_provider())
    old = manager.get_execution_router_registration()

    other_manifest = PluginManifest(
        name="other-plugin",
        version="1.0",
        source="user",
        key="other-plugin",
        capabilities=[EXECUTION_ROUTING_CAPABILITY],
    )
    record_consent("other-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    other = PluginContext(other_manifest, manager)
    pending = other.register_execution_router(_provider("other-plugin", "1.0", "router-b")).subject
    record_execution_router_consent(pending)
    with pytest.raises(ValueError, match="already active"):
        other.register_execution_router(_provider("other-plugin", "1.0", "router-b"))

    active.registration.dispose()
    assert manager.get_execution_router_registration() is None
    assert old.is_current(old.generation) is False


def test_simultaneous_registration_claims_exactly_one_active_slot(hermes_home):
    manager = PluginManager()
    contexts = []
    providers = []
    for plugin_id, provider_id in (("router-a", "provider-a"), ("router-b", "provider-b")):
        manifest = PluginManifest(
            name=plugin_id,
            version="1.0",
            source="user",
            key=plugin_id,
            capabilities=[EXECUTION_ROUTING_CAPABILITY],
        )
        record_consent(plugin_id, [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
        context = PluginContext(manifest, manager)
        provider = _provider(plugin_id, "1.0", provider_id)
        subject = context.register_execution_router(provider).subject
        record_execution_router_consent(subject)
        contexts.append(context)
        providers.append(provider)

    barrier = threading.Barrier(2)
    outcomes = []

    def claim(context, provider):
        barrier.wait()
        try:
            outcomes.append(context.register_execution_router(provider).status)
        except ValueError:
            outcomes.append("conflict")

    workers = [threading.Thread(target=claim, args=pair) for pair in zip(contexts, providers)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(1)

    assert sorted(outcomes) == ["active", "conflict"]
    assert manager.get_execution_router_registration() is not None


def test_tuple_change_is_stale_and_cli_consent_displays_all_five_fields(hermes_home):
    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    ctx, _ = _context()
    subject = ctx.register_execution_router(_provider()).subject
    record_execution_router_consent(subject)
    changed, _ = _context(version="1.2.4")
    stale_subject = changed.register_execution_router(_provider(plugin_version="1.2.4")).subject
    assert execution_router_consent_status("router-plugin").state is ExecutionRouterConsentState.STALE

    from hermes_cli.plugins_cmd import _run_execution_router_consent

    console = MagicMock()
    current_entry = ("router-plugin", "1.2.4", "", "user", hermes_home, "router-plugin")
    with patch("hermes_cli.plugins_cmd._find_plugin_entry", return_value=current_entry), patch(
        "hermes_cli.plugins_cmd._declared_capabilities_for_key",
        return_value=[EXECUTION_ROUTING_CAPABILITY],
    ), patch("hermes_cli.plugins_cmd._is_tty", return_value=True), patch(
        "hermes_cli.plugins_cmd._ask_yes", return_value=True
    ):
        assert _run_execution_router_consent(console, "router-plugin") is True
    rendered = "\n".join(str(call.args[0]) for call in console.print.call_args_list)
    for value in stale_subject.as_tuple():
        assert value in rendered
    raw = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))
    assert raw["plugins"]["entries"]["router-plugin"]["execution_router"]["consent"]


def test_enable_already_enabled_still_offers_pending_exact_consent():
    from hermes_cli.plugins_cmd import cmd_enable

    with patch(
        "hermes_cli.plugins_cmd._resolve_plugin_key_and_source",
        return_value=("router-plugin", "user"),
    ), patch("hermes_cli.plugins_cmd._get_enabled_set", return_value={"router-plugin"}), patch(
        "hermes_cli.plugins_cmd._get_disabled_set", return_value=set()
    ), patch(
        "hermes_cli.plugins_cmd._declared_capabilities_for_key",
        return_value=[EXECUTION_ROUTING_CAPABILITY],
    ), patch(
        "hermes_cli.plugins_cmd._run_capability_consent", return_value=True
    ), patch(
        "hermes_cli.plugins_cmd._run_execution_router_consent", return_value=True
    ) as router_consent:
        cmd_enable("router-plugin")

    router_consent.assert_called_once_with(ANY, "router-plugin")


def test_capabilities_read_is_non_mutating_for_pending_router(hermes_home):
    from hermes_cli.plugins_cmd import cmd_capabilities

    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    ctx, _ = _context()
    ctx.register_execution_router(_provider())
    config_path = hermes_home / "config.yaml"
    before = config_path.read_bytes()
    discovered = ("router-plugin", "1.2.3", "", "user", hermes_home, "router-plugin")

    with patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=[discovered]), patch(
        "hermes_cli.plugins_cmd._declared_capabilities_for_key",
        return_value=[EXECUTION_ROUTING_CAPABILITY],
    ):
        cmd_capabilities("router-plugin")

    assert config_path.read_bytes() == before


def test_normal_load_after_exact_consent_activates_provider(hermes_home, tmp_path):
    plugin_dir = tmp_path / "router-plugin"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text(
        "from agent.execution_router import ExecutionKind, ExecutionRouteDecisionV1, "
        "ExecutionRouterProviderDescriptorV1\n"
        "class Provider:\n"
        "    descriptor = ExecutionRouterProviderDescriptorV1("
        "'router-plugin', '1.2.3', 'router-a', '1.0', tuple(ExecutionKind))\n"
        "    def resolve_execution_route(self, request, cancellation):\n"
        "        return ExecutionRouteDecisionV1.pass_through("
        "request_id=request.request_id, attempt_id=request.attempt_id)\n"
        "def register(ctx):\n"
        "    ctx.register_execution_router(Provider())\n",
        encoding="utf-8",
    )
    manifest = PluginManifest(
        name="router-plugin",
        version="1.2.3",
        source="user",
        path=str(plugin_dir),
        key="router-plugin",
        capabilities=[EXECUTION_ROUTING_CAPABILITY],
    )
    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    first = PluginManager()
    first._load_plugin(manifest)
    assert first.get_execution_router_registration() is None
    assert first._plugins["router-plugin"].error == "execution_router: consent_required"
    subject = execution_router_consent_status("router-plugin").subject
    assert subject is not None
    record_execution_router_consent(subject)

    second = PluginManager()
    second._load_plugin(manifest)

    assert second.get_execution_router_registration() is not None


def test_one_argument_provider_is_rejected_before_pending_state(hermes_home):
    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    provider = _provider()
    provider.resolve_execution_route = lambda request: None
    ctx, manager = _context()

    with pytest.raises(TypeError, match="exactly request and cancellation"):
        ctx.register_execution_router(provider)
    assert manager.get_execution_router_registration() is None
    assert execution_router_consent_status("router-plugin").state is ExecutionRouterConsentState.ABSENT


def test_legacy_router_boolean_cannot_grant_new_capability(hermes_home):
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  entries:\n    router-plugin:\n      execution_router:\n"
        "        generic_capability_granted: true\n",
        encoding="utf-8",
    )
    ctx, _ = _context()
    with pytest.raises(PermissionError, match="not granted"):
        ctx.register_execution_router(_provider())


def test_live_grant_revocation_invalidates_active_generation(hermes_home):
    record_consent("router-plugin", [EXECUTION_ROUTING_CAPABILITY], [EXECUTION_ROUTING_CAPABILITY])
    first, _ = _context()
    subject = first.register_execution_router(_provider()).subject
    record_execution_router_consent(subject)
    ctx, manager = _context()
    ctx.register_execution_router(_provider())
    registration = manager.get_execution_router_registration()
    assert registration.is_current(registration.generation) is True

    raw = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))
    raw["plugins"]["entries"]["router-plugin"]["granted_capabilities"] = []
    raw["plugins"]["entries"]["router-plugin"]["execution_router"].pop(
        "generic_capability_granted", None
    )
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert registration.is_current(registration.generation) is False
    assert manager.get_execution_router_registration() is None


def test_bundled_router_uses_both_consent_phases():
    from hermes_cli.plugins_cmd import cmd_enable

    with patch(
        "hermes_cli.plugins_cmd._resolve_plugin_key_and_source",
        return_value=("router-plugin", "bundled"),
    ), patch("hermes_cli.plugins_cmd._get_enabled_set", return_value={"router-plugin"}), patch(
        "hermes_cli.plugins_cmd._get_disabled_set", return_value=set()
    ), patch(
        "hermes_cli.plugins_cmd._declared_capabilities_for_key",
        return_value=[EXECUTION_ROUTING_CAPABILITY],
    ), patch("hermes_cli.plugins_cmd._run_capability_consent", return_value=True) as generic, patch(
        "hermes_cli.plugins_cmd._run_execution_router_consent", return_value=True
    ) as exact:
        cmd_enable("router-plugin")

    generic.assert_called_once()
    exact.assert_called_once()
