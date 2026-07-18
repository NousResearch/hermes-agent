from types import SimpleNamespace

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform
from gateway.session import SessionSource


class _Runner(GatewayAuthorizationMixin):
    config: object
    adapters: dict
    _profile_adapters: dict


def _source(profile: str) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="user-1",
        chat_id="chat-1",
        profile=profile,
    )


def test_named_standalone_profile_resolves_its_primary_adapter():
    adapter = object()
    runner = _Runner()
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {}

    assert runner._adapter_for_source(_source("project-alpha")) is adapter


def test_multiplex_secondary_profile_does_not_fall_back_to_default_adapter():
    default_adapter = object()
    runner = _Runner()
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner.adapters = {Platform.DISCORD: default_adapter}
    runner._profile_adapters = {"project-alpha": {}}

    assert runner._adapter_for_source(_source("project-alpha")) is None


def test_multiplex_missing_secondary_registry_does_not_fall_back():
    default_adapter = object()
    runner = _Runner()
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner.adapters = {Platform.DISCORD: default_adapter}
    runner._profile_adapters = {}

    assert runner._adapter_for_source(_source("project-alpha")) is None


def test_multiplex_default_profile_resolves_primary_adapter():
    adapter = object()
    runner = _Runner()
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {"project-alpha": {Platform.DISCORD: object()}}

    assert runner._adapter_for_source(_source("default")) is adapter
