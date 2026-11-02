"""Equivalent home paths address one provider slot and generation."""
import logging

from agent.provider_registry import ProviderRegistry
from hermes_constants import hermes_home_key


class Provider:
    name = "sample"


def test_scoped_reads_and_restore_share_canonical_key(tmp_path, monkeypatch):
    registry = ProviderRegistry(label="sample", provider_cls=Provider, logger=logging.getLogger(__name__))
    home = tmp_path / "Profile"
    home.mkdir()
    raw = str(home / ".." / home.name)
    key = hermes_home_key(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    global_provider, scoped, replacement = Provider(), Provider(), Provider()
    registry.register(global_provider)
    registry.register(scoped, scope=raw)
    assert registry.get_provider("sample") is scoped
    assert registry.get_provider("sample", scope=key) is scoped
    assert registry.snapshot_registration("sample") is global_provider
    assert registry.snapshot_registration("sample", scope=key) is scoped
    generation = registry.registry_generation(scope=key)
    registry.register(replacement, scope=key)
    assert registry.registry_generation(scope=raw) != generation
    assert registry.restore_registration("sample", replacement, None, scope=raw)
    assert registry.get_provider("sample", scope=key) is global_provider
