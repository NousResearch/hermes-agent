"""Frozen provider loaded through real discovery and a subprocess plugin host."""

import os
import sys
from dataclasses import dataclass

import pytest

from providers import base


PLUGIN = '''
import inspect
import os
from providers import register_provider
from providers import base


def pair(api_key, base_url):
    credential_type = getattr(base, "PoolRuntimeCredential", None)
    if credential_type is None:
        return {"api_key": api_key, "base_url": base_url}
    return credential_type(api_key, base_url)


class PoolHost(base.ProviderProfile):
    def build_extra_body(self, **context):
        return {"pid": os.getpid()}

    def prepare_pool_entry(self, entry, api_key):
        if entry.id == "legacy":
            return api_key
        if entry.id == "mapping":
            return {"api_key": "untyped", "base_url": None}
        return pair(f"prepared-{entry.id}-{api_key}", f"https://{entry.id}.example/v1")

    def refresh_pool_runtime_credentials(self, entry, *, reason):
        if entry.id == "decline":
            return None
        return pair(f"refreshed-{entry.id}-{reason}", f"https://fresh-{entry.id}.example/v1")


options = {}
if "credential_pool_base_urls" in inspect.signature(base.ProviderProfile).parameters:
    options["credential_pool_base_urls"] = ("https://gateway.example/v1", "https://account.example/v1")
register_provider(PoolHost(name="poolcontract", base_url="https://gateway.example/v1", **options))
'''


@dataclass
class SelectedRow:
    id: str
    access_token: str = "persisted-oauth-secret"


@pytest.fixture
def hosted_profile(tmp_path, monkeypatch):
    import providers
    from hermes_cli import plugins

    home = tmp_path / "hermes"
    plugin_dir = home / "plugins" / "model-providers" / "poolcontract"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: poolcontract\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  isolation: host\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    try:
        profile = providers.get_provider_profile("poolcontract")
        assert profile is not None
        assert type(profile).__module__ == "hermes_cli.plugin_host_profiles"
        assert profile.build_extra_body()["pid"] != os.getpid()
        assert not any(name.endswith("_poolcontract") for name in sys.modules)
        yield profile
    finally:
        host = getattr(plugins.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()


@pytest.mark.platforms("any")
def test_hosted_prepare_preserves_runtime_pair_type_and_selected_row(hosted_profile):
    row = SelectedRow("account-a")
    result = hosted_profile.prepare_pool_entry(row, "selected-key")
    credential_type = getattr(base, "PoolRuntimeCredential", None)
    assert credential_type is not None, "provider SDK must expose runtime credentials"
    assert isinstance(result, credential_type), "host must restore the declared union's credential arm"
    assert result.api_key == "prepared-account-a-selected-key"
    assert result.base_url == "https://account-a.example/v1"
    assert "selected-key" not in repr(result)
    assert row == SelectedRow("account-a")


@pytest.mark.platforms("any")
def test_hosted_pool_endpoint_hints_keep_tuple_contract(hosted_profile):
    assert getattr(hosted_profile, "credential_pool_base_urls", ()) == (
        "https://gateway.example/v1", "https://account.example/v1")


@pytest.mark.platforms("any")
def test_hosted_refresh_is_row_bound_and_can_decline(hosted_profile):
    result = hosted_profile.refresh_pool_runtime_credentials(SelectedRow("account-b"), reason="unauthorized")
    credential_type = getattr(base, "PoolRuntimeCredential", None)
    assert credential_type is not None, "provider SDK must expose runtime credentials"
    assert isinstance(result, credential_type), "host must restore optional runtime credentials"
    assert result.api_key == "refreshed-account-b-unauthorized"
    assert result.base_url == "https://fresh-account-b.example/v1"
    assert hosted_profile.refresh_pool_runtime_credentials(SelectedRow("decline"), reason="stale") is None


@pytest.mark.platforms("any")
def test_hosted_legacy_strings_and_untyped_mappings_are_not_coerced(hosted_profile):
    assert hosted_profile.prepare_pool_entry(SelectedRow("legacy"), "legacy-key") == "legacy-key"
    result = hosted_profile.prepare_pool_entry(SelectedRow("mapping"), "ignored")
    assert type(result) is dict
    assert result == {"api_key": "untyped", "base_url": None}
