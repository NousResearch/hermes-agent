"""The secret store port as a plugin kind, and the local backend's shape.

``resolve_secret_store(name)`` picks a backend by explicit name (no caller is wired to it yet):
``"local"`` is the local profile files; another name loads a provider through real discovery
(``$HERMES_HOME/plugins/<name>/``), and a missing one is an error, never a fallback to the local
files. The local backend serves only the local tenant and its own profile, holds the agent level
only, and says so.
"""
import textwrap

import pytest

from hermes_secret_port import SecretStoreProvider, resolve_secret_store
from hermes_secret_port_local import LocalSecretStoreProvider

_PLUGIN = '''
from hermes_secret_port import SecretStoreProvider


class Probe(SecretStoreProvider):
    @property
    def name(self):
        return "probe"

    def open(self, home=None):
        raise NotImplementedError

    def assert_isolating(self):
        raise RuntimeError("probe role bypasses row security")


def register(ctx):
    ctx.register_secret_store(Probe())
'''


def test_local_names_the_local_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("local", ""):
        provider = resolve_secret_store(name)
        assert isinstance(provider, LocalSecretStoreProvider) and provider.levels == frozenset({"agent"})
        provider.assert_isolating()


def test_a_provider_that_does_not_check_isolation_cannot_be_built():
    class Unchecked(SecretStoreProvider):
        @property
        def name(self):
            return "unchecked"

        def open(self, home=None):
            raise NotImplementedError

    with pytest.raises(TypeError, match="assert_isolating"):
        Unchecked()


def test_a_named_provider_loads_through_discovery_and_runs_its_isolation_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "probe"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN))
    provider = resolve_secret_store("probe")
    assert isinstance(provider, SecretStoreProvider) and provider.name == "probe"
    with pytest.raises(RuntimeError, match="bypasses row security"):
        provider.assert_isolating()


def test_an_unavailable_named_provider_is_an_error_not_the_local_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "offline"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN).replace(
        "    def assert_isolating(self):", "    def is_available(self):\n        return False\n\n    def assert_isolating(self):"))
    with pytest.raises(LookupError, match="not available"):
        resolve_secret_store("offline")


def test_a_missing_named_provider_is_an_error_not_the_local_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(LookupError, match="encrypted"):
        resolve_secret_store("encrypted")
