"""Loading an extension registry provider from a profile's ``$HERMES_HOME/plugins/<name>/``.

A provider is loaded in-process only: under ``plugins.isolation: host`` it is refused with the
reason, never handed back as an opaque proxy whose nested port cannot be called and whose calls
would not carry the caller's scope. A user provider's module belongs to the profile that
installed it: two profiles that each install a provider of the same name each get their own code.
"""
import textwrap

import pytest

from agent.runtime_scope import bind_profile_scope
from hermes_extension_registry_port import resolve_extension_registry

_PROVIDER = '''
from pathlib import Path

from hermes_extension_registry_port import ExtensionRegistryProvider

from ._tag import TAG

Path(__file__).with_name("imported").write_text(TAG)


class Probe(ExtensionRegistryProvider):
    @property
    def name(self):
        return TAG

    def open(self, home=None):
        raise NotImplementedError

    def assert_isolating(self):
        return


def register(ctx):
    ctx.register_extension_registry(Probe())
'''


def _install(home, tag, *, name="cacheprobe", isolation="in_process"):
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PROVIDER))
    (plugin / "_tag.py").write_text(f"TAG = {tag!r}\n")  # a sibling module: namespaced with its plugin
    (home / "config.yaml").write_text(f"plugins:\n  isolation: {isolation}\n")
    return plugin


def test_same_named_providers_in_two_profiles_each_load_their_own_code(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _install(a, "A")
    _install(b, "B")
    with bind_profile_scope(a, hydrate_secrets=False):
        first = resolve_extension_registry("cacheprobe").name
    with bind_profile_scope(b, hydrate_secrets=False):
        second = resolve_extension_registry("cacheprobe").name
    with bind_profile_scope(a, hydrate_secrets=False):
        again = resolve_extension_registry("cacheprobe").name
    assert (first, second, again) == ("A", "B", "A")


def test_under_host_isolation_a_user_provider_is_refused_with_the_reason(tmp_path):
    home = tmp_path / "hosted"
    plugin = _install(home, "HOST", name="hostprobe", isolation="host")
    with bind_profile_scope(home, hydrate_secrets=False):
        with pytest.raises(LookupError, match="plugins.isolation is 'host'"):
            resolve_extension_registry("hostprobe")
    assert not (plugin / "imported").exists()  # its code never ran, in this process or a host
