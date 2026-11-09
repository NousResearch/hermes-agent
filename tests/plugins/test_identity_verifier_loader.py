"""Loading an identity verifier provider, by explicit name.

A provider is loaded in-process only: under ``plugins.isolation: host`` it is refused with the
reason, never handed back as an opaque proxy whose checks would run outside this process. A
user provider's module belongs to the profile that installed it: two profiles that each install
a provider of the same name each get their own code. A name that is missing or unavailable is
an error — nothing falls back — and the bundled ``turn_claim_v1`` reference provider resolves
through the same discovery as any user plugin.
"""
import textwrap

import pytest

from agent.runtime_scope import bind_profile_scope
from hermes_identity_verifier import resolve_identity_verifier

_PROVIDER = '''
from pathlib import Path

from hermes_identity_verifier import IdentityVerifier, VerifiedTurn
from hermes_scope import Principal

from ._tag import TAG

Path(__file__).with_name("imported").write_text(TAG)


class Probe(IdentityVerifier):
    @property
    def name(self):
        return TAG

    async def verify(self, frame):
        return VerifiedTurn(tenant=TAG, agent=TAG, actor=Principal(TAG),
                            owner=Principal(TAG), can_administer=False)


def register(ctx):
    ctx.register_identity_verifier(Probe())
'''

_UNAVAILABLE = _PROVIDER.replace(
    "    async def verify(self, frame):",
    "    def is_available(self):\n"
    "        return False\n\n"
    "    async def verify(self, frame):")


def _install(home, tag, *, name="cacheprobe", isolation="in_process", body=_PROVIDER):
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(body))
    (plugin / "_tag.py").write_text(f"TAG = {tag!r}\n")  # a sibling module: namespaced with its plugin
    (home / "config.yaml").write_text(f"plugins:\n  isolation: {isolation}\n")
    return plugin


def test_same_named_providers_in_two_profiles_each_load_their_own_code(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _install(a, "A")
    _install(b, "B")
    with bind_profile_scope(a, hydrate_secrets=False):
        first = resolve_identity_verifier("cacheprobe").name
    with bind_profile_scope(b, hydrate_secrets=False):
        second = resolve_identity_verifier("cacheprobe").name
    with bind_profile_scope(a, hydrate_secrets=False):
        again = resolve_identity_verifier("cacheprobe").name
    assert (first, second, again) == ("A", "B", "A")


def test_under_host_isolation_a_user_provider_is_refused_with_the_reason(tmp_path):
    home = tmp_path / "hosted"
    plugin = _install(home, "HOST", name="hostprobe", isolation="host")
    with bind_profile_scope(home, hydrate_secrets=False):
        with pytest.raises(LookupError, match="plugins.isolation is 'host'"):
            resolve_identity_verifier("hostprobe")
    assert not (plugin / "imported").exists()  # its code never ran, in this process or a host


def test_a_missing_provider_is_an_error_never_a_fallback(tmp_path):
    with bind_profile_scope(tmp_path, hydrate_secrets=False):
        with pytest.raises(LookupError, match="no such identity verifier"):
            resolve_identity_verifier("not-installed")


def test_an_empty_name_is_an_error_there_is_no_local_verifier(tmp_path):
    with bind_profile_scope(tmp_path, hydrate_secrets=False):
        with pytest.raises(LookupError, match="no identity verifier is configured"):
            resolve_identity_verifier("")


def test_an_unavailable_provider_is_an_error(tmp_path):
    home = tmp_path / "unavailable"
    _install(home, "U", name="unavailable", body=_UNAVAILABLE)
    with bind_profile_scope(home, hydrate_secrets=False):
        with pytest.raises(LookupError, match="not available here"):
            resolve_identity_verifier("unavailable")


def _pinned_issuer(monkeypatch, issuer="https://connector.example"):
    """The deployment's recorded boot config: the pinned trust root the reference verifier
    reads its issuer from."""
    from hermes_cli import hosted_mode

    monkeypatch.setattr(hosted_mode, "_boot_or_effective_config",
                        lambda: {"hosted": {"turn_claim_issuer": issuer}})


def test_the_bundled_reference_provider_resolves_by_name(tmp_path, monkeypatch):
    """``turn_claim_v1`` resolves through the real bundled discovery, and its availability needs
    the deployment's relay id and its pinned issuer (without either no claim could be checked,
    so none admitted)."""
    monkeypatch.setenv("GATEWAY_RELAY_ID", "inst_probe")
    _pinned_issuer(monkeypatch)
    with bind_profile_scope(tmp_path, hydrate_secrets=False):
        verifier = resolve_identity_verifier("turn_claim_v1")
    assert verifier.name == "turn_claim_v1"
    assert verifier.is_available() is True
    monkeypatch.setenv("GATEWAY_RELAY_ID", "")
    assert verifier.is_available() is False  # the same process serves no relay id
    monkeypatch.setenv("GATEWAY_RELAY_ID", "inst_probe")
    _pinned_issuer(monkeypatch, issuer="")
    assert verifier.is_available() is False  # and no pinned issuer


def test_a_user_provider_never_shadows_the_bundled_one(tmp_path, monkeypatch):
    """Bundled wins on collision, like every explicit-name provider kind: a dropped-in dir must
    not stand in for the reference verifier."""
    monkeypatch.setenv("GATEWAY_RELAY_ID", "inst_probe")
    _pinned_issuer(monkeypatch)
    home = tmp_path / "shadow"
    _install(home, "SHADOW", name="turn_claim_v1")
    with bind_profile_scope(home, hydrate_secrets=False):
        assert resolve_identity_verifier("turn_claim_v1").name == "turn_claim_v1"
        assert type(resolve_identity_verifier("turn_claim_v1")).__module__ == \
            "plugins.identity_verifier.turn_claim_v1"
