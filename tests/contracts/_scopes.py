"""Scopes for the contract suites: one profile home per label under the suites' fake Hermes root."""

from __future__ import annotations

from pathlib import Path

from persistence import ProfileScope, StorePin

# The fake root of the suites' profile tree; ``<root>/profiles/<label>`` is a named profile's home.
CONTRACT_ROOT = Path("/contract-homes")
# The retained media root the suites pass to the verbs that retire sessions.
MEDIA_ROOT = "/contract-media/native-inputs"


def profile_scope(profile: str) -> ProfileScope:
    """The scope of profile ``profile``: its home under the fake root's ``profiles`` folder, labelled with its name."""
    return ProfileScope(profile_dir=CONTRACT_ROOT / "profiles" / profile, name=profile)


def pinned_scope(tenant: str, agent: str, *, profile_dir: Path = CONTRACT_ROOT) -> ProfileScope:
    """A hosted scope pinned to ``(tenant, agent)``. ``profile_dir`` is only the process home the scope is built in; a
    pinned scope's store is chosen by its pin alone, so the suites pass a served local profile's folder to prove a pin
    never falls through to that profile's store."""
    return ProfileScope(profile_dir=profile_dir, name="hosted", pin=StorePin(tenant, agent))
