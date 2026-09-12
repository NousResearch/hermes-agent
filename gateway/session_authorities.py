"""One SessionAuthority per reserved profile home; resolved by the active runtime scope.

A multiplexing gateway owns every profile under ``profiles/`` plus the default home. Each
home has its own ``state.db``, admission ledger, runtime epoch and control identity, so each
gets its own authority. Every consumer already runs inside the routed profile's
``_profile_runtime_scope`` (that is what the multiplex fixes on main established), so the
authority for "the profile this code is acting for" is the one keyed by the active home.

A single-profile gateway is a map of one: the launch home's authority answers every lookup,
scoped or not, exactly as before.
"""
from __future__ import annotations

from pathlib import Path

from hermes_constants import get_hermes_home, get_hermes_home_override, hermes_home_key
from hermes_state_runtime import RuntimeStoreError


class SessionAuthorities:
    """Registry of authorities keyed by ``hermes_home_key(home)``."""

    def __init__(self, launch_home):
        self.launch_key = hermes_home_key(launch_home)
        self._by_key: dict[str, object] = {}
        self._names: dict[str, str | None] = {}

    def add(self, home, authority, name=None) -> None:
        key = hermes_home_key(home)
        if key in self._by_key:
            raise RuntimeError(f'duplicate session authority for {home}')
        self._by_key[key] = authority
        self._names[key] = name

    def replace(self, home, authority) -> None:
        """Fill the slot reserved by ``add(home, None, name=...)`` once the authority exists."""
        key = hermes_home_key(home)
        if key not in self._by_key or self._by_key[key] is not None:
            raise RuntimeError(f'no reserved session authority slot for {home}')
        self._by_key[key] = authority

    def profile_name(self, authority):
        """Served profile name for *authority*; None for the launch profile (its adapters are
        ``runner.adapters``, its session keys the historical ``agent:main`` namespace)."""
        key = hermes_home_key(authority.profile_id)
        return None if key == self.launch_key else self._names.get(key)

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self):
        return iter(self._by_key.values())

    def __contains__(self, home) -> bool:
        return hermes_home_key(home) in self._by_key

    @property
    def launch(self):
        return self._by_key[self.launch_key]

    def for_home(self, home):
        """Authority owning *home*, or None when this process does not serve it."""
        return self._by_key.get(hermes_home_key(home))

    def require(self, home):
        authority = self.for_home(home)
        if authority is None:
            raise RuntimeStoreError('profile_mismatch')
        return authority

    def active(self):
        """Authority for the active runtime scope.

        Unscoped code (startup, shutdown, background watchers) belongs to the launch profile.
        Scoped code that names a home this process does not serve gets ``None``: a routed
        profile must never fall back to the launch profile's ledger.
        """
        if len(self._by_key) == 1 or get_hermes_home_override() is None:
            return self._by_key[self.launch_key]
        return self._by_key.get(hermes_home_key(get_hermes_home()))

    def served_profiles(self) -> list[dict]:
        return [{'profile_id': a.profile_id, 'home': a.profile_id} for a in self._by_key.values()]

    def profile_ids(self) -> frozenset[str]:
        return frozenset(a.profile_id for a in self._by_key.values())


def served_profile_name(home) -> str:
    """Canonical profile id of a served home: ``default`` for the root, the directory name for
    ``profiles/<name>`` (via ``hermes_constants.profile_name_for_home``, the resolver main uses).
    Any other custom HERMES_HOME is the process's own single profile and keeps the ``default``
    label the CLI reports for it (``hermes -p`` never targets such a home by name)."""
    from hermes_constants import profile_name_for_home
    return profile_name_for_home(home) or 'default'


def authority_for_home(runner, home):
    """Explicit per-home lookup for callers that already know the routed home."""
    registry = getattr(runner, 'session_authorities', None)
    if registry is None:
        authority = getattr(runner, 'session_authority', None)
        if authority is not None and Path(authority.db.db_path).resolve().parent == Path(home).resolve():
            return authority
        return None
    return registry.for_home(home)


def authority_for_profile_id(runner, profile_id):
    return authority_for_home(runner, Path(profile_id))


def active_authority(runner):
    """Authority for the active runtime scope (see ``SessionAuthorities.active``); the single
    launch authority when the runner predates the registry (bare test runners)."""
    registry = getattr(runner, 'session_authorities', None)
    if registry is None:
        return getattr(runner, 'session_authority', None)
    return registry.active()


def all_authorities(runner):
    registry = getattr(runner, 'session_authorities', None)
    if registry is not None:
        return list(registry)
    authority = getattr(runner, 'session_authority', None)
    return [authority] if authority is not None else []


def owner_scope(authority, *, hydrate_secrets=False):
    """Runtime scope of the profile that owns *authority* — under multiplex only.

    Owner-side handlers (RPC dispatch, cron submit, admitted execution) must read config, jobs
    and policy for the OWNING profile, never the launch profile's ambient scope. A
    single-profile gateway keeps its ambient scope byte-for-byte: entering a scope there
    would read the profile's config/.env for every handler, including bypass (--safe-mode)
    launches that must never touch the profile YAML. Test peers with symbolic ids ('fixture')
    likewise keep the ambient scope.
    """
    from contextlib import nullcontext
    runner = getattr(authority, 'runner', None)
    if not getattr(getattr(runner, 'config', None), 'multiplex_profiles', False):
        return nullcontext()
    home = Path(str(authority.profile_id))
    if not home.is_absolute() or not home.is_dir():
        return nullcontext()
    from gateway.run import _profile_runtime_scope
    return _profile_runtime_scope(home, hydrate_secrets=hydrate_secrets)
