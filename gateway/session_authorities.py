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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home, get_hermes_home_override, hermes_home_key
from hermes_state_runtime import RuntimeStoreError


class SessionAuthorities:
    """Registry of authorities keyed by ``hermes_home_key(home)``."""

    def __init__(self, launch_home, *, multiplexed=False):
        self.launch_key = hermes_home_key(launch_home)
        self._by_key: dict[str, object] = {}
        self._names: dict[str, str | None] = {}
        # A multiplexing runtime (configured, or one that ever served a second home) keeps the
        # scoped-lookup contract after its secondaries are removed: an unserved home is None.
        self.multiplexed = bool(multiplexed)

    def add(self, home, authority, name=None) -> None:
        key = hermes_home_key(home)
        if key in self._by_key:
            raise RuntimeError(f'duplicate session authority for {home}')
        self._by_key[key] = authority
        self._names[key] = name
        if key != self.launch_key:
            self.multiplexed = True

    def replace(self, home, authority) -> None:
        """Fill the slot reserved by ``add(home, None, name=...)`` once the authority exists."""
        key = hermes_home_key(home)
        if key not in self._by_key or self._by_key[key] is not None:
            raise RuntimeError(f'no reserved session authority slot for {home}')
        self._by_key[key] = authority

    def remove(self, home):
        """Drop *home*'s slot (a parked boot, a hot-unserved profile); returns the authority or None.
        The launch slot is never removed."""
        key = hermes_home_key(home)
        if key == self.launch_key:
            raise RuntimeError('the launch session authority cannot be removed')
        self._names.pop(key, None)
        return self._by_key.pop(key, None)

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
        profile must never fall back to the launch profile's ledger — including once a removed
        secondary leaves only the launch slot (a late callback of that profile). Only a runtime
        that never multiplexed is a map of one that answers every scope with the launch authority.
        """
        if not self.multiplexed or get_hermes_home_override() is None:
            return self._by_key[self.launch_key]
        return self._by_key.get(hermes_home_key(get_hermes_home()))

    def serving(self) -> list:
        """Authorities that accept work: not a reserved-but-unbuilt slot, not one retiring (a
        failed serve or an unserve whose turn outlived Stop keeps its slot but refuses admissions)."""
        # A snapshot: hosted-room coordinator threads read the served set while the loop hot-serves.
        return [a for a in list(self._by_key.values()) if a is not None and not getattr(a, 'retiring', False)]

    def served_profiles(self) -> list[dict]:
        return [{'profile_id': a.profile_id, 'home': a.profile_id} for a in self.serving()]

    def profile_ids(self) -> frozenset[str]:
        return frozenset(a.profile_id for a in self.serving())


@dataclass(frozen=True)
class LocalBot:
    """One local bot this process serves: its profile ``name``, ``home`` and owning ``authority``.
    ``problem`` says why a client cannot address it by that name (None when it can); ``retiring``
    marks one being unserved, which still owns its rooms but takes no new work."""
    name: str
    home: Path
    authority: Any
    problem: str | None = None
    retiring: bool = False


def local_bots(runner) -> list[LocalBot]:
    """The local bots this gateway serves, one per built authority, by the name the registry
    reserved each under (the launch profile keeps ``served_profile_name``). The one answer to
    "which bots are local here" for both the Bots roster (``profiles.list``) and group seating
    (``session_hosted_members``). A name two homes answer to, or one that is not its home's own
    name (a linked ``profiles/<name>`` resolves outside ``profiles/`` and reads as ``default``), is
    flagged rather than bound: a client routes, and the member transport names, a bot by name."""
    from collections import Counter
    registry = getattr(runner, 'session_authorities', None)
    if registry is None:
        authority = getattr(runner, 'session_authority', None)
        pairs = [(served_profile_name(authority.profile_id), authority)] if authority is not None else []
    else:
        # A snapshot: hosted-room coordinator threads read the served set while the loop hot-serves.
        pairs = [(registry.profile_name(a) or served_profile_name(a.profile_id), a)
                 for a in list(registry._by_key.values()) if a is not None]
    counts = Counter(name for name, _ in pairs)
    bots = []
    for name, authority in pairs:
        problem = None
        if counts[name] > 1:
            problem = 'two served homes answer to that name'
        elif name != served_profile_name(authority.profile_id):
            problem = 'a linked profile home is not supported yet'
        bots.append(LocalBot(name, Path(authority.profile_id), authority, problem,
                             bool(getattr(authority, 'retiring', False))))
    return bots


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


def require_active_authority(runner, store=None):
    """The active scope's authority, or a refusal: retryable (``runtime_draining``) while its
    profile is parked, else ``profile_mismatch``. With *store* (a callable returning the caller's
    SessionDB, called only once an authority exists), that store must be the authority's: an API
    adapter bound to another profile's SessionDB never admits here."""
    authority = active_authority(runner)
    if authority is None:
        from gateway.run_runtime import refuse_parked_scope
        refuse_parked_scope(runner)  # retryable while its profile is parked
    if authority is None or (store is not None and store() is not authority.db):
        from hermes_state_runtime import RuntimeStoreError
        raise RuntimeStoreError('profile_mismatch')
    return authority


def all_authorities(runner):
    registry = getattr(runner, 'session_authorities', None)
    if registry is not None:
        return list(registry)
    authority = getattr(runner, 'session_authority', None)
    return [authority] if authority is not None else []


def serving_authorities(runner):
    """The authorities that accept work (``SessionAuthorities.serving``: no reserved-but-unbuilt slot,
    none retiring): what startup recovery, service creation and cross-ledger reads cover. Shutdown
    keeps ``all_authorities``, so a retained authority's live writer is still settled."""
    registry = getattr(runner, 'session_authorities', None)
    if registry is not None:
        return registry.serving()
    authority = getattr(runner, 'session_authority', None)
    return [authority] if authority is not None and not getattr(authority, 'retiring', False) else []


def require_serving(authority):
    """Refuse a verb reached through a stale connection to a retired authority: its ownership is
    released, and until a successor bumps the epoch its own epoch still passes the store fence."""
    if getattr(authority, 'retired', False):
        raise RuntimeStoreError('runtime_retired')


def admission_owner(authority, session_id):
    """The admission identity a physical store row answers to: its compression root, then the
    local creation id a canonical reset forks under (a reset child is not a compression child, so
    ``logical_owner`` alone stops at the child and misses the conversation's admissions)."""
    from hermes_state_local import local_lineage_owner
    sid = authority.logical_owner(session_id)
    if not sid:
        return sid
    with authority.db._read_ctx() as conn:
        return local_lineage_owner(conn, sid)


def holds_admission(authority, session_id) -> bool:
    """Whether *authority*'s store holds a canonical admission for *session_id*. A parked profile's
    retained (retiring) store that cannot answer counts as holding one: startup auto-resume fails
    closed for that entry instead of stopping the boot or resuming a turn that FIFO may own."""
    try:
        return bool(authority.db._read_one(
            'SELECT 1 FROM session_admissions WHERE target_session_id IN (?,?) LIMIT 1',
            (session_id, admission_owner(authority, session_id))))
    except Exception as exc:
        if not getattr(authority, 'retiring', False):
            raise
        import logging
        logging.getLogger('gateway.run').warning("Skipping auto-resume of %s: retained store %s cannot answer: %s",
                                                 session_id, getattr(authority, 'profile_id', '?'), exc)
        return True


def owner_scope(authority, *, hydrate_secrets=False):
    """Runtime scope of the profile that owns *authority* — under multiplex only.

    Owner-side handlers (RPC dispatch, cron submit, admitted execution) must read config, jobs
    and policy for the OWNING profile, never the launch profile's ambient scope. A
    single-profile gateway keeps its ambient scope byte-for-byte: entering a scope there
    would read the profile's config/.env for every handler, including bypass (--safe-mode)
    launches that must never touch the profile YAML. Test peers with symbolic ids ('fixture')
    likewise keep the ambient scope.

    A standalone gateway that already hosts a second home (a profile created over the wire froze
    the launch env and made unscoped secret reads fail closed) binds the launch profile's own scope
    instead: unbound, every launch-profile session.create raised UnscopedSecretError (#141623).
    """
    from contextlib import nullcontext
    runner = getattr(authority, 'runner', None)
    if not getattr(getattr(runner, 'config', None), 'multiplex_profiles', False):
        from tui_gateway.launch_profile_policy import launch_profile_scope_if_multiplexed
        return launch_profile_scope_if_multiplexed()
    home = Path(str(authority.profile_id))
    if not home.is_absolute() or not home.is_dir():
        return nullcontext()
    from gateway.run import _profile_runtime_scope
    return _profile_runtime_scope(home, hydrate_secrets=hydrate_secrets)
