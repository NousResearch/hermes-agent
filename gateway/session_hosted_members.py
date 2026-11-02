"""Which local profiles a hosted group on one authority may seat (#139002).

A group seats the profiles this process serves, plus the ``hosted_rooms.profiles`` destinations,
and nothing else: a profile directory alone seats nobody, because a member's turns need a live
owner. Served siblings need no entry, but only a host-scoped caller seats them that way; a
profile-bound caller seats its own profile plus the profiles its entries name, as before. One home
answers to one name. An entry that would break a rule is ignored and logged once, and only a new
member naming it is refused, so one bad entry never takes the rest of the roster down.
"""
import logging
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

# (profile, field, value, detail) already logged: the roster is re-read on every poll and turn,
# and an entry fixed and later broken another way is logged again.
_REPORTED: set[tuple[str, str, str, str]] = set()


def member_roster(authority, *, siblings=True):
    """``(homes, refused)`` for a group on *authority*.

    ``homes`` maps name -> home. It holds *authority*'s own profile, each served sibling when
    ``siblings`` (the room driver, and a host-scoped caller), and each usable entry. A served
    profile keeps its served home under ``siblings``, whatever its entry says; a profile-bound
    caller's entry naming a served profile must match its served home. ``refused`` maps a name a
    new member may not take to ``(reason, field, detail)``. An alias stays in ``homes`` too, so a
    room that already seats it keeps running its other members (its own turns fail, as before).
    """
    from hermes_constants import hermes_home_key
    own, served, refused = _served(authority)
    blocked = set(refused)
    homes = dict(served)
    if not siblings:
        for name in [name for name in homes if name != own]:
            del homes[name]
            refused[name] = ('permission_denied', 'members',
                             'only a host-scoped connection seats a served profile without an entry')
    seated = {hermes_home_key(home): name for name, home in served.items()}
    for name, value in _configured(authority, own).items():
        if name in blocked:
            continue
        field = f'hosted_rooms.profiles.{name}'
        home, reason, detail = _config_entry(name, value, served, seated)
        if siblings and name in served:
            # Served: it needs no entry, and the served home wins over whatever this one says.
            if reason is not None:
                _report(own, field, value, detail)
        elif reason is None:
            homes[name] = home
            seated.setdefault(hermes_home_key(home), name)
            refused.pop(name, None)
        else:
            _report(own, field, value, detail)
            refused[name] = (reason, field, detail)
            if home is None:
                homes.pop(name, None)
            else:
                homes[name] = home
    return homes, refused


def refuse_unseatable(members, refused):
    """Raise the named refusal for the first local member whose profile *refused* names;
    ``validate_roster`` names every other roster problem."""
    for index, raw in enumerate(members if isinstance(members, list) else ()):
        target = raw.get('target') if isinstance(raw, dict) else None
        profile = raw.get('profile') if isinstance(raw, dict) else None
        if isinstance(target, dict) and target.get('kind', 'local') != 'local':
            continue
        if isinstance(profile, str) and profile in refused:
            reason, field, detail = refused[profile]
            raise RuntimeStoreError(reason, field=field, detail=f"member {index} profile '{profile}': {detail}")


def _served(authority):
    """``(own name, name -> home, refused)`` over the local bots this process serves
    (``session_authorities.local_bots``). A bot its clients cannot address by name is refused,
    never bound; a retiring one is not seated in a new group."""
    from gateway.session_authorities import local_bots, served_profile_name
    bots = local_bots(getattr(authority, 'runner', None))
    own = next((bot.name for bot in bots if bot.authority is authority), None)
    if own is None:
        # A bare authority (no registry): it still owns its own rooms.
        own = served_profile_name(authority.profile_id)
    served, refused = {}, {}
    for bot in bots:
        if bot.problem is not None:
            refused[bot.name] = ('invalid_params', 'members', bot.problem.replace(' yet', ' in groups yet'))
        elif bot.retiring and bot.authority is not authority:
            continue
        else:
            served[bot.name] = bot.home
    if own not in refused:
        served.setdefault(own, Path(authority.profile_id))
    return own, served, refused


def _configured(authority, own):
    """The ``hosted_rooms.profiles`` mapping; a section left with no entries is empty."""
    from gateway.run import _load_gateway_config
    from gateway.session_authorities import owner_scope
    # The coordinator's own threads do not inherit the caller's ContextVars.
    with owner_scope(authority):
        section = _load_gateway_config().get('hosted_rooms')
    profiles = section.get('profiles') if isinstance(section, dict) else section
    if profiles is None or isinstance(profiles, dict):
        return profiles or {}
    _report(own, 'hosted_rooms.profiles', profiles, 'must map profile names to profile homes')
    return {}


def _config_entry(name, value, served, seated):
    """``(home, reason, detail)`` for one entry: ``reason`` None seats ``home``; otherwise the entry
    is refused, and a ``home`` it still carries is an alias that existing rooms keep."""
    from gateway.hosted_rooms_common import IDENTIFIER_RE
    from hermes_constants import hermes_home_key
    if not isinstance(name, str) or not IDENTIFIER_RE.fullmatch(name) or not isinstance(value, str):
        return None, 'invalid_params', 'must map a profile name to a home'
    target = Path(value)
    if not target.is_absolute() or target != target.resolve():
        return None, 'invalid_params', 'must be an absolute, resolved home'
    key = hermes_home_key(target)
    if name in served and hermes_home_key(served[name]) != key:
        return None, 'permission_denied', 'this gateway serves that profile from another home'
    if seated.get(key, name) != name:
        # Its turns reach that profile under the wrong name and fail every time.
        return target, 'invalid_params', f"names the home already seated as '{seated[key]}'"
    return served.get(name, target), None, None


def _report(own, field, value, detail):
    """Log one ignored entry once per value, however often the roster is read."""
    key = (own, field, repr(value)[:200], detail)
    if key not in _REPORTED:
        _REPORTED.add(key)
        logger.warning('Ignoring %s for profile=%s: %s', field, own, detail)
