"""Profile-owned room controls over the canonical hosted service."""
import asyncio
import logging
import os
from pathlib import Path
import uuid
import weakref

from gateway.session_persistence import authority_persistence, owner_rule
from persistence.core.errors import AuthorityRefusal, StoreUnavailable

logger = logging.getLogger(__name__)


GROUP_METHODS = {
    'groups.capabilities': 'session:read',
    'groups.list': 'session:read',
    'groups.state': 'session:read',
    'groups.log': 'session:read',
    'groups.create': 'session:control',
    'groups.rename': 'session:control',
    'groups.context': 'session:control',
    'groups.disband': 'session:control',
    'groups.send': 'session:submit',
    'groups.attachment.upload': 'session:submit',
    'groups.attachment.download': 'session:read',
    'groups.stop': 'session:control',
    'groups.continue': 'session:control',
    'groups.retry': 'session:control',
    'groups.discard': 'session:control',
    'groups.approve': 'session:approve',
    'groups.peer.register': 'session:control',
    'groups.peer.invite': 'session:operator',
    'groups.peer.revoke': 'session:operator',
}
_FIELDS = {
    'groups.capabilities': set(),
    'groups.list': {'limit', 'offset', 'include_disbanded'},
    'groups.state': {'room_id', 'include_disbanded'},
    'groups.log': {'room_id', 'since_seq', 'limit', 'include_disbanded'},
    'groups.create': {'room_id', 'name', 'members'},
    'groups.rename': {'room_id', 'event_id', 'name'},
    'groups.context': {'room_id', 'event_id', 'purpose', 'notes', 'routing', 'lead_member_id'},
    'groups.disband': {'room_id', 'cancel_id'},
    'groups.send': {'room_id', 'event_id', 'payload'},
    'groups.attachment.upload': {'room_id', 'upload_id', 'kind', 'name', 'mime', 'data_base64'},
    'groups.attachment.download': {'room_id', 'event_id', 'attachment_id'},
    'groups.stop': {'room_id', 'cancel_id'},
    'groups.continue': {'room_id', 'discussion_event_id'},
    'groups.retry': {'room_id', 'member_id', 'task_id', 'execution_generation'},
    'groups.discard': {'room_id', 'member_id', 'task_id', 'execution_generation'},
    'groups.approve': {'room_id', 'member_id', 'task_id', 'execution_generation',
                       'choice', 'request_id'},
    'groups.peer.register': {'room_id', 'member_id', 'target_url', 'target_profile', 'grant', 'catalog'},
    'groups.peer.invite': {'room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
                           'member_id', 'ttl_seconds', 'status_ttl_seconds'},
    'groups.peer.revoke': {'grant'},
    'profiles.list': {'include_sessions'},
}


async def dispatch_group_control(connection, method, params):
    authority, actor = connection.authority, connection.actor
    capability = GROUP_METHODS.get(method, 'session:read')
    if capability not in actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    if actor.profile_id != authority.profile_id:
        raise AuthorityRefusal('profile_mismatch')
    if method not in _FIELDS or not isinstance(params, dict) or set(params) - (_FIELDS[method] | {'profile'}):
        raise AuthorityRefusal('invalid_params')
    from persistence_assembly import profile_scope
    home = Path(authority.profile_id)
    if _store(authority).scope != await asyncio.to_thread(profile_scope, home=home):
        raise AuthorityRefusal('profile_mismatch')
    from hermes_cli.profiles import profile_matches_home
    profile = params.get('profile')
    if profile is not None and not isinstance(profile, str):
        raise AuthorityRefusal('invalid_params')
    if profile and not profile_matches_home(profile, home):
        raise AuthorityRefusal('profile_mismatch')
    supplied = {key: value for key, value in params.items() if key != 'profile'}
    siblings = _roster_siblings(connection, supplied) if method == 'profiles.list' else ()

    def invoke():
        from gateway.run import _profile_runtime_scope
        from persistence.gateway.hosted_rooms import HostedRoomError
        from gateway import session_group_peers as peers
        with _profile_runtime_scope(home):
            if method == 'profiles.list':
                return _profiles(authority, actor, home, supplied, siblings)
            try:
                if method in peers.TARGET_METHODS:
                    return peers.dispatch_target(authority, method, supplied)
                return _group(authority, actor, method, supplied, host_scope=connection.host_scoped)
            except AuthorityRefusal:
                raise
            except HostedRoomError as exc:
                refusal = _refusal(getattr(exc, 'reason', None) or 'invalid_params',
                                   admitted=getattr(exc, 'admitted', None))
                if getattr(exc, 'reason', None):
                    refusal.wire_message = str(exc)
                raise refusal from exc
            except (ValueError, TypeError) as exc:
                raise _refusal('invalid_params', admitted=getattr(exc, 'admitted', None)) from exc
    return await asyncio.to_thread(invoke)


def _refusal(reason, *, admitted=None):
    """The wire refusal; ``admitted=False`` (a groups.send refused before its user event
    committed: ``HostedRoomService.send``, or no driver to take it) rides along as error data."""
    refusal = AuthorityRefusal(reason)
    if admitted is False:
        refusal.admitted = False
    return refusal


def _store(authority):
    """*authority*'s storage interface handle; without one, ``StoreUnavailable``."""
    store = authority_persistence(authority)
    if store is None:
        raise StoreUnavailable('the authority has no storage interface handle')
    return store


def _group(authority, actor, method, params, *, host_scope=False):
    from gateway import hosted_rooms as rooms
    from gateway.hosted_room_context import groups_state_result, set_room_context
    if method == 'groups.disband':
        params = {**params, 'room_id': rooms._room_id(params.get('room_id'))}
    store = _store(authority)
    gateway_id = rooms.local_authority_gateway_id()
    service = getattr(authority, 'hosted_room_service', None)
    room_authorizer = getattr(service, 'authorize_room', None)
    if service is not None:
        if service.placement != 'profile' or service.store.scope != store.scope:
            raise AuthorityRefusal('profile_mismatch')
        status = service.runtime.status()
        if not status.get('running') or status.get('stopping'):
            service = None

    execution_methods = {'groups.send', 'groups.stop', 'groups.continue', 'groups.retry', 'groups.discard',
                         'groups.approve'}
    owner = None
    if getattr(authority, 'hosted_room_service', None) is not None and 'room_id' in params:
        if room_authorizer is None:
            raise AuthorityRefusal('permission_denied')
        if method == 'groups.create':
            # The owner claim runs inside the room insert's write: a create rejected by
            # validation or storage must not leave an owner-only reservation behind.
            owner = actor.subject
        else:
            room_authorizer(actor.subject, params['room_id'])
    if method in {'groups.attachment.upload', 'groups.attachment.download'}:
        if service is None:
            raise AuthorityRefusal('runtime_coordination_required')
        from gateway.session_hosted_attachments import upload, download
        handler = upload if method == 'groups.attachment.upload' else download
        return handler(service, actor, params)
    if method in execution_methods:
        if service is None:
            if method == 'groups.send':
                raise _refusal('runtime_coordination_required', admitted=False)
            raise AuthorityRefusal('runtime_coordination_required')
        if not params.get('room_id'):
            raise AuthorityRefusal('invalid_params')
        return _execution_control(service, method, params)
    if method == 'groups.peer.register':
        if service is None:
            raise AuthorityRefusal('runtime_coordination_required')
        from gateway.session_group_peers import register
        return register(service, params)

    def capabilities():
        from gateway.session_group_peers import room_link
        from gateway.session_hosted_members import member_roster
        # The roster groups.create will accept from THIS caller, so a picker never offers a bot
        # (parked, retiring, a sibling a profile-bound socket cannot seat) that create refuses.
        homes, refused = member_roster(authority, siblings=host_scope)
        return {'protocol_version': rooms.PROTOCOL_VERSION, 'driver': service is not None,
                'persistent_process': True, 'authority_gateway_id': gateway_id,
                'room_link': room_link(authority),
                'features': ['room_identity', 'monotonic_log', 'replayable_disband', 'peer_renewal_horizon'],
                'methods': list(GROUP_METHODS), 'max_log_limit': rooms.MAX_LOG_LIMIT,
                'seatable_profiles': sorted(name for name in homes if name not in refused)}

    def listing():
        limit, offset = params.get('limit', rooms.MAX_ROOM_LIST_LIMIT), params.get('offset', 0)
        if getattr(authority, 'hosted_room_service', None) is None:
            result = rooms.list_rooms(store, placement='profile', **params)
            return {'rooms': result, 'next_offset': offset + limit if len(result) == limit else None}
        authorize = room_authorizer
        if authorize is None:
            raise AuthorityRefusal('permission_denied')

        def visible(room):
            try:
                authorize(actor.subject, room['room_id'])
            except AuthorityRefusal as exc:
                if exc.reason != 'permission_denied':
                    raise
                return False
            return True
        # Page the actor-visible collection, not the room table: scan the underlying cursor until
        # the page holds ``limit`` visible rooms, and continue at the next VISIBLE room (or None),
        # so a page is never short or empty while visible rooms remain after it. ``offset`` is
        # that underlying continuation cursor, as returned in ``next_offset``.
        found, cursor, next_offset = [], offset, None
        while next_offset is None:
            batch = rooms.list_rooms(store, placement='profile', **{**params, 'offset': cursor})
            for room in batch:
                if visible(room):
                    if len(found) == limit:
                        next_offset = cursor
                        break
                    found.append(room)
                cursor += 1
            if len(batch) < limit:
                break
        return {'rooms': found, 'next_offset': next_offset}

    def create():
        from gateway.hosted_room_discussion import DiscussionValidationError, validate_roster
        from gateway.session_hosted_members import member_roster, refuse_unseatable
        # Only a host-scoped grant, the one that may address siblings directly
        # (``AuthorityConnection._sibling_for``), seats served siblings with no entry; a
        # profile-bound one seats its own profile plus its hosted_rooms.profiles entries, as
        # before. Both create paths validate here, by the driver's rule.
        homes, refused = member_roster(authority, siblings=host_scope)
        refuse_unseatable(params.get('members'), refused)
        try:
            members = validate_roster(params.get('members'), local_profiles=homes)
            if service is not None:
                return {'room': service.create_room(**params, owner=owner)}
        except DiscussionValidationError as exc:
            # Name the refused member; the bare code left Desktop and gateway.log silent (#139002).
            raise AuthorityRefusal('invalid_params', field='members', detail=str(exc)) from exc
        normalized = [{'member_id': m.member_id, 'profile': m.profile, 'handle': m.handle,
                       'target': dict(m.target or {}),
                       **({'display_name': m.display_name} if m.display_name else {})} for m in members]
        if owner is not None:
            return {'room': authority.hosted_room_service.insert_room(
                room_id=params.get('room_id'), name=params.get('name'), members=normalized, owner=owner)}
        return {'room': rooms.create_room(store, placement='profile', room_id=params.get('room_id'),
                name=params.get('name'), members=normalized, authority_gateway_id=gateway_id)}

    def disband():
        return _disband(getattr(authority, 'hosted_room_service', None), service, store, gateway_id, params)

    def state():
        return groups_state_result(
            store, rooms.room_state(store, placement='profile', **params), service, placement='profile')

    handlers = {
        'groups.capabilities': capabilities,
        'groups.list': listing,
        'groups.create': create,
        'groups.state': state,
        'groups.log': lambda: rooms.read_events(store, placement='profile', **params),
        'groups.rename': lambda: {'room': rooms.rename_room(store, placement='profile', **params)},
        'groups.context': lambda: set_room_context(store, placement='profile', **params),
        'groups.disband': disband,
    }
    return handlers[method]()


def _disband(attached, service, store, gateway_id, params):
    """Stop the room's work (waiting for every member's acknowledged Stop), revoke each peer
    route at its member gateway, then tombstone; any refusal leaves the room for a retry."""
    from gateway import hosted_rooms as rooms
    from gateway.hosted_room_driver import list_tasks
    from persistence.gateway.hosted_rooms import RoomHeldGrantsPendingError
    from tui_gateway.hosted_room_service import disband_fence, route_publication_lock
    room_id = rooms._room_id(params.get('room_id'))
    with disband_fence(attached, room_id):
        state = rooms.room_state(store, placement='profile', room_id=room_id, include_disbanded=True)
        if (state.get('disbanded_at') is None and attached is None
                and rooms.room_has_peer_obligations(store, placement='profile', room_id=room_id)):
            raise AuthorityRefusal('runtime_coordination_required')
        if service is not None and state.get('disbanded_at') is None:
            service.stop_room(room_id, cancel_id=params.get('cancel_id') or 'room-disbanded',
                              require_acknowledged=True)
        with route_publication_lock(attached):
            state = rooms.room_state(store, placement='profile', room_id=room_id, include_disbanded=True)
            # Metadata control must not destroy an active execution or bypass Stop.
            if service is None and any(list_tasks(store, placement='profile', room_id=room_id, status=status)
                   for status in ('queued', 'running', 'stopping', 'indeterminate', 'deferred')):
                raise AuthorityRefusal('runtime_coordination_required')
        for attempt in range(3):
            if state.get('disbanded_at') is None and attached is not None:
                _revoke_routes_first(attached, room_id)
            try:
                with route_publication_lock(attached):
                    state = rooms.room_state(store, placement='profile', room_id=room_id, include_disbanded=True)
                    return {'tombstone': rooms.disband_room(store, placement='profile', room_id=room_id,
                            expected_gateway_id=gateway_id, expected_epoch=state['authority_epoch'])}
            except RoomHeldGrantsPendingError:
                if attached is None:
                    raise AuthorityRefusal('runtime_coordination_required')
                if attempt == 2:
                    raise


def _revoke_routes_first(service, room_id):
    """Revoke each peer route at its member gateway; one that cannot be reached keeps the room."""
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    try:
        service.revoke_room_routes(room_id)
    except PeerRunsHTTPError as exc:
        raise AuthorityRefusal('peer_unreachable') from exc


def _execution_control(service, method, params):
    def send():
        from gateway.hosted_rooms import user_event_id
        event = service.send(room_id=params.get('room_id'),
                             event_id=user_event_id(params.get('event_id')),
                             payload=params.get('payload'))
        return {'event': event, 'client_event_id': params.get('event_id'),
                'accepted': True, 'driver_started': True}

    def attempt_control():
        if (type(params.get('execution_generation')) is not int
                or params['execution_generation'] < 1
                or not params.get('member_id') or not params.get('task_id')):
            raise AuthorityRefusal('invalid_params')
        operation = service.discard_room_task if method == 'groups.discard' else service.retry_room_task
        task = operation(**params)
        identity = task['identity']
        receipt = {field: getattr(identity, field) for field in
                   ('room_id', 'task_id', 'thread_id', 'turn_id')}
        receipt.update({field: task[field] for field in
                        ('status', 'execution_generation', 'cancel_generation')})
        return {'discarded' if method == 'groups.discard' else 'retried': True, 'task': receipt}

    def approve():
        if (type(params.get('execution_generation')) is not int
                or params['execution_generation'] < 1
                or params.get('choice') not in {'once', 'deny'}
                or not isinstance(params.get('request_id'), str) or not params['request_id']):
            raise AuthorityRefusal('invalid_params')
        try:
            return {'approved': True, 'result': service.approve_room_task(**params)}
        except RuntimeError as exc:
            if str(exc) == 'room approval is no longer pending':
                raise AuthorityRefusal(str(exc)) from exc
            raise

    handlers = {
        'groups.send': send,
        'groups.stop': lambda: {'cancelled': service.stop_room(
            params.get('room_id'), cancel_id=params.get('cancel_id') or uuid.uuid4().hex)},
        'groups.continue': lambda: {'continued': True, 'event': service.continue_room(
            params.get('room_id'), discussion_event_id=str(params.get('discussion_event_id') or ''))},
        'groups.retry': attempt_control,
        'groups.discard': attempt_control,
        'groups.approve': approve,
    }
    return handlers[method]()


# Sibling authorities whose last roster read failed: warned once per outage, then logged at debug.
# Weak, so an unserved profile drops out with its authority and a re-served one (a new authority)
# is warned again.
_DEGRADED = weakref.WeakSet()


def _roster_siblings(connection, params):
    """The other local bots this process serves (``session_authorities.local_bots``, the set group
    seating uses too), as ``[(bot, reader)]``, snapshotted on the owner loop, where serve and
    unserve mutate the registry; an unbuilt slot is skipped. ``reader`` is the principal that row's
    session fields are read as, or None for a metadata-only row: only a host-scoped socket routes
    session verbs to a sibling (``AuthorityConnection._sibling_for``), and only a bot addressable
    by its own name (no ``problem``) is the one a click on its row opens. It carries this socket's
    subject and capabilities, which is what the routed connection would carry; none is built, so
    no poll (routed or not) adds connections."""
    from gateway.session_authorities import local_bots
    from gateway.session_contract import SessionActor
    authority, actor = connection.authority, connection.actor
    if getattr(getattr(authority, 'runner', None), 'session_authorities', None) is None:
        return []
    readable = connection.host_scoped and params.get('include_sessions', True) is True
    return [(bot, SessionActor(actor.subject, bot.authority.profile_id, actor.capabilities, actor.transport_id)
             if readable and bot.problem is None else None)
            for bot in local_bots(authority.runner) if bot.authority is not authority]


def _bot_sessions(authority, actor):
    """A roster row's session fields, read from *authority*'s own store as *actor*: the exact
    "Bot Chat" title (the status dot's session and the row's age) and the newest visible local
    conversation, each only when the actor owns it or holds operator access. Reused while the
    store has not moved, as the pre-unified roster did (#117257): every connection polls every 5 s.
    Discovery is read-only: no restore/unarchive and no new store handle."""
    from tui_gateway.profile_roster_cache import cached_reader_session_fields
    operator = 'session:operator' in actor.capabilities
    return cached_reader_session_fields(_store(authority).scope.profile_dir, (actor.subject, operator),
                                        lambda: _read_bot_sessions(_store(authority), actor.subject, operator))


def _read_bot_sessions(store, subject, operator):
    """*store*'s roster fields. A store closed meanwhile (its profile unserved mid-poll) refuses rather
    than reopening a writer into a home that is being deleted."""
    from dataclasses import asdict
    roster = store.sessions.roster_sessions(owner_user_id=None if operator else subject, owner=owner_rule())
    return {name: asdict(value) if value is not None else None for name, value in (
        ('last_session', roster.last_session), ('worker_session', roster.worker_session),
        ('canonical_session', roster.canonical_session))}


def _sibling_sessions(authority, reader):
    """A sibling row's session fields. A retiring sibling is not read: unserve marks it before it
    closes the store, and this runs right before the read (a close that still lands mid-read is
    refused by ``reopen=False``). Any failure leaves only that row metadata-only, as the
    pre-unified roster degraded per profile, instead of failing the whole roster."""
    if authority.retiring or authority.retired:
        return {}
    try:
        fields = _bot_sessions(authority, reader)
    except Exception as exc:  # health: allow BLE001 -- one bot's store must never fail the whole roster
        log = logger.debug if authority in _DEGRADED else logger.warning
        _DEGRADED.add(authority)
        log('Bots roster: %s is metadata-only, its session store did not answer (%s)', authority.profile_id, exc)
        return {}
    _DEGRADED.discard(authority)
    return fields


def _profiles(authority, actor, home, params, siblings=()):
    """The Bots roster: this profile's row, then every profile it can name. ``siblings`` is
    ``_roster_siblings``' snapshot; a sibling with a reader carries the same session fields."""
    from hermes_cli.profiles import _profile_info
    import hermes_yaml as yaml
    include_sessions = params.get('include_sessions', True)
    if type(include_sessions) is not bool:
        raise AuthorityRefusal('invalid_params')
    from gateway.session_authorities import served_profile_name
    name = served_profile_name(home)
    profile = _profile_info(name, home, is_default=name == 'default')
    row = {'name': name, 'path': str(home), 'is_default': profile.is_default,
           'model': profile.model, 'provider': profile.provider,
           'description': profile.description or '', 'display_name': profile.display_name or '',
           'skill_count': profile.skill_count or 0}
    path = home / 'profile.yaml'
    meta = yaml.safe_load(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}
    meta = meta if isinstance(meta, dict) else {}
    row['ui_meta_revisions'] = _ui_meta_revisions(meta)
    if isinstance(meta.get('ui_meta'), dict):
        row['ui_meta'] = meta['ui_meta']
    row['has_avatar'] = _has_avatar(home)
    if include_sessions:
        row.update(_bot_sessions(authority, actor))
    profiles = [row]
    seen = {home.resolve()}

    def discovered(configured, target, reach=None):
        # Discovery alone is never a licence to read another owner's state/configuration: only a
        # sibling this connection already routes to (``reach``) adds its session fields.
        if target.resolve() in seen:
            return
        seen.add(target.resolve())
        entry = {'name': configured, 'path': str(target), 'is_default': configured == 'default',
                 'model': '', 'provider': '', 'description': '', 'display_name': '',
                 'skill_count': 0, 'has_avatar': _has_avatar(target), 'ui_meta_revisions': {}}
        if include_sessions and reach is not None:
            entry.update(_sibling_sessions(*reach))
        meta = _discovered_profile_meta(configured, target / 'profile.yaml')
        if isinstance(meta, dict):
            # The profile's own metadata, not a placeholder: Desktop seeds the Edit profile dialog
            # from these fields and skips the avatar push when ``has_avatar`` is set.
            entry['description'] = str(meta.get('description') or '').strip()
            entry['display_name'] = str(meta.get('display_name') or '')
            entry['ui_meta_revisions'] = _ui_meta_revisions(meta)
            if isinstance(meta.get('ui_meta'), dict):
                entry['ui_meta'] = meta['ui_meta']
        profiles.append(entry)

    # Sibling profiles this process multiplexes: the roster lists them with the same
    # standing as a hosted-room destination, and the client routes their sessions by name.
    for bot, reader in siblings:
        # The row keeps the name its home resolves to (the client routes by it); a bot that name
        # does not reach (a linked ``profiles/<name>`` reads as ``default``) has no reader, so its
        # row stays metadata-only.
        discovered(served_profile_name(bot.home), bot.home,
                   (bot.authority, reader) if reader is not None else None)
    service = getattr(authority, 'hosted_room_service', None)
    if service is not None:
        for configured, target in service.profile_homes().items():
            discovered(configured, target)
    # Every other profile under this root, as the pre-authority ``profiles.list`` named them
    # (RT-38): a host running one gateway per profile serves no siblings, and Desktop drops every
    # union agent the active source leaves out, so a default-only answer hid the other bots.
    # Metadata only, like any discovery above.
    from hermes_cli.profiles import _get_default_hermes_home
    root = _get_default_hermes_home()
    if root.is_dir():
        discovered('default', root)
    for target in _named_profile_dirs():
        discovered(target.name, target)
    # ``install_id`` (same value as /api/status) lets a multi-connection client prove which
    # machine answered; the roster trace logged ``answered=-`` without it. It already returns
    # None for an unreadable root.
    from hermes_cli.install_identity import get_install_id
    return {'profiles': profiles, 'bot_mode_protocol': True, 'install_id': get_install_id() or ''}


# The Bots pane polls ``profiles.list`` every 5 s. Re-listing ``profiles/`` and re-parsing every
# sibling ``profile.yaml`` each time cost ~60 ms at 50 profiles, so both are cached, and neither
# cache trusts a timestamp to notice a change. Linux stamps mtime AND ctime from the coarse clock
# (one jiffy, 1-10 ms), FAT/SMB/HFS+ tick coarser still, and a directory's size and inode survive
# a file appearing and vanishing inside it; so a create-then-delete inside one tick leaves every
# stat field of the profile dir as it was. The listing is therefore keyed on what
# ``_iter_named_profile_dirs`` actually decides by (each entry's identity markers and tombstone,
# present or not), and a parsed ``profile.yaml`` on the file's bytes: the read is cheap, the
# YAML parse is the cost the cache avoids.
_NAMED_DIRS_CACHE: dict[str, tuple[tuple, list]] = {}
_PROFILE_META_CACHE: dict[str, tuple[bytes, object]] = {}


def _stat_key(stat):
    return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)


def _lstat_key(path):
    try:
        stat = os.lstat(path)
    except OSError:
        return None
    return (stat.st_mode, *_stat_key(stat))


def _named_profile_dirs():
    from hermes_cli.profiles import _get_profiles_root, _iter_named_profile_dirs
    from hermes_constants import _PROFILE_IDENTITY_MARKERS, profile_tombstone_path
    profiles_root = _get_profiles_root()
    try:
        with os.scandir(profiles_root) as entries:
            rows = []
            for entry in entries:
                # Presence (not just timestamps) of every marker ``named_profile_has_identity``
                # checks, and of the tombstone ``named_profile_is_deleted`` checks: one lstat each.
                markers = tuple(_lstat_key(os.path.join(entry.path, marker))
                                for marker in _PROFILE_IDENTITY_MARKERS)
                tombstone = os.path.exists(profile_tombstone_path(Path(entry.path)))
                rows.append((entry.name, _stat_key(entry.stat()), markers, tombstone))
            signature = (_stat_key(os.stat(profiles_root)), tuple(sorted(rows)))
    except OSError:
        signature = None
    cached = _NAMED_DIRS_CACHE.get(str(profiles_root))
    if signature is not None and cached and cached[0] == signature:
        return cached[1]
    try:
        targets = _iter_named_profile_dirs()
    except OSError as exc:
        # An unreadable ``profiles/`` drops the siblings, never the authority's own roster.
        logger.warning('profiles.list: cannot list %s: %s', profiles_root, exc)
        return []
    if signature is not None:
        _NAMED_DIRS_CACHE[str(profiles_root)] = (signature, targets)
    return targets


_ASSET_EXTS = ('png', 'jpg', 'webp')


def _has_avatar(home):
    """Cheap existence flag so a roster paint skips a ``get_asset`` probe per profile."""
    return any((home / 'assets' / f'avatar.{ext}').is_file() for ext in _ASSET_EXTS)


def _ui_meta_revisions(meta):
    """``_ui_meta_revisions`` from a parsed profile.yaml: str keys, non-bool ints, floor 0.

    Always present (``{}`` when the profile never wrote one) so a client can feature-detect
    gateway-owned CAS instead of reading a missing field as revision zero.
    """
    revisions = meta.get('_ui_meta_revisions') if isinstance(meta, dict) else None
    return {str(k): max(0, v) for k, v in revisions.items()
            if type(v) is int} if isinstance(revisions, dict) else {}


def _discovered_profile_meta(name, meta_path):
    """A sibling's parsed ``profile.yaml``, or ``{}``. One unreadable file (a legacy code page,
    bad YAML, a permission error) degrades that row to basic metadata; it never fails the roster."""
    import hermes_yaml as yaml
    try:
        raw = meta_path.read_bytes()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return {}
    except OSError as exc:
        logger.warning('profiles.list: skipping unreadable profile.yaml for %s (%s): %s', name, meta_path, exc)
        return {}
    cached = _PROFILE_META_CACHE.get(str(meta_path))
    if cached and cached[0] == raw:
        return cached[1]
    try:
        meta = yaml.safe_load(raw.decode('utf-8-sig'))
    except (ValueError, yaml.YAMLError) as exc:
        logger.warning('profiles.list: skipping unreadable profile.yaml for %s (%s): %s', name, meta_path, exc)
        meta = {}
    _PROFILE_META_CACHE[str(meta_path)] = (raw, meta)
    return meta
