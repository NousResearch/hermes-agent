"""Profile-owned room controls over the canonical hosted service."""
import asyncio
import logging
from pathlib import Path
import weakref

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


GROUP_METHODS = {
    'groups.capabilities': 'session:read',
    'groups.list': 'session:read',
    'groups.state': 'session:read',
    'groups.log': 'session:read',
    'groups.create': 'session:control',
    'groups.rename': 'session:control',
    'groups.disband': 'session:control',
    'groups.send': 'session:submit',
    'groups.attachment.upload': 'session:submit',
    'groups.attachment.download': 'session:read',
    'groups.stop': 'session:control',
    'groups.retry': 'session:control',
    'groups.discard': 'session:control',
    'groups.approve': 'session:approve',
}
_FIELDS = {
    'groups.capabilities': set(),
    'groups.list': {'limit', 'offset', 'include_disbanded'},
    'groups.state': {'room_id', 'include_disbanded'},
    'groups.log': {'room_id', 'since_seq', 'limit', 'include_disbanded'},
    'groups.create': {'room_id', 'name', 'members'},
    'groups.rename': {'room_id', 'event_id', 'name'},
    'groups.disband': {'room_id', 'cancel_id'},
    'groups.send': {'room_id', 'event_id', 'payload'},
    'groups.attachment.upload': {'room_id', 'upload_id', 'kind', 'name', 'mime', 'data_base64'},
    'groups.attachment.download': {'room_id', 'event_id', 'attachment_id'},
    'groups.stop': {'room_id', 'cancel_id'},
    'groups.retry': {'room_id', 'member_id', 'task_id', 'execution_generation'},
    'groups.discard': {'room_id', 'member_id', 'task_id', 'execution_generation'},
    'groups.approve': {'room_id', 'member_id', 'task_id', 'execution_generation',
                       'choice', 'request_id'},
    'profiles.list': {'include_sessions'},
}


async def dispatch_group_control(connection, method, params):
    authority, actor = connection.authority, connection.actor
    capability = GROUP_METHODS.get(method, 'session:read')
    if capability not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if method not in _FIELDS or not isinstance(params, dict) or set(params) - (_FIELDS[method] | {'profile'}):
        raise RuntimeStoreError('invalid_params')
    home = Path(authority.profile_id)
    db_path = await asyncio.to_thread(Path(authority.db.db_path).resolve)
    if db_path.parent != await asyncio.to_thread(home.resolve):
        raise RuntimeStoreError('profile_mismatch')
    from hermes_cli.profiles import profile_matches_home
    profile = params.get('profile')
    if profile is not None and not isinstance(profile, str):
        raise RuntimeStoreError('invalid_params')
    if profile and not profile_matches_home(profile, home):
        raise RuntimeStoreError('profile_mismatch')
    supplied = {key: value for key, value in params.items() if key != 'profile'}
    siblings = _roster_siblings(connection, supplied) if method == 'profiles.list' else ()

    def invoke():
        from gateway.run import _profile_runtime_scope
        from gateway.hosted_rooms import HostedRoomError
        with _profile_runtime_scope(home):
            if method == 'profiles.list':
                return _profiles(authority, actor, home, supplied, siblings)
            try:
                return _group(authority, actor, method, supplied, host_scope=connection.host_scoped)
            except RuntimeStoreError:
                raise
            except HostedRoomError as exc:
                raise RuntimeStoreError(getattr(exc, 'reason', None) or 'invalid_params') from exc
            except (ValueError, TypeError) as exc:
                raise RuntimeStoreError('invalid_params') from exc
    return await asyncio.to_thread(invoke)


def _group(authority, actor, method, params, *, host_scope=False):
    from gateway import hosted_rooms as rooms
    db_path = authority.db.db_path
    gateway_id = rooms.local_authority_gateway_id()
    service = getattr(authority, 'hosted_room_service', None)
    room_authorizer = getattr(service, 'authorize_room', None)
    if service is not None:
        if Path(service.db_path).resolve() != Path(db_path).resolve():
            raise RuntimeStoreError('profile_mismatch')
        status = service.runtime.status()
        if not status.get('running') or status.get('stopping'):
            service = None

    execution_methods = {'groups.send', 'groups.stop', 'groups.retry', 'groups.discard', 'groups.approve'}
    admit = None
    if getattr(authority, 'hosted_room_service', None) is not None and 'room_id' in params:
        if room_authorizer is None:
            raise RuntimeStoreError('permission_denied')
        if method == 'groups.create':
            # Claim ownership inside the room insert's transaction: a create rejected by
            # validation or storage must not leave an owner-only reservation behind.
            def admit(conn):
                room_authorizer(actor.subject, params['room_id'], create=True, conn=conn)
        else:
            room_authorizer(actor.subject, params['room_id'])
    if method in {'groups.attachment.upload', 'groups.attachment.download'}:
        if service is None:
            raise RuntimeStoreError('runtime_coordination_required')
        from gateway.session_hosted_attachments import upload, download
        handler = upload if method == 'groups.attachment.upload' else download
        return handler(service, actor, params)
    if method in execution_methods:
        if service is None:
            raise RuntimeStoreError('runtime_coordination_required')
        if not params.get('room_id'):
            raise RuntimeStoreError('invalid_params')
        return _execution_control(service, method, params)

    def capabilities():
        return {'protocol_version': rooms.PROTOCOL_VERSION, 'driver': service is not None,
                'persistent_process': True, 'authority_gateway_id': gateway_id,
                'room_link': {'enabled': False, 'reason': 'canonical_driver_required'},
                'features': ['room_identity', 'monotonic_log', 'replayable_disband'],
                'methods': list(GROUP_METHODS), 'max_log_limit': rooms.MAX_LOG_LIMIT}

    def listing():
        limit, offset = params.get('limit', rooms.MAX_ROOM_LIST_LIMIT), params.get('offset', 0)
        if getattr(authority, 'hosted_room_service', None) is None:
            result = rooms.list_rooms(db_path, **params)
            return {'rooms': result, 'next_offset': offset + limit if len(result) == limit else None}
        authorize = room_authorizer
        if authorize is None:
            raise RuntimeStoreError('permission_denied')

        def visible(room):
            try:
                authorize(actor.subject, room['room_id'])
            except RuntimeStoreError as exc:
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
            batch = rooms.list_rooms(db_path, **{**params, 'offset': cursor})
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
                return {'room': service.create_room(**params, admit=admit)}
        except DiscussionValidationError as exc:
            # Name the refused member; the bare code left Desktop and gateway.log silent (#139002).
            raise RuntimeStoreError('invalid_params', field='members', detail=str(exc)) from exc
        normalized = [{'member_id': m.member_id, 'profile': m.profile, 'handle': m.handle,
                       'target': dict(m.target or {}),
                       **({'display_name': m.display_name} if m.display_name else {})} for m in members]
        return {'room': rooms.create_room(db_path, room_id=params.get('room_id'),
                name=params.get('name'), members=normalized, authority_gateway_id=gateway_id,
                admit=admit)}

    def disband():
        from gateway.hosted_room_driver import list_tasks
        state = rooms.room_state(db_path, room_id=params.get('room_id'), include_disbanded=True)
        if service is not None and state.get('disbanded_at') is None:
            service.stop_room(params.get('room_id'),
                              cancel_id=params.get('cancel_id') or 'room-disbanded',
                              require_acknowledged=True)
            service.revoke_room_routes(params.get('room_id'))
        # Metadata control must not destroy an active execution or bypass Stop.
        if service is None and any(list_tasks(db_path, room_id=params.get('room_id'), status=status)
               for status in ('queued', 'running', 'stopping', 'indeterminate', 'deferred')):
            raise RuntimeStoreError('runtime_coordination_required')
        state = rooms.room_state(db_path, room_id=params.get('room_id'), include_disbanded=True)
        return {'tombstone': rooms.disband_room(db_path, room_id=params.get('room_id'),
                expected_gateway_id=gateway_id, expected_epoch=state['authority_epoch'])}

    def state():
        room = rooms.room_state(db_path, **params)
        result = {'room': room}
        if service is not None and room.get('disbanded_at') is None:
            result['driver_status'] = service.status(room['room_id'])
        return result

    handlers = {
        'groups.capabilities': capabilities,
        'groups.list': listing,
        'groups.create': create,
        'groups.state': state,
        'groups.log': lambda: rooms.read_events(db_path, **params),
        'groups.rename': lambda: {'room': rooms.rename_room(db_path, **params)},
        'groups.disband': disband,
    }
    return handlers[method]()


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
            raise RuntimeStoreError('invalid_params')
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
            raise RuntimeStoreError('invalid_params')
        return {'approved': True, 'result': service.approve_room_task(**params)}

    handlers = {
        'groups.send': send,
        'groups.stop': lambda: {'cancelled': service.stop_room(
            params.get('room_id'), cancel_id=params.get('cancel_id') or 'desktop-stop')},
        'groups.retry': attempt_control,
        'groups.discard': attempt_control,
        'groups.approve': approve,
    }
    return handlers[method]()


# A local chat: minted by this owner (``local-*``), or a pre-upgrade chat it adopted, which keeps its
# historical id as chat_id under a ``gateway.local_legacy.v1:`` binding. These are the chats
# ``resolve_local_target`` resolves by chat_id without adopting anything (its policy-receipt arm adds
# none: a receipt-bound chat_id that is not ``local-*`` is an adopted one). A messaging or API
# conversation with the same title is neither.
_LOCAL_CHAT = "(chat_id LIKE 'local-%' OR EXISTS (SELECT 1 FROM state_meta WHERE key = ? || sessions.chat_id))"
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
    from gateway.session_contract import Principal
    authority, actor = connection.authority, connection.actor
    if getattr(getattr(authority, 'runner', None), 'session_authorities', None) is None:
        return []
    readable = connection.host_scoped and params.get('include_sessions', True) is True
    return [(bot, Principal(actor.subject, bot.authority.profile_id, actor.capabilities, actor.transport_id)
             if readable and bot.problem is None else None)
            for bot in local_bots(authority.runner) if bot.authority is not authority]


def _bot_sessions(authority, actor):
    """A roster row's session fields, read from *authority*'s own store as *actor*: the exact
    "Bot Chat" title (the status dot's session and the row's age) and the newest visible local
    conversation, each only when the actor owns it or holds operator access. Reused while the
    store has not moved, as the pre-unified roster did (#117257): every connection polls every 5 s.
    Discovery is read-only: no legacy restore/unarchive or new SessionDB handle."""
    from tui_gateway.profile_roster_cache import cached_reader_session_fields
    operator = 'session:operator' in actor.capabilities
    return cached_reader_session_fields(Path(authority.db.db_path).parent, (actor.subject, operator),
                                        lambda: _read_bot_sessions(authority.db, actor.subject, operator))


def _read_bot_sessions(db, subject, operator):
    from hermes_state_local_migration import LEGACY_PREFIX
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES
    hidden_sources = ','.join('?' * len(INTERNAL_LISTING_SOURCES))
    owner_filter, owner_params = ('', ()) if operator else ('user_id=? AND ', (subject,))
    # One read connection for the whole row. A store closed meanwhile (its profile unserved
    # mid-poll) refuses rather than reopening a writer into a home that is being deleted.
    with db._read_ctx(reopen=False) as conn:
        def summary(root):
            tip = db._compression_tip_on(conn, root['id'])
            target = conn.execute('SELECT * FROM sessions WHERE id=?', (tip,)).fetchone() or root
            message = conn.execute(
                "SELECT content FROM messages WHERE session_id=? AND active=1 "
                "AND role IN ('user','assistant') AND TRIM(COALESCE(content,''))!='' "
                "ORDER BY id DESC LIMIT 1", (tip,)).fetchone()
            text = ' '.join(str(message[0] or '').split()) if message else ''
            return {'id': root['id'], 'resolved_id': tip, 'title': target['title'] or '',
                    'root_title': root['title'] or '',
                    'preview': text[:80] + '...' if len(text) > 80 else text,
                    'started_at': target['started_at'] or 0,
                    'last_active': target['last_activity_at'] or target['started_at'] or 0,
                    'message_count': target['message_count'] or 0}

        fields = {'last_session': None, 'worker_session': None, 'canonical_session': None}
        # The named registry is not a recency window and canonical chats are hidden.
        canonical = conn.execute(f'SELECT * FROM sessions WHERE title=? AND {_LOCAL_CHAT}',
                                 ('Bot Chat', LEGACY_PREFIX)).fetchone()
        if canonical and (canonical['user_id'] == subject or operator) and not canonical['archived']:
            fields['canonical_session'] = summary(canonical)
        # The newest conversation a picker shows: a hidden row (a Bot Chat, group member plumbing)
        # never wins, as in the pre-unified roster's listing.
        latest = conn.execute(
            f"SELECT * FROM sessions WHERE {owner_filter}{_LOCAL_CHAT} AND archived=0 AND hidden=0 "
            f"AND COALESCE(source,'') NOT IN ({hidden_sources}) "
            "ORDER BY COALESCE(last_activity_at,started_at) DESC LIMIT 1",
            (*owner_params, LEGACY_PREFIX, *INTERNAL_LISTING_SOURCES)).fetchone()
        if latest and fields['canonical_session'] and latest['id'] == canonical['id']:
            fields['last_session'] = dict(fields['canonical_session'])
        elif latest:
            fields['last_session'] = summary(latest)
    return fields


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
        raise RuntimeStoreError('invalid_params')
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
    revisions = meta.get('_ui_meta_revisions')
    row['ui_meta_revisions'] = {str(k): max(0, v) for k, v in revisions.items()
                              if type(v) is int} if isinstance(revisions, dict) else {}
    if isinstance(meta.get('ui_meta'), dict):
        row['ui_meta'] = meta['ui_meta']
    row['has_avatar'] = any((home / 'assets' / f'avatar.{ext}').is_file() for ext in ('png', 'jpg', 'webp'))
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
                 'model': '', 'provider': '', 'description': '', 'display_name': configured,
                 'skill_count': 0}
        if include_sessions and reach is not None:
            entry.update(_sibling_sessions(*reach))
        meta_path = target / 'profile.yaml'
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding='utf-8-sig')) if meta_path.is_file() else {}
        except (OSError, yaml.YAMLError):
            meta = {}
        if isinstance(meta, dict):
            entry['display_name'] = str(meta.get('display_name') or configured)
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
    return {'profiles': profiles, 'bot_mode_protocol': True}
