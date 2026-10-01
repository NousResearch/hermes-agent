"""Profile-owned room controls over the canonical hosted service."""
import asyncio
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


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
    'groups.attachment.list': 'session:read',
    'groups.stop': 'session:control',
    'groups.retry': 'session:control',
    'groups.discard': 'session:control',
    'groups.approve': 'session:approve',
    'groups.replica_state': 'session:operator',
    'groups.replication.prepare': 'session:control',
    'groups.replication.enroll': 'session:operator',
    'groups.replication.revoke': 'session:operator',
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
    'groups.disband': {'room_id', 'cancel_id', 'confirm_quarantined'},
    'groups.send': {'room_id', 'event_id', 'payload'},
    'groups.attachment.upload': {'room_id', 'upload_id', 'kind', 'name', 'mime', 'data_base64'},
    'groups.attachment.download': {'room_id', 'event_id', 'attachment_id'},
    'groups.attachment.list': {'room_id', 'cursor', 'limit', 'query', 'producer_member_id'},
    'groups.stop': {'room_id', 'cancel_id'},
    'groups.retry': {'room_id', 'member_id', 'task_id', 'execution_generation'},
    'groups.discard': {'room_id', 'member_id', 'task_id', 'execution_generation'},
    'groups.approve': {'room_id', 'member_id', 'task_id', 'execution_generation',
                       'choice', 'request_id'},
    'groups.peer.register': {'room_id', 'member_id', 'target_url', 'target_profile', 'grant', 'catalog'},
    'groups.peer.invite': {'room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
                           'member_id', 'ttl_seconds', 'status_ttl_seconds', 'replication', 'work_records', 'passive_only', 'request_id', 'requested_at'},
    'groups.peer.revoke': {'grant'},
    'groups.replica_state': {'room_id'},
    'groups.replication.prepare': {'room_id', 'target_install_id', 'endpoint', 'enrollment_id',
                                   'replace_enrollment_id'},
    'groups.replication.enroll': {'enrollment', 'expected_enrollment_id', 'expected_state'},
    'groups.replication.revoke': {'room_id', 'enrollment_id'},
    'profiles.list': {'include_sessions'},
}


async def dispatch_group_control(connection, method, params, *, author=None, remember=None):
    """Internal callers only (RPC clients are Desktop): ``author`` records who typed a
    ``groups.send``; ``remember`` names the chat whose ``groups.approve`` choice ``always``
    remembers the exact operation (``gateway.group_chat_rules``)."""
    authority, actor = connection.authority, connection.actor
    if author is not None and (method != 'groups.send' or type(author) is not dict
                               or author.get('kind') != 'user' or author.get('id') == 'desktop'):
        raise RuntimeStoreError('invalid_params')
    if remember is not None and (method != 'groups.approve' or type(remember) is not dict):
        raise RuntimeStoreError('invalid_params')
    capability = GROUP_METHODS.get(method, 'session:read')
    if capability not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if method not in _FIELDS or not isinstance(params, dict) or set(params) - (_FIELDS[method] | {'profile'}):
        raise RuntimeStoreError('invalid_params')
    home = Path(authority.profile_id)
    if Path(authority.db.db_path).resolve().parent != home.resolve():
        raise RuntimeStoreError('profile_mismatch')
    from hermes_cli.profiles import profile_matches_home
    profile = params.get('profile')
    if profile is not None and not isinstance(profile, str):
        raise RuntimeStoreError('invalid_params')
    if profile and not profile_matches_home(profile, home):
        raise RuntimeStoreError('profile_mismatch')
    supplied = {key: value for key, value in params.items() if key != 'profile'}

    def invoke():
        from gateway.run import _profile_runtime_scope
        from gateway.hosted_rooms import HostedRoomError
        from gateway import session_group_peers as peers
        from gateway import session_group_replication as replication
        with _profile_runtime_scope(home):
            if method == 'profiles.list':
                return _profiles(authority, actor, home, supplied)
            try:
                if method in peers.TARGET_METHODS:
                    return peers.dispatch_target(authority, method, supplied, actor_subject=actor.subject)
                if method in replication.TARGET_METHODS:
                    return replication.dispatch_target(authority, method, supplied)
                return _group(authority, actor, home, method, supplied, author=author, remember=remember)
            except RuntimeStoreError:
                raise
            except HostedRoomError as exc:
                from gateway.hosted_rooms import RoomNotFoundError
                reason = getattr(exc, 'reason', None) or ('room_not_found' if method == 'groups.disband' and isinstance(exc, RoomNotFoundError) else 'invalid_params')
                raise RuntimeStoreError(reason) from exc
            except (ValueError, TypeError) as exc:
                raise RuntimeStoreError('invalid_params') from exc
    return await asyncio.to_thread(invoke)


def _group(authority, actor, home, method, params, *, author=None, remember=None):
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
    if getattr(authority, 'hosted_room_service', None) is not None and 'room_id' in params:
        if room_authorizer is None:
            raise RuntimeStoreError('permission_denied')
        room_authorizer(actor.subject, params['room_id'], create=method == 'groups.create')
    if method in {'groups.attachment.upload', 'groups.attachment.download', 'groups.attachment.list'}:
        if service is None:
            raise RuntimeStoreError('runtime_coordination_required')
        from gateway.session_hosted_attachments import upload, download, list_files
        handler = {'groups.attachment.upload': upload, 'groups.attachment.download': download,
                   'groups.attachment.list': list_files}[method]
        return handler(service, actor, params)
    if method in execution_methods:
        if service is None:
            raise RuntimeStoreError('runtime_coordination_required')
        if not params.get('room_id'):
            raise RuntimeStoreError('invalid_params')
        return _execution_control(service, method, params, author=author, remember=remember)
    if method == 'groups.peer.register':
        if service is None:
            raise RuntimeStoreError('runtime_coordination_required')
        from gateway.session_group_peers import register
        return register(service, params)
    if method == 'groups.replication.prepare':
        if service is None:
            raise RuntimeStoreError('runtime_coordination_required')
        from gateway.session_group_replication import prepare
        return prepare(service, params)

    def capabilities():
        import time
        from gateway.session_group_peers import room_link
        return {'protocol_version': rooms.PROTOCOL_VERSION, 'driver': service is not None,
                'persistent_process': True, 'authority_gateway_id': gateway_id, 'server_time': time.time(),
                'room_link': room_link(authority),
                'features': ['room_identity', 'monotonic_log', 'replayable_disband', 'peer_setup_recovery'],
                'methods': list(GROUP_METHODS), 'max_log_limit': rooms.MAX_LOG_LIMIT}

    def listing():
        limit, offset = params.get('limit', rooms.MAX_ROOM_LIST_LIMIT), params.get('offset', 0)
        result = rooms.list_rooms(db_path, **params)
        next_offset = offset + limit if len(result) == limit else None
        if getattr(authority, 'hosted_room_service', None) is not None:
            visible = []
            for room in result:
                if room_authorizer is None:
                    raise RuntimeStoreError('permission_denied')
                try:
                    room_authorizer(actor.subject, room['room_id'])
                except RuntimeStoreError as exc:
                    if exc.reason != 'permission_denied':
                        raise
                else:
                    visible.append(room)
            result = visible
        return {'rooms': result, 'next_offset': next_offset}

    def create():
        if service is not None:
            return {'room': service.create_room(**params)}
        from gateway.hosted_room_discussion import validate_roster
        from gateway.session_authorities import served_profile_name
        name = served_profile_name(home)
        profiles = {name}
        if name == 'default' and (home / 'profiles').is_dir():
            profiles.update(path.name for path in (home / 'profiles').iterdir() if path.is_dir())
        members = validate_roster(params.get('members'), local_profiles=profiles)
        normalized = [{'member_id': m.member_id, 'profile': m.profile, 'handle': m.handle,
                       'target': dict(m.target or {}),
                       **({'display_name': m.display_name} if m.display_name else {})} for m in members]
        return {'room': rooms.create_room(db_path, room_id=params.get('room_id'),
                name=params.get('name'), members=normalized, authority_gateway_id=gateway_id)}

    def disband():
        from contextlib import nullcontext
        # Registration publishes under this lock too, so no peer route appears mid-Disband.
        with getattr(service, 'peer_route_lock', None) or nullcontext():
            return disband_unlocked()

    def disband_unlocked():
        from gateway.hosted_room_driver import list_tasks
        if rooms.quarantine_reason(db_path, room_id=params.get('room_id')) is not None:
            # Stop and route revocation would act as the room's authority: a quarantined room only
            # ends here, after explicit confirmation, with its history kept.
            return {'tombstone': rooms.disband_quarantined_room(
                db_path, room_id=params.get('room_id'), confirmed=params.get('confirm_quarantined') is True)}
        state = rooms.room_state(db_path, room_id=params.get('room_id'), include_disbanded=True)
        if service is not None and state.get('disbanded_at') is None:
            begin = getattr(service, 'begin_disband', None)
            if begin is not None:
                begin(params.get('room_id'))
            service.stop_room(params.get('room_id'),
                              cancel_id=params.get('cancel_id') or 'room-disbanded',
                              require_acknowledged=True)
            service.revoke_room_routes(params.get('room_id'))
        # Metadata control must not destroy an active execution or bypass Stop,
        # nor forget a peer route whose target grant only the driver can revoke.
        if service is None and any(list_tasks(db_path, room_id=params.get('room_id'), status=status)
               for status in ('queued', 'running', 'stopping', 'indeterminate', 'deferred')):
            raise RuntimeStoreError('runtime_coordination_required')
        if service is None and state.get('disbanded_at') is None and any(
                link['room_id'] == params.get('room_id') for link in rooms.list_room_link_records(db_path)):
            raise RuntimeStoreError('runtime_coordination_required')
        state = rooms.room_state(db_path, room_id=params.get('room_id'), include_disbanded=True)
        return {'tombstone': rooms.disband_room(db_path, room_id=params.get('room_id'),
                expected_gateway_id=gateway_id, expected_epoch=state['authority_epoch'])}

    def state():
        room = rooms.room_state(db_path, **params)
        result = {'room': room}
        if service is not None:
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


def _execution_control(service, method, params, *, author=None, remember=None):
    def send():
        from gateway.hosted_rooms import user_event_id
        event = service.send(room_id=params.get('room_id'),
                             event_id=user_event_id(params.get('event_id')),
                             payload=params.get('payload'), **({'actor': author} if author else {}))
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
                or params.get('choice') not in ({'always'} if remember is not None else {'once', 'deny'})
                or not isinstance(params.get('request_id'), str) or not params['request_id']):
            raise RuntimeStoreError('invalid_params')
        return {'approved': True, 'result': service.approve_room_task(
            **params, **({'remember': remember} if remember is not None else {}))}

    handlers = {
        'groups.send': send,
        'groups.stop': lambda: {'cancelled': service.stop_room(
            params.get('room_id'), cancel_id=params.get('cancel_id') or 'desktop-stop')},
        'groups.retry': attempt_control,
        'groups.discard': attempt_control,
        'groups.approve': approve,
    }
    return handlers[method]()


def _profiles(authority, actor, home, params):
    from hermes_cli.profiles import _profile_info, read_profile_meta
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
        row.update(last_session=None, worker_session=None, canonical_session=None)
        # Discovery is read-only: no legacy restore/unarchive or new SessionDB handle.
        def summary(session):
            tip = authority.db.get_compression_tip(session['id']) or session['id']
            target = authority.db.get_session(tip) or session
            with authority.db._lock:
                message = authority.db._conn.execute(
                    "SELECT content FROM messages WHERE session_id=? AND active=1 "
                    "AND role IN ('user','assistant') AND TRIM(COALESCE(content,''))!='' "
                    "ORDER BY id DESC LIMIT 1", (tip,)).fetchone()
            text = ' '.join(str(message[0] or '').split()) if message else ''
            return {'id': session['id'], 'resolved_id': tip, 'title': target.get('title') or '',
                    'root_title': session.get('title') or '',
                    'preview': text[:80] + '...' if len(text) > 80 else text,
                    'started_at': target.get('started_at') or 0,
                    'last_active': target.get('last_activity_at') or target.get('started_at') or 0,
                    'message_count': target.get('message_count') or 0}

        # The named registry is not a recency window and canonical chats are hidden.
        canonical = authority.db.get_session_by_title('Bot Chat')
        if (canonical and (canonical.get('user_id') == actor.subject
                           or 'session:operator' in actor.capabilities)
                and str(canonical.get('chat_id') or '').startswith('local-') and not canonical.get('archived')):
            row['canonical_session'] = summary(canonical)
        if 'session:operator' in actor.capabilities:
            owner_filter, owner_params = '', ()
        else:
            owner_filter, owner_params = 'user_id=? AND ', (actor.subject,)
        with authority.db._lock:
            latest = authority.db._conn.execute(
                f"SELECT id FROM sessions WHERE {owner_filter}chat_id LIKE 'local-%' AND archived=0 "
                "ORDER BY COALESCE(last_activity_at,started_at) DESC LIMIT 1",
                owner_params).fetchone()
        if latest:
            row['last_session'] = summary(authority.db.get_session(latest[0]))
    profiles = [row]
    seen = {home.resolve()}

    def discovered(configured, target):
        # Discovery metadata only: never a licence to read another owner's state/configuration.
        if target.resolve() in seen:
            return
        seen.add(target.resolve())
        entry = {'name': configured, 'path': str(target), 'is_default': configured == 'default',
                 'model': '', 'provider': '', 'description': '', 'display_name': configured,
                 'skill_count': 0}
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
    registry = getattr(getattr(authority, 'runner', None), 'session_authorities', None)
    for sibling in (registry or ()):
        discovered(served_profile_name(Path(sibling.profile_id)), Path(sibling.profile_id))
    service = getattr(authority, 'hosted_room_service', None)
    if service is not None:
        for configured, target in service.profile_homes().items():
            discovered(configured, target)
    return {'profiles': profiles, 'bot_mode_protocol': True}
