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
        with _profile_runtime_scope(home):
            if method == 'profiles.list':
                return _profiles(authority, actor, home, supplied)
            try:
                return _group(authority, actor, home, method, supplied)
            except RuntimeStoreError:
                raise
            except HostedRoomError as exc:
                raise RuntimeStoreError(getattr(exc, 'reason', None) or 'invalid_params') from exc
            except (ValueError, TypeError) as exc:
                raise RuntimeStoreError('invalid_params') from exc
    return await asyncio.to_thread(invoke)


def _group(authority, actor, home, method, params):
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


def _profiles(authority, actor, home, params):
    from hermes_cli.profiles import _profile_info, read_profile_meta
    import yaml
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
    meta = yaml.safe_load(path.read_text(encoding='utf-8')) if path.is_file() else {}
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
        if (canonical and canonical.get('user_id') == actor.subject
                and str(canonical.get('chat_id') or '').startswith('local-') and not canonical.get('archived')):
            row['canonical_session'] = summary(canonical)
        with authority.db._lock:
            latest = authority.db._conn.execute(
                "SELECT id FROM sessions WHERE user_id=? AND chat_id LIKE 'local-%' AND archived=0 "
                "ORDER BY COALESCE(last_activity_at,started_at) DESC LIMIT 1", (actor.subject,)).fetchone()
        if latest:
            row['last_session'] = summary(authority.db.get_session(latest[0]))
    profiles = [row]
    service = getattr(authority, 'hosted_room_service', None)
    if service is not None:
        for configured, target in service.profile_homes().items():
            if target != home:
                # Configured execution destinations are discovery metadata, not
                # permission to read another owner's state/configuration.
                profiles.append({'name': configured, 'path': str(target), 'is_default': False,
                                 'model': '', 'provider': '', 'description': '',
                                 'display_name': configured, 'skill_count': 0})
    return {'profiles': profiles, 'bot_mode_protocol': True}
