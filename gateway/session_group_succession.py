"""Canonical Group Chat succession controls: ``groups.succession.status``, ``.prepare``, ``.promote``,
``.keep``, ``.branch_log``, ``.learn``, ``.move``, ``.continue_anyway`` and ``.handover_all``.

``prepare`` and ``promote`` run on the computer the group should continue on (the target's own
gateway); ``keep`` runs on either of the two computers after ``continued_on_two`` (or on a new
host after a careful move, to go back); ``move`` and ``continue_anyway`` run on the host. Each
checks inside the call that the caller acts for the room's owner here: the subject recorded as
its owner on this computer, or this computer's operator (``not_owner`` otherwise). ``learn``
hands this computer a chain of later hosts; the proofs are the authority, so reading the room is
enough. ``handover_all`` (Desktop's sleep hook) moves only the rooms the caller owns. A shared
messaging chat (``messaging:shared:``) only reads the status and passes on a chain. See
``gateway/hosted_room_succession_move.py`` and ``gateway/hosted_room_succession_automatic.py``.
Errors carry their stable code and, for clients, only ``other`` or ``target``.
"""
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

TARGET_METHODS = frozenset({'groups.succession.status', 'groups.succession.prepare', 'groups.succession.promote',
                            'groups.succession.keep', 'groups.succession.branch_log', 'groups.succession.learn',
                            'groups.succession.move', 'groups.succession.move_now',
                            'groups.succession.continue_anyway',
                            'groups.succession.handover_all'})
_CLIENT_DETAIL = ('other', 'target')


def context(authority, actor):
    """The move context for a caller of this computer's default profile."""
    from gateway import hosted_room_succession_move as move
    from gateway.platforms.api_server_room_succession import continuation_minter
    from gateway.session_authorities import served_profile_name
    from gateway.session_group_peers import _api_server
    if served_profile_name(Path(authority.profile_id)) != 'default':
        raise RuntimeStoreError('default_profile_required')
    adapter = _api_server(authority)
    store = getattr(adapter, '_run_idempotency_store', None)
    if store is None or store.durable is not True:
        raise RuntimeStoreError('durable_run_storage_required')
    db_path = Path(authority.db.db_path)
    return move.MoveContext(
        db_path=db_path, runs_store=store, service=getattr(authority, 'hosted_room_service', None),
        actor_subject=getattr(actor, 'subject', None),
        operator='session:operator' in getattr(actor, 'capabilities', ()),
        mint_grants=continuation_minter(adapter, db_path, replace_same_epoch=False))


def _text(params, key):
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeStoreError('invalid_params')
    return value


_READ_METHODS = frozenset({'groups.succession.status', 'groups.succession.branch_log', 'groups.succession.learn'})
_OWNER = 'gateway.hosted.owner.v1:'
# A messaging chat other people read (messaging's transport id ``messaging:shared:<grant>``) carries
# the owner's subject, yet it never acts for the owner, nor shows the messages a split set aside.
_SHARED_CHAT = 'messaging:shared:'
_SHARED_CHAT_METHODS = frozenset({'groups.succession.status', 'groups.succession.learn'})


def _authorize_read(authority, actor, room_id):
    """Reading a group's succession state, or handing it a chain, needs what ``groups.log`` needs: this
    installation's operator, or the principal recorded as the room's owner here (its creator on the
    host; whoever consented on a backup)."""
    from gateway.hosted_room_succession import owner_subject_locked
    from gateway import hosted_rooms as rooms
    from contextlib import closing
    if 'session:operator' in getattr(actor, 'capabilities', ()):
        return
    subject = getattr(actor, 'subject', None)
    with closing(rooms._read_connection(Path(authority.db.db_path))) as conn:
        owner = owner_subject_locked(conn, room_id)
    if not subject or owner != subject:
        raise RuntimeStoreError('permission_denied')


def dispatch_target(authority, actor, method, params):
    from gateway import hosted_room_succession_handover as handover
    from gateway import hosted_room_succession_move as move
    from gateway import hosted_room_succession_return as returning
    from gateway.hosted_room_succession import SuccessionError
    from gateway.hosted_room_succession_status import status
    if str(getattr(actor, 'transport_id', '') or '').startswith(_SHARED_CHAT) and method not in _SHARED_CHAT_METHODS:
        raise RuntimeStoreError('permission_denied')
    if method in _READ_METHODS:
        _authorize_read(authority, actor, _text(params, 'room_id'))
    ctx = context(authority, actor)
    if method == 'groups.succession.handover_all':
        reason = params.get('reason')
        if reason not in {'sleep', 'stop', 'quit'}:
            raise RuntimeStoreError('invalid_params')
        return handover.handover_all(ctx, reason=reason)
    room_id = _text(params, 'room_id')
    try:
        if method == 'groups.succession.learn':
            return returning.learn(ctx, room_id, params.get('events'))
        if method == 'groups.succession.move':
            if not move.is_owner(ctx, room_id):
                raise SuccessionError("only the group's owner can move it", reason='not_owner')
            handover.request_move(ctx, room_id, _text(params, 'target_install_id'))
            return status(ctx, room_id)
        if method == 'groups.succession.move_now':
            if not move.is_owner(ctx, room_id):
                raise SuccessionError("only the group's owner can move it", reason='not_owner')
            handover.move_now(ctx, room_id)
            return status(ctx, room_id)
        if method == 'groups.succession.continue_anyway':
            from gateway import hosted_room_succession_automatic as automatic
            return automatic.continue_anyway(ctx, room_id)
        if method == 'groups.succession.status':
            return status(ctx, room_id)
        if method == 'groups.succession.prepare':
            return move.preview(ctx, room_id, _text(params, 'target_install_id'))
        if method == 'groups.succession.promote':
            if set(params) != {'room_id', 'target_install_id', 'preview_id', 'confirm'}:
                raise RuntimeStoreError('invalid_params')
            return move.continue_here(ctx, room_id, _text(params, 'target_install_id'),
                                      preview_id=params['preview_id'], confirm=params['confirm'])
        if method == 'groups.succession.keep':
            return move.keep(ctx, room_id, _text(params, 'install_id'))
        after_seq, limit = params.get('after_seq', 0), params.get('limit', 200)
        if type(after_seq) is not int or after_seq < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise RuntimeStoreError('invalid_params')
        status(ctx, room_id)  # the caller may read this computer's record of the room at all
        return returning.branch_log(ctx.db_path, room_id, _text(params, 'branch_id'), after_seq=after_seq,
                                    limit=limit)
    except SuccessionError as exc:
        detail = {key: exc.detail[key] for key in _CLIENT_DETAIL if key in exc.detail}
        raise RuntimeStoreError(exc.reason, detail) from exc


def record_consent_owner(authority, actor, method, params):
    """The caller that consented here to continue a group, or to keep a backup copy of it, becomes its
    owner on this computer."""
    from gateway.hosted_room_succession import record_owner_subject
    if not isinstance(params.get('room_id'), str):
        return
    if (method == 'groups.peer.invite' and (params.get('successor') is True or params.get('custody_only') is True)) or (
            method == 'groups.custody.allow' and params.get('successor') is True):
        record_owner_subject(authority.db.db_path, params['room_id'], getattr(actor, 'subject', None))
