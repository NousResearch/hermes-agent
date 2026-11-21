"""The Inbox (``cron/inbox.py``) on the canonical gateway: list, mark-read, and continuing an item.

``inbox.list`` / ``inbox.read`` read and write the owning profile's store only, like every session
method; read state is the session's own ``last_read_at`` watermark, so a second client (the mobile app)
sees the same unread items.

Continuing: a message typed into a FINISHED inbox run is no longer refused ``cron_session_read_only``.
On that first continuation the run's frozen cron policy is re-resolved ONCE as a fresh interactive
session of the continuing client's own surface (``build_policy`` with that source: its toolsets via
``_resolve_toolsets`` / ``platform_toolsets``, its platform and so its platform hint), committed as the
session's next policy generation, and the stored system prompt and ``tools[]`` pin are cleared so the
next turn builds them for that surface. That is the one deliberate prompt-cache break; afterwards the
session is an ordinary chat of that surface and stays byte-stable. A run still executing stays read-only.

Which surfaces may continue an item is ``CONTINUE_SOURCES``, keyed by creation label. The surface itself
is derived only from facts the gateway established (``continuing_surface``, from #154913's
``branch_surface`` by @Doueen), never from request parameters, so a new client (mobile) gets its own
toolset and hint by adding its label here and in ``gateway.session_policy.SURFACES``.
"""
import asyncio
import json
from dataclasses import asdict, replace
from functools import partial

from persistence.core.errors import AuthorityRefusal

from cron.inbox import INBOX_MARKER
from gateway.session_persistence import authority_persistence
from hermes_cli.dashboard_auth.ws_tickets import INTERNAL_PROVIDER

# Creation labels whose clients may continue an inbox item (``SURFACES`` key -> continues).
CONTINUE_SOURCES = frozenset({'gui'})
LIST_LIMIT, LIST_MAX = 50, 200
# Stamped once an item was continued: its open row is then a chat, not a run still executing.
CONTINUED = 'cron_inbox_continued'


def handlers(connection):
    return {'inbox.list': partial(list_items, connection), 'inbox.read': partial(mark_read, connection)}


def continuing_surface(actor, transport):
    """The creation label of the client continuing an item, from what the gateway verified: the Desktop's
    own declaration on a native ticket (``transport.desktop_local``), else the operator capability of a
    dashboard-listener socket (the Web app and a remote Desktop run the Desktop renderer). A native
    ticket without the Desktop's declaration is the CLI or TUI: it holds the operator capability too,
    so that capability alone never makes it ``gui``."""
    if getattr(transport, 'desktop_local', False):
        return 'gui'
    identity = getattr(transport, 'auth_identity', None) or {}
    # A native ticket is the CLI/TUI here (the Desktop said ``desktop_local``); the server-internal
    # credential is the dashboard's embedded TUI child. Both hold the operator capability.
    terminal = identity.get('native_bootstrap') is True or identity.get('provider') == INTERNAL_PROVIDER
    return 'gui' if not terminal and 'session:operator' in getattr(actor, 'capabilities', ()) else 'cli'


def _item(store, session_id):
    record = store.sessions.get(session_id)
    if record is None or not isinstance(marker := (record.model_config or {}).get(INBOX_MARKER), dict):
        return None
    summary = store.sessions.summary(session_id, compact_rows=True)
    # Unread means a result not yet seen. A continued item is the user's own chat now: its turns move
    # the watermark's activity past the read time, which must not re-badge the Inbox.
    continued = bool((record.model_config or {}).get(CONTINUED))
    return {'session_id': str(session_id), 'job_id': str(marker.get('job_id') or ''),
            'job_name': str(marker.get('job_name') or marker.get('job_id') or ''),
            'delivered_at': float(marker.get('delivered_at') or 0), 'preview': str(marker.get('preview') or ''),
            'unread': bool(summary is not None and summary.unread and not continued),
            'running': record.ended_at is None and not continued}


def _list(store, limit):
    # The id scan is one query returning ids, newest run first; reading an item costs two more, so only
    # the newest LIST_MAX are read. The unread badge counts that window (an unread run older than the
    # newest LIST_MAX is not counted); the page is its newest *limit* by delivery time.
    ids = store.sessions.session_ids_with_model_config_key(INBOX_MARKER)[:LIST_MAX]
    items = [item for sid in ids if (item := _item(store, sid)) is not None]
    items.sort(key=lambda item: item['delivered_at'], reverse=True)
    return {'items': items[:limit], 'unread_count': sum(item['unread'] for item in items)}


async def list_items(connection, ref, params):
    if 'session:read' not in connection.actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    limit = params.get('limit', LIST_LIMIT)
    if type(limit) is not int or not 1 <= limit <= LIST_MAX:
        raise AuthorityRefusal('invalid_params', field='limit', detail=f'must be an integer 1..{LIST_MAX}')
    return await asyncio.to_thread(_list, authority_persistence(connection.authority), limit)


def _mark(store, session_id, read):
    if _item(store, session_id) is None:
        raise AuthorityRefusal('not_found')
    store.sessions.set_read(session_id, read)
    return {'session_id': session_id, 'unread': not read}


async def mark_read(connection, ref, params):
    if 'session:control' not in connection.actor.capabilities:
        raise AuthorityRefusal('permission_denied')
    read = params.get('read', True)
    if not isinstance(params.get('session_id'), str) or type(read) is not bool:
        raise AuthorityRefusal('invalid_params')
    return await asyncio.to_thread(_mark, authority_persistence(connection.authority), params['session_id'], read)


def _continuable(store, session_id):
    """``(stored policy, generation)`` of a finished inbox run still on its cron policy, else None."""
    record = store.sessions.get(session_id)
    if record is None or record.ended_at is None or not isinstance((record.model_config or {}).get(INBOX_MARKER), dict):
        return None
    stored = store.bindings.receipt(session_id).policy
    if stored.get('source') != 'cron':
        return None
    return stored, record.runtime_generation


def _interactive_policy(authority, session_id, stored, surface, desktop_local):
    """A fresh chat's launch policy for *surface*, in the run's folder (``stage_local_session``'s rules)."""
    from gateway.run import _resolve_gateway_model
    from gateway.session_local import load_profile_config, profile_env_scope
    from gateway.session_policy import LOCAL_DESKTOP_CHAT, bind_launch_key, build_policy
    home = authority_persistence(authority).scope.profile_dir
    with profile_env_scope(home):
        config = load_profile_config(home)
    private = {}
    policy = build_policy({'source': surface, 'cwd': stored['cwd']}, config, private_secrets=private)
    if policy.model is None:
        policy = replace(policy, model=_resolve_gateway_model(policy.config()) or None)
    if desktop_local and surface == 'gui':
        policy = replace(policy, config_json=json.dumps({**json.loads(policy.config_json), LOCAL_DESKTOP_CHAT: True}))
    return bind_launch_key(authority, session_id, policy, None, config_secrets=private)


def _clear_prompt_snapshot(store, session_id):
    """The cron surface's stored prompt and tools[] pin: the next turn builds both for the new surface."""
    target = store.bindings.receipt(session_id).current_target
    store.sessions.update_system_prompt(target, None)
    store.sessions.update_tool_pin(target, None)
    store.sessions.patch_model_config(session_id, {CONTINUED: True})


async def continue_if_inbox(connection, ref):
    """Before a submit: re-resolve a finished inbox run as the continuing client's chat, once."""
    authority = connection.authority
    store = authority_persistence(authority)
    found = await asyncio.to_thread(_continuable, store, ref.session_id)
    if found is None:
        return
    surface = continuing_surface(connection.actor, connection.transport)
    if surface not in CONTINUE_SOURCES:
        return  # the run keeps its read-only refusal on a surface that may not continue it
    stored, generation = found
    from gateway.session_authorities import owner_scope
    from gateway.session_policy_refresh import _publish
    from gateway.session_runtime_workers import tracked_write
    with owner_scope(authority):
        policy = await asyncio.to_thread(_interactive_policy, authority, ref.session_id, stored, surface,
                                         getattr(connection.transport, 'desktop_local', False))
    await tracked_write(authority, partial(store.runtime.commit_refreshed_policy, authority.epoch,
                                           session_id=ref.session_id, old=stored, new=asdict(policy),
                                           generation=generation),
                        then=lambda _generation: _publish(authority, ref))
    await asyncio.to_thread(_clear_prompt_snapshot, store, ref.session_id)
