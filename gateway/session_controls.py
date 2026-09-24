"""Authenticated WS projection of the gateway authority; no TUI execution fallback."""
from dataclasses import asdict
from pathlib import Path
import sqlite3
import uuid

from gateway.config import Platform
from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL, Principal, SessionRef, Submission
from hermes_state_runtime import RuntimeStoreError


# Verbs whose param sets are closed and never carried a ``profile`` key.
_PROFILE_IMPLICIT = frozenset({'session.create', 'session.list', 'session.info', 'session.resume', 'session.detach',
                               'prompt.submit', 'prompt.receipt', 'prompt.cancel', 'prompt.resolve_unknown',
                               'session.interrupt', 'session.events.since', 'session.mutate',
                               'approval.respond', 'clarify.respond'})


class AuthorityConnection:
    def __init__(self, authority, transport, identity, *, operator=False):
        self.authority = authority
        self.transport = transport
        capabilities = frozenset({'session:read', 'session:submit', 'session:control',
                                  'session:approve', 'session:respond'})
        if identity:
            capabilities |= {'session:create'}
        if 'capabilities' in identity:
            capabilities = frozenset(identity['capabilities'])
        # Only verified transport gates may issue this capability, never identity data.
        capabilities -= {'session:operator'}
        if operator is True:
            capabilities |= {'session:operator'}
        if (not identity.get('user_id')
                or identity.get('instance_id', authority.instance_id) != authority.instance_id):
            capabilities = frozenset()
        from gateway.session_identity import authenticated_subject
        self.actor = Principal(authenticated_subject(identity),
                               identity.get('profile_id', authority.profile_id),
                               capabilities, uuid.uuid4().hex)
        from gateway.session_local_migration import bind_native_transport
        bind_native_transport(authority, self.actor, identity)
        self.native_owner = identity.get('native_bootstrap') is True
        self.subscriptions = {}
        self._identity, self._operator, self._siblings = identity, operator, {}
        authority.events[self.actor.transport_id] = transport

    def _sibling_for(self, profile):
        """Connection bound to the sibling authority a ``profile`` param names, or None for our own
        home. One native socket serves every profile this process multiplexes (the Desktop's
        shared-primary route); a profile nobody here serves is a mismatch, never a fallback."""
        from hermes_cli.profiles import profile_matches_home
        if profile_matches_home(profile, Path(self.authority.profile_id)):
            return None
        registry = getattr(getattr(self.authority, 'runner', None), 'session_authorities', None)
        for sibling in (registry or ()):
            if sibling is not self.authority and profile_matches_home(profile, Path(sibling.profile_id)):
                connection = self._siblings.get(sibling.profile_id)
                if connection is None:
                    identity = dict(self._identity, profile_id=sibling.profile_id, instance_id=sibling.instance_id)
                    connection = AuthorityConnection(sibling, self.transport, identity, operator=self._operator)
                    self._siblings[sibling.profile_id] = connection
                return connection
        raise RuntimeStoreError('profile_mismatch')

    async def dispatch(self, request):
        rid = request.get('id')
        method = request.get('method')
        params = request.get('params') or {}
        if not isinstance(params, dict):
            # Every handler indexes params as a mapping; refuse the frame before ``.get``.
            return {'jsonrpc': '2.0', 'id': rid, 'error': {
                'code': 4001, 'message': 'invalid_params', 'data': {'reason': 'invalid_params'}}}
        profile = params.get('profile')
        if isinstance(profile, str) and profile:
            try:
                routed = self._sibling_for(profile)
            except RuntimeStoreError as exc:
                return {'jsonrpc': '2.0', 'id': rid, 'error': {
                    'code': 4001, 'message': exc.reason, 'data': {'reason': exc.reason}}}
            if routed is not None:
                return await routed.dispatch(request)
            # Our own home: the scope is implicit for the strict session verbs, which refuse
            # unknown keys; every other handler validates ``profile`` itself.
            if method in _PROFILE_IMPLICIT:
                params = {key: value for key, value in params.items() if key != 'profile'}
        ref = SessionRef(self.actor.profile_id, params.get('session_id', ''))
        handlers = {'session.create': self.create, 'ping': self.ping, 'runtime.describe': self.describe,
                    'commands.catalog': self.command_catalog, 'complete.slash': self.slash_completions,
                    'slash.exec': self.slash_exec, 'command.dispatch': self.command_dispatch,
                    'session.list': self.list_sessions, 'session.info': self.info,
                    'session.mutate': self.mutate, 'bot_relay.deliver': self.bot_deliver,
                    'bot_relay.roster.sync': self.bot_roster, 'bot_relay.outbox.drain': self.bot_outbox,
                    'bot_relay.reply': self.bot_reply, 'a2a.forward': self.a2a_forward,
                    'kanban.run': self.kanban_run,
                    'cron.submit': self.cron_submit, 'cron.status': self.cron_status,
                    'cron.cancel': self.cron_cancel, 'cron.recover': self.cron_recover,
                    'worker.register': self.worker_register, 'worker.adopt': self.worker_adopt,
                    'worker.persist': self.worker_persist,
                    'setup.status': self.setup_status, 'setup.runtime_check': self.setup_runtime_check,
                    'session.resume': self.resume, 'session.detach': self.detach,
                    'prompt.submit': self.submit,
                    'prompt.receipt': self.receipt, 'prompt.cancel': self.cancel,
                    'prompt.resolve_unknown': self.resolve_unknown,
                    'session.interrupt': self.interrupt, 'session.events.since': self.events_since,
                    'approval.respond': self.respond, 'clarify.respond': self.respond_clarify}
        from gateway.session_busy_controls import handlers as busy_handlers
        handlers.update(busy_handlers(self))
        from gateway.session_config import handlers as config_handlers
        handlers.update(config_handlers(self))
        from gateway.session_ancillary import handlers as ancillary_handlers
        handlers.update(ancillary_handlers(self))
        from gateway.session_images import attach_bytes
        from functools import partial
        handlers['image.attach_bytes'] = partial(attach_bytes, self)
        try:
            from gateway.session_group_controls import GROUP_METHODS, dispatch_group_control
            # Every handler reads config/jobs/policy for the OWNING profile: enter its home so
            # ``_load_gateway_config`` / ``build_policy(get_hermes_home())`` never snapshot the
            # launch profile's config into a served secondary's session. Under a single profile
            # the scope is that profile's own home, so behaviour is unchanged.
            from gateway.session_authorities import owner_scope
            with owner_scope(self.authority):
                if method in GROUP_METHODS or method == 'profiles.list':
                    result = await dispatch_group_control(self, method, params)
                    return {'jsonrpc': '2.0', 'id': rid, 'result': result}
                if method not in handlers:
                    # JSON-RPC's own verdict: clients key compat fallbacks on -32601, and a
                    # 4001 'invalid_params' would read as a bad argument on a method that exists.
                    return {'jsonrpc': '2.0', 'id': rid, 'error': {
                        'code': -32601, 'message': f'unknown method: {method}',
                        'data': {'reason': 'unknown_method'}}}
                result = await handlers[method](ref, params)
            return {'jsonrpc': '2.0', 'id': rid, 'result': result}
        except RuntimeStoreError as exc:
            return {'jsonrpc': '2.0', 'id': rid, 'error': {
                'code': 4001, 'message': exc.reason, 'data': {'reason': exc.reason}}}
        except sqlite3.Error:
            return {'jsonrpc': '2.0', 'id': rid, 'error': {
                'code': 5001, 'message': 'storage_unavailable', 'data': {'reason': 'storage_unavailable'}}}

    async def a2a_forward(self, ref, params):
        from gateway.session_a2a import forward
        return await forward(self, params)

    async def kanban_run(self, ref, params):
        from gateway.session_kanban import run_task
        return await run_task(self, params)

    async def cron_recover(self, ref, params):
        from gateway.session_cron import rpc
        return await rpc(self, 'recover', params)

    async def cron_submit(self, ref, params):
        from gateway.session_cron import rpc
        return await rpc(self, 'submit', params)

    async def cron_status(self, ref, params):
        from gateway.session_cron import rpc
        return await rpc(self, 'status', params)

    async def cron_cancel(self, ref, params):
        from gateway.session_cron import rpc
        return await rpc(self, 'cancel', params)

    async def bot_roster(self, ref, params):
        from gateway.session_bot import relay_operation
        return relay_operation(self, 'roster', params)

    async def bot_outbox(self, ref, params):
        from gateway.session_bot import relay_operation
        return relay_operation(self, 'outbox', params)

    async def bot_reply(self, ref, params):
        from gateway.session_bot import relay_operation
        return relay_operation(self, 'reply', params)

    async def bot_deliver(self, ref, params):
        from gateway.session_bot import deliver
        return await deliver(self, params)

    async def worker_register(self, ref, params):
        from gateway.session_worker import worker_request
        return await worker_request(self, ref, params, operation='register')

    async def worker_adopt(self, ref, params):
        from gateway.session_worker import worker_request
        return await worker_request(self, ref, params, operation='adopt')

    async def worker_persist(self, ref, params):
        from gateway.session_worker import worker_request
        return await worker_request(self, ref, params, operation='persist')

    async def slash_exec(self, ref, params):
        from gateway.session_commands import execute_command
        return await execute_command(self, ref, params)

    async def command_dispatch(self, ref, params):
        from gateway.session_commands import execute_command
        return await execute_command(self, ref, params, dispatch=True)

    async def command_catalog(self, ref, params):
        from gateway.session_discovery import discover_commands
        return await discover_commands(self.authority, self.actor, params)

    async def slash_completions(self, ref, params):
        from gateway.session_discovery import discover_commands
        return await discover_commands(self.authority, self.actor, params, completion=True)

    async def setup_status(self, ref, params):
        from gateway.session_readiness import readiness
        return await readiness(self.authority, self.actor, params, runtime=False)

    async def setup_runtime_check(self, ref, params):
        from gateway.session_readiness import readiness
        return await readiness(self.authority, self.actor, params, runtime=True)

    async def create(self, ref, params):
        from gateway.session_local import create_local_session
        from gateway.session_local_title import resolve_titled_session, title_new_session, validate_title
        # ``-c <title> --create-if-missing``: resolve-or-create is one owner step (no await
        # between lookup and creation), so concurrent programmatic callers converge.
        title = validate_title(params.pop('title')) if 'title' in params else None
        # Bot Mode's forever-chat mint: born hidden from the sidebar; `follow_profile_config` is
        # the canonical default (a local session's runtime is built from the profile's current
        # config on every attach), so the flag is accepted for the legacy contract and implied.
        hidden = params.pop('hidden', False)
        params.pop('follow_profile_config', None)
        if type(hidden) is not bool:
            raise RuntimeStoreError('invalid_params')
        ref = title and resolve_titled_session(self.authority, self.actor, title, missing_ok=True)
        if not ref:
            ref = create_local_session(self.authority, self.actor, params)
            if title:
                title_new_session(self.authority, ref, title)
            if hidden:
                self.authority.db.set_session_hidden(ref.session_id, True)
        return await self.resume(ref, {})

    async def ping(self, ref, params):
        if not self.actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        if params:
            raise RuntimeStoreError('invalid_params')
        return {'pong': True}

    async def describe(self, ref, params):
        await self.ping(ref, params)
        from gateway.session_policy import CREATE_FIELDS
        return {'instance_id': self.authority.instance_id, 'profile_id': self.authority.profile_id,
                'authority_epoch': self.authority.epoch,
                'capabilities': ['durable-admission-v1', 'event-replay-v1', 'local-cli-create-v1', 'acp-editor-policy-v1', 'acp-session-mcp-v1'],
                'session_create': {'sources': ['cli', 'tui', 'gui', 'acp'],
                                   'parameters': sorted(CREATE_FIELDS | {'title'})}}

    async def info(self, ref, params):
        from gateway.session_local import local_session_info
        if set(params) != {'session_id'}:
            raise RuntimeStoreError('invalid_params')
        await self.authority.resolve(self.actor, ref)
        return local_session_info(self.authority, ref)

    async def list_sessions(self, ref, params):
        if 'session:read' not in self.actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        limit = params.get('limit', 200)
        if set(params) - {'limit', 'title', 'include_hidden'} or type(limit) is not int or not 1 <= limit <= 200:
            raise RuntimeStoreError('invalid_params')
        if 'title' in params:
            # Title-as-identity lookup (the Bots roster's forever-chat registry): the same
            # resolver `session.resume title=` uses, so a hidden canonical chat answers here
            # instead of being re-minted. Never confirms another principal's titles.
            from gateway.session_local_title import resolve_titled_session
            found = resolve_titled_session(self.authority, self.actor, params['title'], missing_ok=True)
            if found is None:
                return {'sessions': [], 'scope': 'stored'}
            row = self.authority.db.get_session(found.session_id) or {}
            root = self.authority.db.get_session(
                self.authority.db.resolve_session_by_title(params['title']) or found.session_id) or row
            return {'sessions': [{'session_id': found.session_id, 'id': root.get('id') or found.session_id,
                                  'resolved_id': found.session_id, 'title': row.get('title') or '',
                                  'root_title': root.get('title') or '', 'source': row.get('source'),
                                  'started_at': row.get('started_at'),
                                  'message_count': row.get('message_count', 0),
                                  'running': found.session_id in self.authority.sessions}],
                    'scope': 'stored'}
        sessions = []
        for sid in tuple(self.authority.sessions):
            candidate = SessionRef(self.actor.profile_id, sid)
            try:
                handle = await self.authority.resolve(self.actor, candidate)
            except RuntimeStoreError as exc:
                if exc.reason not in {'permission_denied', 'profile_mismatch', 'not_found'}:
                    raise
                continue
            row = self.authority.db.get_session(sid)
            sessions.append({'session_id': sid, 'id': sid, 'title': row.get('title') or '',
                             'source': row.get('source'), 'started_at': row.get('started_at'),
                             'message_count': row.get('message_count', 0),
                             'running': handle.execution_state == 'running'})
        sessions.sort(key=lambda row: row['started_at'] or 0, reverse=True)
        return {'sessions': sessions[:limit], 'scope': 'live'}

    async def resume(self, ref, params):
        from gateway.session_local import local_session_info
        if 'title' in params:
            from gateway.session_local_title import resolve_titled_session
            ref = resolve_titled_session(self.authority, self.actor, params['title'])
        if 'title' not in params:
            from gateway.session_local_migration import resolve_local_target
            row = self.authority.db.get_session(ref.session_id)
            if row and row['source'] in {'cli', 'tui', 'gui'}:
                ref = resolve_local_target(self.authority, self.actor,
                    self.authority.db.get_compression_tip(ref.session_id) or ref.session_id)
        if params.get('editor') is not None:
            self.authority.authorize(self.actor, ref, 'session:control')
            from gateway.session_local_mcp import resume_editor_mcp
            resume_editor_mcp(self.authority, ref, params['editor'])
        # A resume by id/title is the user asking for THIS conversation to continue; a stamped
        # tui_shutdown / ws_disconnect row would otherwise be routed as stale on the next submit.
        self.authority.authorize(self.actor, ref, 'session:read')
        from gateway.session_local_recovery import reopen_local_session
        reopen_local_session(self.authority, ref)
        snapshot = await self.authority.attach(self.actor, ref)
        self.subscriptions[ref.session_id] = snapshot.subscription_id
        # Only local routes have a frozen local launch policy to project.
        info = {'desktop_protocol': CANONICAL_GATEWAY_PROTOCOL}
        source = self.authority.sessions[snapshot.handle.ref.session_id].source
        if source is not None and source.platform == Platform.LOCAL:
            info = local_session_info(self.authority, snapshot.handle.ref)
        return {'session_id': ref.session_id, 'stored_session_id': ref.session_id,
                'messages': list(snapshot.history), 'message_count': len(snapshot.history),
                'running': snapshot.handle.execution_state == 'running',
                'authority_epoch': snapshot.handle.authority_epoch,
                'replay_epoch': snapshot.replay_epoch, 'last_sequence': snapshot.last_sequence,
                'subscription_id': snapshot.subscription_id, 'revision': snapshot.handle.revision,
                'execution_generation': snapshot.handle.execution_generation,
                'pending': [asdict(r) for r in snapshot.pending],
                'prompts': list(snapshot.prompts), 'info': info}

    async def detach(self, ref, params):
        if set(params) != {'session_id', 'subscription_id'}:
            raise RuntimeStoreError('invalid_params')
        session_id = params['session_id']
        subscription_id = params['subscription_id']
        if (not isinstance(session_id, str) or not session_id
                or not isinstance(subscription_id, str) or not subscription_id):
            raise RuntimeStoreError('invalid_params')
        result = {'session_id': session_id, 'subscription_id': subscription_id,
                  'detached': False}
        if self.subscriptions.get(session_id) != subscription_id:
            return result
        await self.authority.detach(self.actor, subscription_id)
        if self.subscriptions.get(session_id) == subscription_id:
            del self.subscriptions[session_id]
        result['detached'] = True
        return result

    async def events_since(self, ref, params):
        self.authority.authorize(self.actor, ref, 'session:read')
        sequence = params.get('last_sequence', params.get('last_seen', 0))
        epoch = params.get('replay_epoch')
        if type(sequence) is not int or sequence < 0 or (epoch is not None and not isinstance(epoch, str)):
            raise RuntimeStoreError('invalid_params')
        return self.authority.sessions[ref.session_id].event_stream.since(epoch, sequence)

    async def submit(self, ref, params):
        if ref.session_id not in self.subscriptions:
            raise RuntimeStoreError('permission_denied')
        forbidden = set(params) - {'session_id', 'text', 'submission_id', 'input_id', 'queued', 'attachments', 'finite',
                                   'surface', 'voice_context', 'interrupted'}
        if forbidden:
            raise RuntimeStoreError('invalid_params')
        request_id = params.get('submission_id') or params.get('input_id')
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeStoreError('invalid_params')
        from gateway.session_finite import admit_finite
        from gateway.session_surface import submit_surface_fields
        payload = {'text': params.get('text'), **admit_finite(params), **submit_surface_fields(params)}
        if 'attachments' in params:
            payload['attachments'] = params['attachments']
        receipt = await self.authority.submit(self.actor, Submission(request_id, ref, payload, 'queue'))
        return asdict(receipt)

    async def mutate(self, ref, params):
        from gateway.session_mutations import mutate_session
        result = await mutate_session(self.authority, self.actor, ref, params)
        # The branching viewer navigates straight into its new child; attach it
        # here (create parity) so the first submit is not refused as a stranger.
        child = result.get('branched_session_id') if isinstance(result, dict) else None
        if child and child not in self.subscriptions:
            await self.resume(SessionRef(ref.profile_id, child), {})
        return result

    async def receipt(self, ref, params):
        return asdict(await self.authority.receipt(self.actor, ref, params.get('admission_id')))

    async def cancel(self, ref, params):
        return asdict(await self.authority.cancel_queued(self.actor, ref, params.get('admission_id')))

    async def resolve_unknown(self, ref, params):
        if set(params) != {'session_id', 'admission_id', 'execution_generation'}:
            raise RuntimeStoreError('invalid_params')
        return asdict(await self.authority.resolve_unknown(
            self.actor, ref, params['admission_id'], params['execution_generation']))

    async def interrupt(self, ref, params):
        from gateway.session_managed_worker import interrupt_managed
        if interrupt_managed(self.authority, self.actor, ref, params.get('execution_generation')):
            return asdict(self.authority._handle(ref))
        return asdict(await self.authority.interrupt(self.actor, ref, params.get('execution_generation')))

    async def respond(self, ref, params):
        if set(params) != {"session_id", "execution_generation", "prompt_id", "choice"}:
            raise RuntimeStoreError("invalid_params")
        return await self.authority.respond(self.actor, ref, params["execution_generation"],
                                            params["prompt_id"], {"choice": params["choice"]})

    async def respond_clarify(self, ref, params):
        if set(params) != {"session_id", "execution_generation", "prompt_id", "answer"}:
            raise RuntimeStoreError("invalid_params")
        return await self.authority.respond(self.actor, ref, params["execution_generation"],
            params["prompt_id"], {"answer": params["answer"]}, kind="clarify")

    async def close(self):
        for sibling in self._siblings.values():
            await sibling.close()
        self._siblings.clear()
        # Deletion may already have evicted a subscribed session; its membership died
        # with it, and one retired ID must not leave the others (or the transport) attached.
        for session_id, subscription in list(self.subscriptions.items()):
            live = self.authority.sessions.get(session_id)
            if live is not None and subscription in live.subscribers:
                await self.authority.detach(self.actor, subscription)
        from gateway.session_local_migration import unbind_native_transport
        unbind_native_transport(self.authority, self.actor)
        self.subscriptions.clear()
        self.authority.events.pop(self.actor.transport_id, None)
