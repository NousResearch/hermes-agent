"""Authenticated WS projection of the gateway authority; no TUI execution fallback."""
import asyncio
import json
import logging
import re
from dataclasses import asdict
from functools import partial
from pathlib import Path
import sqlite3
import uuid

from gateway.config import Platform
from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL, Principal, SessionRef, Submission
from hermes_constants import profile_name_for_home
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


# A refusal names the check that fired: the client prints ``field``/``detail`` beside the reason
# code, and this log line is what a report can quote when the client's copy is not enough.
def _refusal_data(exc: RuntimeStoreError) -> dict:
    """The ``error.data`` for one ``RuntimeStoreError``: always the reason, plus the field and the
    detail when the raiser named them."""
    data = {'reason': exc.reason}
    for key in ('field', 'detail'):
        value = getattr(exc, key, None)
        if value:
            data[key] = value
    for key in ('revision', 'execution_generation'):
        if isinstance(getattr(exc, key, None), int):
            data[key] = getattr(exc, key)
    return data


def _profile_label(profile_id) -> str:
    """The profile as a log line should carry it: a named profile by name, a root profile (keyed by its
    home path) by its name, never the path itself."""
    if not profile_id:
        return 'default'
    text = str(profile_id)
    if '/' not in text and '\\' not in text:
        return text
    return profile_name_for_home(text) or 'default'


def _log_refusal(profile, method, exc: RuntimeStoreError) -> None:
    """One line per refusal, at the level its audience deserves.

    A refusal that names the check that fired is a launch diagnostic (warning). A bare reason code is
    the client's own state: ``permission_denied`` on a submit, ``not_found``, ``session_busy`` are
    routine, and a retrying client must not be able to flood the gateway log, so those stay at debug
    where they were silent before. The shape follows the gateway's own refusal log lines
    (``gateway/run_notifications.py``): what was refused, for which profile, and why."""
    line = '%s refused for profile=%s: %s%s%s'
    args = (method, _profile_label(profile), exc.reason,
            f' field={exc.field}' if getattr(exc, 'field', None) else '',
            f' detail={exc.detail}' if getattr(exc, 'detail', None) else '')
    if getattr(exc, 'field', None) or getattr(exc, 'detail', None):
        logger.warning(line, *args)
    else:
        logger.debug(line, *args)


# ``session.detach end_reason``: a short snake_case label, stored verbatim as ``sessions.end_reason``.
_END_REASON = re.compile(r'[a-z][a-z_]{0,31}')
# Verbs whose param sets are closed and never carried a ``profile`` key.
_PROFILE_IMPLICIT = frozenset({'session.create', 'session.list', 'session.info', 'session.resume', 'session.detach',
                               'prompt.submit', 'prompt.receipt', 'prompt.cancel', 'prompt.resolve_unknown',
                               'session.interrupt', 'session.events.since', 'session.mutate', 'session.stats',
                               'approval.respond', 'clarify.respond'})


def display_projection(history, profile_home):
    """Main's TUI display rows for a snapshot's stored rows: the sidecar's own
    ``_history_to_messages`` (compaction handoffs stripped, ``/steer`` unwrapped, tool rows
    labelled from the paired assistant ``tool_calls``), never a re-implementation."""
    from tui_gateway import server
    rows = [{**row, '_row_id': row['row_id']} if 'row_id' in row else row for row in history]
    return server._history_to_messages(rows, profile_home=profile_home)


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

    @property
    def host_scoped(self):
        """A host-scoped grant (the Desktop's shared-primary socket, the dashboard gate) acts for
        every profile this process serves; any other grant is bound to its own profile."""
        return self._identity.get('profile_scope') == 'host'

    def _sibling_for(self, profile):
        """Connection bound to the sibling authority a ``profile`` param names, or None for our own
        home. Only a host-scoped grant (the Desktop's shared-primary socket, the dashboard gate)
        reaches a sibling; a profile-bound ticket naming any other profile is a mismatch, exactly
        like a profile nobody here serves. Never a fallback, never a rewritten profile_id."""
        from hermes_cli.profiles import profile_matches_home
        if profile_matches_home(profile, Path(self.authority.profile_id)):
            return None
        if not self.host_scoped:
            raise RuntimeStoreError('profile_mismatch')
        registry = getattr(getattr(self.authority, 'runner', None), 'session_authorities', None)
        for sibling in (registry or ()):
            if sibling is not self.authority and profile_matches_home(profile, Path(sibling.profile_id)):
                connection = self._siblings.get(sibling.profile_id)
                # A re-served profile is a new authority under the same id: never keep routing to
                # the retired one (it refuses every admission for the rest of this socket's life).
                if connection is None or connection.authority is not sibling:
                    identity = dict(self._identity, profile_id=sibling.profile_id, instance_id=sibling.instance_id)
                    connection = AuthorityConnection(sibling, self.transport, identity, operator=self._operator)
                    self._siblings[sibling.profile_id] = connection
                return connection
        raise RuntimeStoreError('profile_mismatch')

    def handlers(self):
        """Method → handler for every verb ``dispatch`` serves outside the group controls; each
        name is declared in ``tui_gateway/contracts`` (``METHODS`` or ``CANONICAL_METHODS``)."""
        handlers = {'session.create': self.create, 'ping': self.ping, 'runtime.describe': self.describe,
                    'commands.catalog': self.command_catalog, 'complete.slash': self.slash_completions,
                    'plugins.list': self.plugins_list, 'command.observe': self.command_observe,
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
                    'approval.respond': self.respond, 'clarify.respond': self.respond_clarify,
                    'clarify.lock': self.lock_clarify}
        from gateway.session_busy_controls import handlers as busy_handlers
        handlers.update(busy_handlers(self))
        from gateway.session_config import handlers as config_handlers
        handlers.update(config_handlers(self))
        from gateway.session_tui_config import handlers as tui_config_handlers
        handlers.update(tui_config_handlers(self))
        from gateway.session_ancillary import handlers as ancillary_handlers
        handlers.update(ancillary_handlers(self))
        from gateway.session_shell import handlers as shell_handlers
        handlers.update(shell_handlers(self))
        from gateway.session_tui_ops import handlers as tui_ops_handlers
        handlers.update(tui_ops_handlers(self))
        from gateway.session_tui_side import handlers as side_handlers
        handlers.update(side_handlers(self))
        from gateway.session_images import attach_bytes
        from functools import partial
        handlers['image.attach_bytes'] = partial(attach_bytes, self)
        from gateway.session_complete_path import complete_path
        handlers['complete.path'] = partial(complete_path, self)
        from gateway.session_prune import prune
        handlers['session.prune'] = partial(prune, self)
        from gateway.session_usage import stats
        handlers['session.stats'] = partial(stats, self)
        from gateway.session_active_list import active_list
        handlers['session.active_list'] = partial(active_list, self)
        from gateway.session_viewer_automation import automation
        handlers['session.automation'] = partial(automation, self)
        from gateway.session_reactions import react
        handlers['message.react'] = partial(react, self)
        from gateway.session_control_actions import control
        handlers['session.control'] = partial(control, self)
        return handlers

    async def dispatch(self, request):
        from gateway.session_sudo_prompt import is_response_frame
        if is_response_frame(request):
            # A viewer answering one of the owner's server→client requests (the sudo prompt): a
            # response gets no reply (JSON-RPC), and only a viewer entitled to respond on that
            # session settles it.
            self._resolve_response(request)
            return None
        rid = request.get('id')
        method = request.get('method')
        params = request.get('params') or {}
        if not isinstance(params, dict):
            # Every handler indexes params as a mapping; refuse the frame before ``.get``.
            return {'jsonrpc': '2.0', 'id': rid, 'error': {
                'code': 4001, 'message': 'invalid_params', 'data': {'reason': 'invalid_params'}}}
        from gateway.drain_control import ws_method_refused_while_draining
        if self._operator is True and ws_method_refused_while_draining(getattr(self.authority, 'runner', None), method):
            # An interactive socket reaches a draining owner only in a restart's after-turn wait:
            # it may answer the parked turn and queue input, never change this owner's state.
            exc = RuntimeStoreError('runtime_draining')
            _log_refusal(getattr(self.actor, 'profile_id', None), method, exc)
            return {'jsonrpc': '2.0', 'id': rid, 'error': {
                'code': 4001, 'message': exc.reason, 'data': _refusal_data(exc)}}
        profile = params.get('profile')
        if isinstance(profile, str) and profile:
            try:
                routed = self._sibling_for(profile)
            except RuntimeStoreError as exc:
                _log_refusal(getattr(self.actor, 'profile_id', None), method, exc)
                return {'jsonrpc': '2.0', 'id': rid, 'error': {
                    'code': 4001, 'message': exc.reason, 'data': _refusal_data(exc)}}
            if routed is not None:
                return await routed.dispatch(request)
            # Our own home: the scope is implicit for the strict session verbs, which refuse
            # unknown keys; every other handler validates ``profile`` itself.
            if method in _PROFILE_IMPLICIT:
                params = {key: value for key, value in params.items() if key != 'profile'}
        ref = SessionRef(self.actor.profile_id, params.get('session_id', ''))
        from tui_gateway.contracts.registry import (
            CANONICAL_METHODS, METHODS, canonical_param_problems, check_params_accepted, check_result,
        )
        contract = CANONICAL_METHODS.get(method)
        if contract is not None and (problems := canonical_param_problems(contract, params)):
            # The declared wire contract (``tui_gateway/contracts/canonical.py``) is the closed
            # key set: an unknown or missing key never reaches the handler.
            return {'jsonrpc': '2.0', 'id': rid, 'error': {
                'code': 4001, 'message': 'invalid_params', 'data': {'reason': 'invalid_params', 'fields': problems}}}
        original_params = dict(params)
        contract = contract or METHODS.get(method)
        handlers = self.handlers()
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
                elif method not in handlers:
                    # JSON-RPC's own verdict: clients key compat fallbacks on -32601, and a
                    # 4001 'invalid_params' would read as a bad argument on a method that exists.
                    return {'jsonrpc': '2.0', 'id': rid, 'error': {
                        'code': -32601, 'message': f'unknown method: {method}',
                        'data': {'reason': 'unknown_method'}}}
                else:
                    result = await handlers[method](ref, params)
            check_params_accepted(contract, original_params)
            check_result(contract, result)
            return {'jsonrpc': '2.0', 'id': rid, 'result': result}
        except RuntimeStoreError as exc:
            _log_refusal(getattr(self.actor, 'profile_id', None), method, exc)
            return {'jsonrpc': '2.0', 'id': rid, 'error': {
                'code': 4001, 'message': exc.reason, 'data': _refusal_data(exc)}}
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

    async def command_observe(self, ref, params):
        from gateway.session_commands import observe_command
        return await observe_command(self, ref, params)

    async def command_catalog(self, ref, params):
        from gateway.session_discovery import discover_commands
        return await discover_commands(self.authority, self.actor, params)

    async def plugins_list(self, ref, params):
        from gateway.session_discovery import list_plugins
        return await list_plugins(self.authority, self.actor, params)

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
        from gateway.session_local_title import validate_title
        title = validate_title(params.pop('title')) if 'title' in params else None
        if title is None:
            return await self._create(params, None)
        # ``-c <title> --create-if-missing``: resolve -> create -> title is one owner step per
        # authority (the commits run off the loop), so concurrent programmatic callers converge.
        lock = getattr(self.authority, '_titled_create_lock', None)
        if lock is None:
            lock = self.authority._titled_create_lock = asyncio.Lock()
        async with lock:
            return await self._create(params, title)

    async def _create(self, params, title):
        from gateway.session_local import stage_local_session
        from gateway.session_local_recovery import restore_local_session
        from gateway.session_local_title import resolve_titled_session, title_new_session
        from gateway.session_runtime_workers import tracked_write
        from hermes_state_local import commit_local_session
        # Bot Mode's forever-chat mint: born hidden from the sidebar. `follow_profile_config` stays in
        # the frozen request: the policy re-reads the profile's current route every turn.
        hidden = params.pop('hidden', False)
        if type(hidden) is not bool:
            raise RuntimeStoreError('invalid_params', field='hidden', detail='must be a boolean')
        ref = title and resolve_titled_session(self.authority, self.actor, title, missing_ok=True)
        if not ref:
            skills_prompt = None
            if 'skills' in params:
                # Skill files are read off the owner loop, once per NEW session (a retried
                # request_id resumes the frozen policy and never re-renders).
                from gateway.session_local_recovery import local_identity
                from gateway.session_policy import render_launch_skills
                params.setdefault('request_id', uuid.uuid4().hex)
                sid = local_identity(self.authority.profile_id, self.actor.subject, params['request_id'])
                if self.authority.db.get_session(sid) is None:
                    skills_prompt = await asyncio.to_thread(render_launch_skills, params, sid)
            # Host-bound client values apply only over the gateway's own ticketed attach (run_api).
            ref = stage_local_session(self.authority, self.actor, params, skills_prompt=skills_prompt,
                                      client_local=self.native_owner)
            if isinstance(ref, dict):
                # The creation commit runs off the owner loop: a held writer must not freeze
                # every session behind one new chat (N50).
                receipt = ref
                await tracked_write(self.authority, partial(
                    commit_local_session, self.authority.db, epoch=self.authority.epoch, receipt=receipt))
                ref = restore_local_session(self.authority, receipt['session_id'])
            # `--accept-hooks`: record this launch's consent; ``resume`` below registers the hooks.
            if params.get('accept_hooks') is True:
                from gateway.session_policy import accept_launch_hooks, restore_policy
                from hermes_state_local import local_receipt
                policy = restore_policy(local_receipt(self.authority.db, ref.session_id)['policy'])
                await asyncio.to_thread(accept_launch_hooks, policy)
            if title:
                await tracked_write(self.authority, partial(title_new_session, self.authority, ref, title))
            if hidden:
                await tracked_write(self.authority, partial(self.authority.db.set_session_hidden, ref.session_id, True))
        return await self.resume(ref, {})

    async def ping(self, ref, params):
        if not self.actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        if params:
            raise RuntimeStoreError('invalid_params')
        return {'pong': True}

    async def describe(self, ref, params):
        await self.ping(ref, params)
        from gateway.session_policy import CREATE_FIELDS, SURFACES
        from gateway.session_skin import resolved_skin
        return {'instance_id': self.authority.instance_id, 'profile_id': self.authority.profile_id,
                'authority_epoch': self.authority.epoch,
                'capabilities': ['durable-admission-v1', 'event-replay-v1', 'local-cli-create-v1', 'acp-editor-policy-v1', 'acp-session-mcp-v1'],
                'session_create': {'sources': sorted(SURFACES),
                                   'parameters': sorted(CREATE_FIELDS | {'title'})},
                # Main's ``gateway.ready.skin``: the canonical client's own ready carries it.
                'skin': await asyncio.to_thread(resolved_skin)}

    async def info(self, ref, params):
        from gateway.session_local_inventory import snapshot_info
        if set(params) != {'session_id'}:
            raise RuntimeStoreError('invalid_params')
        await self.authority.resolve(self.actor, ref)
        return await snapshot_info(self.authority, ref)

    async def list_sessions(self, ref, params):
        if 'session:read' not in self.actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        limit = params.get('limit', 200)
        if set(params) - {'limit', 'title', 'include_hidden', 'scope'} or type(limit) is not int or not 1 <= limit <= 200:
            raise RuntimeStoreError('invalid_params')
        if params.get('scope', 'live') not in {'live', 'stored'} or ('title' in params and 'scope' in params):
            raise RuntimeStoreError('invalid_params')
        if params.get('scope') == 'stored':
            from gateway.session_listing import stored_picker_sessions
            return await asyncio.to_thread(stored_picker_sessions, self.authority, self.actor,
                limit=limit, include_hidden=params.get('include_hidden', False))
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
        from hermes_state_sessions import INTERNAL_LISTING_SOURCES
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
            # The human picker feed (TUI switcher): kanban workers, `--source tool` integrations and
            # finite one-shot runs are not conversations, as on every other picker.
            if row.get('source') in INTERNAL_LISTING_SOURCES or (
                    not params.get('include_hidden') and (row.get('hidden') or row.get('archived'))):
                continue
            from gateway.session_local_close import closed_and_unviewed
            if closed_and_unviewed(self.authority, sid, handle):
                # Closed by its last viewer and idle: main lists it as resumable (the stored
                # feed), not as a live session.
                continue
            sessions.append({'session_id': sid, 'id': sid, 'title': row.get('title') or '',
                             'source': row.get('source'), 'started_at': row.get('started_at'),
                             'message_count': row.get('message_count', 0), 'model': row.get('model'),
                             'running': handle.execution_state == 'running'})
        sessions.sort(key=lambda row: row['started_at'] or 0, reverse=True)
        return {'sessions': sessions[:limit], 'scope': 'live'}

    async def resume(self, ref, params):
        from gateway.session_local_inventory import snapshot_info
        if 'latest' in params:
            # Bare `-c` / `--resume latest`: the most recent session of that surface family the
            # actor may read (workspace first), resolved by the owner like a title.
            from gateway.session_local_title import resolve_latest_session
            ref = resolve_latest_session(self.authority, self.actor, params['latest'], params.get('workspace'))
        elif 'title' in params:
            from gateway.session_local_title import resolve_titled_session
            ref = resolve_titled_session(self.authority, self.actor, params['title'])
        else:
            from gateway.session_local_migration import resolve_local_target
            from gateway.session_policy_refresh import refresh_settled
            from hermes_state_local_migration import LEGACY_LOCAL_SOURCES
            # Another viewer's refresh mid-cleanup has committed a policy it has not published yet.
            await refresh_settled(self.authority, ref.session_id)
            row = self.authority.db.get_session(ref.session_id)
            # Canonical LOCAL compression/reset segments keep the receipt on their logical
            # owner, not on the physical transcript ID the history picker / Desktop supplies
            # after list_sessions projects the tip (#139306).
            if row and row['source'] in LEGACY_LOCAL_SOURCES | {Platform.LOCAL.value}:
                ref = resolve_local_target(self.authority, self.actor,
                    self.authority.db.get_compression_tip(ref.session_id) or ref.session_id)
            elif row:
                # A messaging compression continuation attaches to its lineage root, the session
                # its lane's turns stream to; the reply names that root.
                from gateway.session_route_recovery import messaging_owner_ref
                ref = messaging_owner_ref(self.authority, ref)
        if params.get('editor') is not None:
            self.authority.authorize(self.actor, ref, 'session:control')
            from gateway.session_local_mcp import resume_editor_mcp
            resume_editor_mcp(self.authority, ref, params['editor'])
        if 'api_key' in params:
            # A launch-only key re-supplied after a restart (`--resume <id> --api-key`).
            self.authority.authorize(self.actor, ref, 'session:control')
            from gateway.session_policy import rebind_launch_key
            rebind_launch_key(self.authority, ref, params['api_key'])
        # Read-only mount (#85303): an ended row stays ended until the first admitted turn
        # reopens it (``reopen_local_session`` in the drain), so opening a finished chat
        # never re-lights DB-derived liveness with no new activity.
        self.authority.authorize(self.actor, ref, 'session:read')
        if 'session:control' in self.actor.capabilities and ref.session_id not in self.subscriptions:
            # A client attaching (main's `hermes -c` was a new process) re-reads the config-derived settings
            # (CFG-2); a snapshot poll on this attachment does not.
            from gateway.session_policy_refresh import refresh_resumed_policy
            await refresh_resumed_policy(self.authority, ref)
        snapshot = await self.authority.attach(self.actor, ref)
        self.subscriptions[ref.session_id] = snapshot.subscription_id
        self._spawn_notice(ref)
        # Only local routes have a frozen local launch policy to project; with it, main's banner and
        # status-bar inventory (tools, skills, MCP, version, branch) for that policy.
        info = {'desktop_protocol': CANONICAL_GATEWAY_PROTOCOL}
        source = self.authority.sessions[snapshot.handle.ref.session_id].source
        if source is not None and source.platform == Platform.LOCAL:
            from gateway.session_policy import policy_for_source, refresh_profile_hooks
            await asyncio.to_thread(refresh_profile_hooks, policy_for_source(self.authority.runner, source))
            info = await snapshot_info(self.authority, snapshot.handle.ref)
        display = {}
        if params.get('display_projection'):
            display['display_messages'] = display_projection(snapshot.history, self.authority.profile_id)
        return {'session_id': ref.session_id, 'stored_session_id': ref.session_id,
                'messages': list(snapshot.history), **display, 'message_count': len(snapshot.history),
                'running': snapshot.handle.execution_state == 'running',
                'authority_epoch': snapshot.handle.authority_epoch,
                'replay_epoch': snapshot.replay_epoch, 'last_sequence': snapshot.last_sequence,
                'subscription_id': snapshot.subscription_id, 'revision': snapshot.handle.revision,
                'execution_generation': snapshot.handle.execution_generation,
                'pending': [asdict(r) for r in snapshot.pending],
                'prompts': list(snapshot.prompts), 'info': info}

    def _spawn_notice(self, ref):
        """Low-disk auto-prune question for this viewer, off the resume reply's path."""
        from gateway.run_housekeeping_disk import publish_local_auto_prune_notice
        task = asyncio.get_running_loop().create_task(publish_local_auto_prune_notice(self.authority, ref))
        tasks = self.__dict__.setdefault('_notice_tasks', set())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def detach(self, ref, params):
        if set(params) - {'end_reason'} != {'session_id', 'subscription_id'}:
            raise RuntimeStoreError('invalid_params')
        session_id = params['session_id']
        subscription_id = params['subscription_id']
        # ``end_reason``: this viewer is CLOSING the conversation (main's classic CLI stamped its row
        # ``cli_close`` / ``new_session`` / ``resumed_other`` / ``branched`` itself); the owner ends the
        # idle local row once no viewer is left. Absent, a detach is only a viewer leaving.
        end_reason = params.get('end_reason')
        if (not isinstance(session_id, str) or not session_id
                or not isinstance(subscription_id, str) or not subscription_id
                or (end_reason is not None
                    and not (isinstance(end_reason, str) and _END_REASON.fullmatch(end_reason)))):
            raise RuntimeStoreError('invalid_params')
        result = {'session_id': session_id, 'subscription_id': subscription_id,
                  'detached': False}
        if self.subscriptions.get(session_id) != subscription_id:
            return result
        await self.authority.detach(self.actor, subscription_id, end_reason=end_reason)
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
        forbidden = set(params) - {'session_id', 'text', 'submission_id', 'input_id', 'queued', 'attachments', 'finite', 'unattended',
                                   'surface', 'voice_context', 'interrupted', 'voice_turn', 'display_kind', 'title_preview',
                                   'turn_author', 'api_text', 'model_override', 'clarify_batch'}
        if forbidden:
            raise RuntimeStoreError('invalid_params')
        request_id = params.get('submission_id') or params.get('input_id')
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeStoreError('invalid_params')
        from gateway.session_finite import admit_finite
        from gateway.session_display import submit_display_fields
        from gateway.session_surface import submit_surface_fields
        from gateway.session_cli_turn import submit_cli_turn_fields
        payload = {'text': params.get('text'), **admit_finite(params, submitter=(self, ref)), **submit_surface_fields(params),
                   **submit_display_fields(params), **submit_cli_turn_fields(params)}
        if 'attachments' in params:
            payload['attachments'] = params['attachments']
        # The interactive WS is the one surface a restart's after-turn wait admits (#139041).
        receipt = await self.authority.submit(self.actor, Submission(request_id, ref, payload, 'queue'),
                                              queue_while_restarting=self._operator is True)
        return asdict(receipt)

    async def mutate(self, ref, params):
        from gateway.session_mutations import mutate_session
        from hermes_state_mutation_guards import MUTATION_GUARD_REFUSALS
        try:
            result = await mutate_session(self.authority, self.actor, ref, params)
        except MUTATION_GUARD_REFUSALS as exc:
            # A live turn lease / compression lock is the same retryable busy verdict a
            # running admission gets (REST maps it to 409), not an internal error.
            raise RuntimeStoreError('session_busy') from exc
        except RuntimeStoreError as exc:
            if exc.reason in {'revision_conflict', 'stale_generation'}:
                # The fresh fences ride on the refusal, so the client retries once without a re-read.
                row = await asyncio.to_thread(self.authority.db.get_session, ref.session_id)
                if row is not None:
                    exc.revision, exc.execution_generation = row['runtime_revision'], row['runtime_generation']
            raise
        # The branching viewer navigates straight into its new child; attach it
        # here (create parity) so the first submit is not refused as a stranger.
        child = result.get('branched_session_id') if isinstance(result, dict) else None
        if child and child not in self.subscriptions:
            await self.resume(SessionRef(ref.profile_id, child), {})
        return result

    async def receipt(self, ref, params):
        receipt = await self.authority.receipt(self.actor, ref, params.get('admission_id'))
        out = asdict(receipt)
        if params.get('include_result') is True and receipt.status == 'terminal':
            # The exact structured result the turn committed with settlement (`finish_result`):
            # a finite viewer reads its ledger (tokens, cost, model, exit reason) from here, the same
            # dict the in-process one-shot wrote from `run_conversation`.
            from gateway.session_results import admission_result
            saved = admission_result(self.authority.db, receipt.admission_id)
            if saved is not None:
                out['result'] = saved['result']
                out['usage'] = saved.get('usage') or {}
        return out

    async def cancel(self, ref, params):
        admission_id = params.get('admission_id')
        if (await self.authority.receipt(self.actor, ref, admission_id)).status != 'queued':
            return asdict(await self.authority.cancel_queued(self.actor, ref, admission_id))  # started: unaffected
        # Queued at this read, claimed by the drain before the cancel commit: Stop that exact claim (as
        # API and ACP do) instead of refusing while the withdrawn prompt runs.
        row = await self.authority.cancel_or_claimed(self.actor, ref, admission_id)
        if row is not None:
            await self.interrupt(ref, {'execution_generation': row['generation']})
        return asdict(await self.authority.receipt(self.actor, ref, admission_id))

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

    def _may_respond(self, session_id):
        """True when this viewer may answer a server request the owner opened on *session_id*."""
        try:
            self.authority.authorize(self.actor, SessionRef(self.actor.profile_id, session_id), 'session:respond')
        except RuntimeStoreError:
            return False
        return True

    def _resolve_response(self, frame):
        from gateway.session_sudo_prompt import resolve
        if not resolve(frame, session_ok=self._may_respond):
            import logging
            logging.getLogger(__name__).debug('dropping response for unknown server request id=%r', frame.get('id'))

    async def lock_clarify(self, _ref, params):
        """Per-question ``clarify.lock`` for a batch ``clarify`` the owner fanned out.

        Desktop and Ink lock one question at a time on that ``srq-`` id. The last lock resolves
        the request. Only a viewer entitled to ``session:respond`` on the pending request's own
        session may lock it, the same check a response frame gets.
        """
        from gateway.session_sudo_prompt import lock_answer
        request_id = str(params.get('request_id') or '')
        question_id = str(params.get('question_id') or '')
        if not request_id or not question_id:
            raise RuntimeStoreError('invalid_params')
        answer = params.get('answer', '')
        if answer is not None and not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False)
        try:
            remaining = lock_answer(request_id, question_id, answer, session_ok=self._may_respond)
        except ValueError:
            raise RuntimeStoreError('invalid_params') from None
        # ``expired``: the wait already ended while the card was still up.
        if remaining is None:
            return {'status': 'expired'}
        return {'status': 'ok', 'remaining': remaining}

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
