"""Ink's ``/bg`` (``prompt.background``) and ``/btw`` (``prompt.btw``) on the shared owner.

Each verb replies ``{task_id}`` at once and the answer lands later on the parent session's event
stream as ``background.complete`` ``{task_id, text}`` / ``btw.complete`` ``{task_id, question,
text}`` (``error: <exc>`` on failure) — the sidecar's contract (``tui_gateway/methods_prompt.py``).
The side agent is detached: never an admission, never a FIFO turn, never a row in the parent
transcript. It runs on a daemon thread
under the owning profile's scope and the session's frozen launch policy (cwd, terminal, model
route, toolsets), so a session that never ran a turn (no resident agent) works the same.
"""
from functools import partial
import contextlib
import logging
import threading
import uuid

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

# Strong references to the running side-agent threads (dropped as each one finishes).
_RUNNING = set()
_RUNTIME_KEYS = ('api_key', 'base_url', 'provider', 'requested_provider', 'api_mode', 'command', 'args',
                 'credential_pool', 'max_tokens', 'capabilities')


def handlers(connection):
    return {'prompt.background': partial(background, connection), 'prompt.btw': partial(btw, connection),
            'preview.restart': partial(preview_restart, connection)}


def _admit(connection, ref, params, prefix, required='text', optional=()):
    """Shared admission: ``(live, policy, text, task_id)`` or a refusal before any work starts."""
    text = params.get(required)
    if (set(params) - {'session_id', 'profile', required, *optional} or not isinstance(ref.session_id, str)
            or not ref.session_id or not isinstance(text, str) or not text.strip()
            or any(params.get(key) is not None and not isinstance(params[key], str) for key in optional)):
        raise RuntimeStoreError('invalid_params')
    authority = connection.authority
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:submit')
    # The answer rides this connection's event subscription, like a submit's reply.
    if ref.session_id not in connection.subscriptions:
        raise RuntimeStoreError('permission_denied')
    authority._require_admission_open()
    from gateway.session_policy import policy_for_source
    live = authority.sessions[ref.session_id]
    policy = policy_for_source(authority.runner, live.source)
    if policy is None:
        raise RuntimeStoreError('permission_denied')
    return live, policy, text, f'{prefix}_{uuid.uuid4().hex[:6]}'


@contextlib.contextmanager
def _side_scope(authority, policy, task_id):
    """The owning profile's home/secrets, the session's frozen policy, and the task's own session
    identity (approval key, HERMES_SESSION_ID) for the side agent's thread."""
    from gateway.session_authorities import owner_scope
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_policy import policy_scope
    with owner_scope(authority, hydrate_secrets=True), policy_scope(policy, authority=authority):
        tokens = set_session_vars(session_key=task_id, session_id=task_id, cwd=policy.cwd)
        try:
            yield
        finally:
            clear_session_vars(tokens)


def _resolve_runtime(authority, policy):
    """``(model, runtime)`` the session's next turn would use: its frozen policy route."""
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    return _resolve_policy_agent_runtime(authority.runner, policy)


def _side_agent(authority, policy, live, task_id, **overrides):
    """A fresh ``AIAgent`` on the session's frozen route (call inside ``_side_scope``)."""
    from agent.skill_utils import parse_config_string_list
    from gateway.run import _checkpoint_agent_kwargs
    from run_agent import AIAgent
    runner = authority.runner
    model, runtime = _resolve_runtime(authority, policy)
    config = policy.config(authority)
    pr = getattr(runner, '_provider_routing', None) or {}
    refresh_fallback = getattr(runner, '_refresh_fallback_model', None)
    kwargs = dict(
        model=model, **{key: runtime.get(key) for key in _RUNTIME_KEYS if key in runtime},
        **_checkpoint_agent_kwargs(config),
        max_iterations=policy.max_turns, quiet_mode=True, verbose_logging=False,
        enabled_toolsets=list(policy.current_toolsets(config)),
        disabled_toolsets=parse_config_string_list((config.get('agent') or {}).get('disabled_toolsets')) or None,
        reasoning_config=policy.current_reasoning_config(runner, live.route),
        service_tier=policy.current_service_tier(runner, live.route),
        request_overrides=dict(runtime.get('request_overrides') or {}),
        providers_allowed=pr.get('only'), providers_ignored=pr.get('ignore'), providers_order=pr.get('order'),
        provider_sort=pr.get('sort'), provider_require_parameters=pr.get('require_parameters', False),
        provider_data_collection=pr.get('data_collection'),
        # Its own session in the owning profile's store, like the sidecar's ``bg_*`` rows.
        session_id=task_id, platform=policy.platform, session_db=authority.db,
        fallback_model=refresh_fallback() if callable(refresh_fallback) else None,
        load_soul_identity=not policy.ignore_rules, skip_memory=policy.ignore_rules,
        side_agent=True)
    kwargs.update(overrides)
    return AIAgent(**kwargs)


def _spawn(authority, ref, task_id, event_type, body, extra=None):
    """Run ``body()`` on a daemon thread; publish its text (or ``error: <exc>``) on the session."""
    extra = extra or {}

    def run():
        try:
            text = body()
        except Exception as exc:
            logger.warning('Side agent %s failed', task_id, exc_info=True)
            text = f'error: {getattr(exc, "reason", None) or exc}'
        try:
            live = authority.sessions.get(ref.session_id)
            if live is not None:
                live.event_stream.publish(ref.session_id, {'task_id': task_id, **extra, 'text': str(text or '')},
                                          event_type=event_type)
        finally:
            _RUNNING.discard(thread)

    thread = threading.Thread(target=run, name=f'side-agent-{task_id}', daemon=True)
    _RUNNING.add(thread)
    thread.start()
    return {'task_id': task_id}


async def background(connection, ref, params):
    """``/bg <prompt>``: a fresh agent on the session's route runs the prompt to completion."""
    live, policy, text, task_id = _admit(connection, ref, params, 'bg')
    authority = connection.authority

    def body():
        from tui_gateway.methods_prompt import _final_response_text
        with _side_scope(authority, policy, task_id):
            agent = _side_agent(authority, policy, live, task_id)
            try:
                result = agent.run_conversation(user_message=text, task_id=task_id)
            finally:
                # AIAgent.close() is the owner boundary (memory, tool subprocesses, clients).
                with contextlib.suppress(Exception):
                    agent.close()
        return _final_response_text(result)

    return _spawn(authority, ref, task_id, 'background.complete', body)


async def btw(connection, ref, params):
    """``/btw <question>``: a side question over a snapshot of the conversation
    (``agent/side_question.py``); the live history, FIFO and prompt cache stay untouched."""
    _live, policy, text, task_id = _admit(connection, ref, params, 'btw')
    authority = connection.authority
    # The resident agent's conversation (a cache-parity fork replays it); a session with no
    # resident agent (never ran a turn here, or evicted) answers from its stored transcript.
    agent = authority.agent(ref)
    snapshot = list(getattr(agent, '_session_messages', None) or [])
    from gateway.session_local_recovery import transcript_target
    target = None if snapshot else transcript_target(authority, ref)

    def body():
        from agent.side_question import answer_side_question
        history = snapshot or authority.db.get_messages_as_conversation(target)
        with _side_scope(authority, policy, task_id):
            model, runtime = _resolve_runtime(authority, policy)
            main_runtime = {'model': model, 'session_id': ref.session_id,
                            **{key: runtime.get(key) for key in ('provider', 'base_url', 'api_key', 'api_mode')}}
            return answer_side_question(text, history, parent_agent=agent, main_runtime=main_runtime) or ''

    return _spawn(authority, ref, task_id, 'btw.complete', body, extra={'question': text})


async def preview_restart(connection, ref, params):
    """Desktop preview pane "restart dev server": a hidden terminal+file agent brings the preview
    URL's server back up (rules: ``tui_gateway/methods_prompt.py`` ``_PREVIEW_RESTART_RULES``).
    Detached like ``/bg``; progress rides ``preview.restart.progress``, the answer
    ``preview.restart.complete`` ``{task_id, text}``. Context is the session's stored transcript."""
    live, policy, url, task_id = _admit(connection, ref, params, 'preview', required='url',
                                        optional=('cwd', 'context'))
    authority = connection.authority
    cwd, context = (str(params.get(key) or '').strip() for key in ('cwd', 'context'))
    from gateway.session_local_recovery import transcript_target
    target = transcript_target(authority, ref)

    def progress(text):
        if (text := str(text or '').strip()) and (stream := authority.sessions.get(ref.session_id)) is not None:
            stream.event_stream.publish(ref.session_id, {'task_id': task_id, 'text': text},
                                        event_type='preview.restart.progress')

    def body():
        from tui_gateway.agent_callbacks import _preview_restart_history
        from tui_gateway.methods_prompt import (_final_response_text, _preview_restart_prompt,
                                                _validated_preview_cwd)
        preview_cwd = _validated_preview_cwd(cwd)
        history = _preview_restart_history(authority.db.get_messages_as_conversation(target) if target else [])
        prompt = _preview_restart_prompt(url, cwd, context, bool(history))
        with _side_scope(authority, policy, task_id):
            if preview_cwd:
                from tools.terminal_tool import register_task_env_overrides
                register_task_env_overrides(task_id, {'cwd': preview_cwd})
            try:
                progress('Starting hidden restart agent')
                # Deliberately NOT closed via AIAgent.close(): it would kill the background
                # server this task exists to leave running.
                agent = _side_agent(authority, policy, live, task_id, enabled_toolsets=['terminal', 'file'],
                                    disabled_toolsets=None, session_db=None, skip_memory=True,
                                    tool_start_callback=lambda _id, name, _args: progress(f'Running {name}'))
                result = agent.run_conversation(user_message=prompt, task_id=task_id,
                                                conversation_history=history or None)
            finally:
                with contextlib.suppress(Exception):
                    from tools.terminal_tool import clear_task_env_overrides
                    clear_task_env_overrides(task_id)
        return _final_response_text(result)

    return _spawn(authority, ref, task_id, 'preview.restart.complete', body)
