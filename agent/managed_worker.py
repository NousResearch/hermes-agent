"""Private exec entry. No agent/config imports before adopted process authority.

stdin is the owner's bounded bootstrap/control pipe; a duplicate of stdout is
reserved for framed events before runtime imports redirect ordinary output.
Neither assignment secrets nor launch credentials appear in argv or logs.
"""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise

import json
import logging
import os
import re
from pathlib import Path
import sys
import threading

logger = logging.getLogger(__name__)

MAX_FRAME = 4 * 1024 * 1024
BOOTSTRAP_FIELDS = {'version', 'home', 'scope', 'policy', 'api_key', 'text', 'route', 'user_id', 'chat_id',
                    'turn_author', 'safe_mode', 'ignore_user_config'}
# The bounded run_conversation fields a turn receipt keeps (outcome, exit reason, tokens, cost,
# model): what the in-process projection commits and ``-z --usage-file`` reads.
RESULT_FIELDS = ('final_response', 'failed', 'interrupted', 'completed', 'partial', 'error', 'turn_exit_reason',
                 'api_calls', 'model', 'provider', 'prompt_tokens', 'completion_tokens', 'total_tokens',
                 'cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens',
                 'estimated_cost_usd', 'cost_status', 'cost_source', 'service_tier', 'failure_reason',
                 'session_id', 'requested_model', 'served_model', '_turn_input_tokens', '_turn_output_tokens')


def result_frame(result):
    """Worker side: the receipt fields of ``run_conversation``'s result, scalars only, text bounded."""
    return {k: (v[:4096] if isinstance(v, str) and k != 'final_response' else v) for k, v in result.items()
            if k in RESULT_FIELDS and (v is None or type(v) in (bool, int, float, str))}


def accept_result(result):
    """Owner side of the worker's result frame: ``(result, usage)`` in the in-process turn's shape
    (gateway projection: prompt/completion tokens as input/output), or ValueError."""
    if (not isinstance(result, dict) or set(result) - set(RESULT_FIELDS) or not isinstance(result.get('final_response'), str)
            or any(not (value is None or type(value) in (bool, int, float) or (isinstance(value, str) and len(value) <= 4096))
                   for key, value in result.items() if key != 'final_response')):
        raise ValueError('invalid_worker_result')
    accepted = dict(result)
    turn_in, turn_out = accepted.pop('_turn_input_tokens', None), accepted.pop('_turn_output_tokens', None)
    if any(value is not None and (type(value) is not int or value < 0) for value in (turn_in, turn_out)):
        raise ValueError('invalid_worker_result')
    incoming, outgoing = accepted.pop('prompt_tokens', None), accepted.pop('completion_tokens', None)
    if incoming is None and outgoing is None:
        return accepted, {}
    usage = {'input_tokens': incoming or 0 if turn_in is None else turn_in,
             'output_tokens': outgoing or 0 if turn_out is None else turn_out}
    usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']
    accepted.update(input_tokens=incoming or 0, output_tokens=outgoing or 0)
    return accepted, usage


def read_frame(stream):
    line = stream.readline(MAX_FRAME + 1)
    if not line:
        raise EOFError('managed_worker_pipe_closed')
    if len(line) > MAX_FRAME or not line.endswith(b'\n'):
        raise ValueError('managed_worker_frame_too_large')
    frame = json.loads(line)
    if not isinstance(frame, dict):
        raise ValueError('invalid_managed_worker_frame')
    return frame


def encode_frame(frame):
    data = json.dumps(frame, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    if len(data) > MAX_FRAME:
        raise ValueError('managed_worker_frame_too_large')
    return data


def validate_bootstrap(frame):
    if set(frame) - {'attachments_v1'} != BOOTSTRAP_FIELDS or frame['version'] != 1:
        raise ValueError('invalid_managed_worker_bootstrap')
    scope = frame['scope']
    fields = {'profile_id', 'session_id', 'execution_id', 'generation', 'pid', 'birth', 'secret', 'epoch'}
    if (not isinstance(scope, dict) or set(scope) != fields or scope['pid'] != os.getpid()
            or scope['profile_id'] != frame['home'] or not Path(frame['home']).is_absolute()
            or not isinstance(frame['policy'], dict)
            or any(not isinstance(frame[k], str) for k in ('text', 'route', 'user_id', 'chat_id'))
            or type(frame['safe_mode']) is not bool or type(frame['ignore_user_config']) is not bool
            or (frame['safe_mode'] and not frame['ignore_user_config'])
            or (frame['api_key'] is not None and not isinstance(frame['api_key'], str))):
        raise ValueError('invalid_managed_worker_bootstrap')
    from agent.turn_author import parse_turn_author
    if frame['turn_author'] is not None and (not isinstance(frame['turn_author'], dict)
                                            or parse_turn_author(frame['turn_author']) is None):
        raise ValueError('invalid_managed_worker_bootstrap')
    return frame


def bind_bypass_policy(frame):
    """Freeze the owner's bypass assignment process-wide before any config/provider/agent import.

    Ordinary assignments bind nothing; the frozen explicit config (never the profile) is what
    every config reader in this process returns afterwards.
    """
    if not frame['ignore_user_config']:
        return
    from agent.safe_worker_policy import _bind_safe_worker_policy
    _bind_safe_worker_policy(safe_mode=frame['safe_mode'], ignore_user_config=True,
                             config=json.loads(frame['policy']['config_json']))


class WorkerChannel:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()

    def send(self, kind, **payload):
        data = encode_frame({'type': kind, **payload})
        with self.lock:
            self.stream.write(data)
            self.stream.flush()


_INTERRUPTED = object()


class WorkerControls:
    def __init__(self, channel, route):
        self.channel, self.route = channel, route
        self.agent = None
        self.stopped = threading.Event()
        self.finish = threading.Event()
        self.clarifications = {}
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self.read, name='managed-controls', daemon=True)
        self.reader.start()

    def stop(self):
        self.stopped.set()
        if self.agent is not None:
            self.agent.interrupt()
        with self.lock:
            for state in self.clarifications.values():
                state['answer'] = _INTERRUPTED
                state['event'].set()

    def read(self):
        from tools.approval import resolve_gateway_approval
        try:
            while True:
                frame = read_frame(sys.stdin.buffer)
                if frame == {'type': 'stop'}:
                    self.stop()
                    continue
                if frame == {'type': 'finish'}:
                    self.finish.set()
                    return
                if (set(frame) != {'type', 'prompt_id', 'value'} or frame['type'] not in {'approval', 'clarify'}
                        or not isinstance(frame['prompt_id'], str) or not isinstance(frame['value'], str)
                        or len(frame['value']) > 16384):
                    raise ValueError('invalid_managed_control')
                if frame['type'] == 'approval':
                    if frame['value'] not in {'once', 'deny', 'session', 'always'}:
                        raise ValueError('invalid_managed_control')
                    resolve_gateway_approval(self.route, frame['value'], request_id=frame['prompt_id'])
                else:
                    with self.lock:
                        state = self.clarifications.get(frame['prompt_id'])
                        if state is not None:
                            state['answer'] = frame['value']
                            state['event'].set()
        except (EOFError, OSError, ValueError):
            self.stop()
            self.finish.set()

    def approval(self, data):
        from tools.approval import ack_gateway_approval
        fields = {'request_id', 'command', 'description', 'allow_session', 'allow_permanent', 'smart_denied', 'edit'}
        self.channel.send('approval', data={k: v for k, v in data.items() if k in fields})
        ack_gateway_approval(self.route, data['request_id'])

    def clarify(self, questions):
        """clarify_tool's batch contract: ``callback(questions) -> {answers, outcome, notice?}``.
        One owner prompt per question, stopping at the first that is interrupted or unanswered;
        an empty answer is a skip (``None``), as on the other surfaces."""
        answers = {}
        reply = {'answers': answers, 'outcome': 'submitted'}
        for entry in questions:
            answer = self._ask(entry['question'], entry['choices'], entry['multi_select'])
            if answer is _INTERRUPTED:
                reply['outcome'] = 'cancelled'
                break
            if answer is None:
                reply['outcome'] = 'timed_out'
                break
            answers[entry['qid']] = answer.strip() or None
        return reply

    def _ask(self, question, choices, multi_select):
        import uuid
        prompt_id = uuid.uuid4().hex
        state = {'event': threading.Event(), 'answer': None}
        with self.lock:
            if self.stopped.is_set() or len(self.clarifications) >= 16:
                return _INTERRUPTED
            self.clarifications[prompt_id] = state
        try:
            self.channel.send('clarify', prompt_id=prompt_id, question=question,
                              choices=list(choices or []), multi_select=bool(multi_select))
            state['event'].wait(3600)
            return state['answer']
        finally:
            with self.lock:
                self.clarifications.pop(prompt_id, None)
            self.channel.send('prompt_settled', prompt_id=prompt_id)


_OUTBOX_NAME = re.compile(r'[A-Za-z0-9_-]{1,200}')


def outbox_dir(home, execution_id):
    """execution_id is a ledger key ('admission-worker:<hex>'); ':' is not a legal Windows path
    character, so the private outbox directory is a portable spelling of the same identity."""
    # A legacy-imported admission id is arbitrary text; never let it name a path outside
    # worker-outboxes (``../`` on POSIX, ``..\\`` or drive spellings on Windows).
    name = execution_id.replace(':', '-') if isinstance(execution_id, str) else ''
    if not _OUTBOX_NAME.fullmatch(name):
        raise ValueError('invalid_execution_id')
    return Path(home) / 'worker-outboxes' / name


def discover_profile_mcp(policy):
    """The owner's MCP registry never crosses into this fresh interpreter; connect the profile's
    configured servers here, filtered to the frozen toolset selection, before tool discovery snapshots
    agent.tools. Safe mode keeps its deliberate no-MCP policy (discover_mcp_tools returns [])."""
    from tools.mcp_oauth import suppress_interactive_oauth
    from tools.mcp_tool_discovery import discover_mcp_tools
    with suppress_interactive_oauth():
        discover_mcp_tools(allowed_mcp_names=list(policy.toolsets))


def retire_agent(agent):
    """A settled admission is a turn boundary, not the end of the session: the owner's
    in-process agent keeps its background processes, sandbox and browser between turns
    (release_clients), so the worker must too. Only memory extraction is turn-final work."""
    # A boundary task queued this turn (LLM-bound extraction) would be cancelled by the
    # provider shutdown's short drain; give it the same bounded head start cli_shutdown does.
    manager = getattr(agent, '_memory_manager', None)
    if manager is not None:
        manager.flush_pending(timeout=10)
    messages = getattr(agent, '_session_messages', None)
    agent.shutdown_memory_provider(messages if isinstance(messages, list) else None)
    agent.release_clients()


def tool_frame(call_id, name, args, *result):
    """The callback arguments an in-process turn publishes from (executor-redacted display args,
    the result and its failure verdict); the owner builds the shared tool event from them."""
    frame = {'tool_call_id': str(call_id or ''), 'name': str(name or 'tool'), 'args': args if isinstance(args, dict) else {}}
    if result:
        from agent.display import _detect_tool_failure
        frame['result'] = result[0] if isinstance(result[0], str) else str(result[0])
        frame['is_error'] = bool(_detect_tool_failure(frame['name'], result[0])[0])
    from gateway.session_tool_events import bounded_args, bounded_result
    frame['args'] = bounded_args(frame['args'])
    if result:
        frame['result'] = bounded_result(result[0], frame['is_error'])
    return json.loads(json.dumps(frame, default=str))


def execute(frame, channel):
    # The owner RPC below imports gateway/config modules (hermes_cli.config, providers,
    # hermes_cli.plugins) transitively; the policy must already be frozen when they load.
    bind_bypass_policy(frame)
    from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC
    scope = dict(frame['scope'])
    rpc = WorkerRPC(frame['home'])
    adopted = rpc('worker.adopt', **{k: v for k, v in scope.items() if k != 'epoch'})
    if adopted['owner_epoch'] != scope['epoch']:
        raise RuntimeError('stale_epoch')
    store = RuntimeSessionStore(rpc, scope, outbox_dir(frame['home'], scope['execution_id']))
    from gateway.session_kanban import bind_worker_context
    bind_worker_context(frame)
    # Background processes a previous admission of this session started outlive that worker
    # (F24); adopt them so process_manage in this turn can poll and kill them.
    from tools.process_registry import process_registry
    process_registry.recover_from_checkpoint()
    # Store construction binds the delegation ledger before tool discovery.
    from gateway.session_policy import restore_policy, policy_scope
    policy = restore_policy(frame['policy'])
    discover_profile_mcp(policy)
    from run_agent import AIAgent
    from tools.approval import register_gateway_notify, unregister_gateway_notify
    from tools.approval_context import set_current_session_key
    set_current_session_key(frame['route'])
    os.environ['HERMES_GATEWAY_SESSION'] = '1'
    controls = WorkerControls(channel, frame['route'])
    register_gateway_notify(frame['route'], controls.approval)
    agent = None
    try:
        with policy_scope(policy):
            agent = _construct_agent(frame, AIAgent, model=policy.model, provider=policy.provider, base_url=policy.base_url,
                api_key=frame['api_key'], session_db=store, session_id=scope['session_id'],
                enabled_toolsets=list(policy.toolsets), max_iterations=policy.max_turns,
                reasoning_config=policy.reasoning_config, platform=policy.source,
                gateway_session_key=frame['route'], user_id=frame['user_id'], chat_id=frame['chat_id'],
                skip_context_files=policy.ignore_rules, load_soul_identity=not policy.ignore_rules,
                skip_memory=policy.ignore_rules, skip_background_review=True, quiet_mode=True,
                stream_delta_callback=lambda text: channel.send('delta', text=text) if text else None,
                clarify_callback=controls.clarify,
                tool_start_callback=lambda call_id, name, args: channel.send('tool.start', **tool_frame(call_id, name, args)),
                tool_complete_callback=lambda call_id, name, args, result: channel.send('tool.complete',
                                                                                         **tool_frame(call_id, name, args, result)))
            controls.agent = agent
            if controls.stopped.is_set():
                agent.interrupt()
            channel.send('ready', pid=os.getpid())
            history = store.get_messages_as_conversation(scope['session_id'])
            from gateway.session_kanban import run_worker_turns
            if 'attachments_v1' in frame:
                from gateway.session_ingress_media import restore_attachments
                from agent.image_routing import build_native_content_parts
                media = restore_attachments(frame)
                content, skipped = build_native_content_parts(frame['text'], media['media_urls'])
                if skipped:
                    raise ValueError('managed_attachment_unavailable')
                frame = {**frame, 'text': content}
            before = (getattr(agent, 'session_prompt_tokens', 0) or 0,
                      getattr(agent, 'session_completion_tokens', 0) or 0)
            result = run_worker_turns(agent, frame, history)
            result['_turn_input_tokens'] = max(0, (getattr(agent, 'session_prompt_tokens', 0) or 0) - before[0])
            result['_turn_output_tokens'] = max(0, (getattr(agent, 'session_completion_tokens', 0) or 0) - before[1])
            retire_agent(agent)
            agent = None
            # The auto-title thread bills through this store from a daemon thread; a title landing
            # after execution.finish is a stale_generation write that fails the close-time flush.
            from agent.title_generator import wait_for_title_upgrades
            wait_for_title_upgrades()
            store.flush_token_counts()
            if result.get('final_response') is None and (result.get('interrupted') or result.get('failed')):
                result['final_response'] = ''
            channel.send('result', result=result_frame(result))
            if not controls.finish.wait(30):
                raise ValueError('managed_finish_timeout')
            store.finish()
            channel.send('finished')
    finally:
        unregister_gateway_notify(frame['route'])
        if agent is not None:
            retire_agent(agent)
        store.close()


def _construct_agent(frame, factory, **kwargs):
    """Agent construction resolves the provider credentials; a Kanban attempt that fails here
    still books the one-shot exit mapping (quota wall 75, re-login 78) before the crash."""
    try:
        return factory(**kwargs)
    except Exception as exc:
        from gateway.session_kanban import record_start_failure
        record_start_failure(frame, exc)
        raise


def hello():
    """Identity the owner verifies before it reserves: this interpreter's pid and birth plus
    the ancestor chain it observes. Launchers (uv's venv python.exe) put the owner's Popen
    handle one or two hops above; the owner, never this process, decides whether they match."""
    import psutil
    me = psutil.Process()
    return {'pid': me.pid, 'birth': me.create_time(), 'ancestors': [p.pid for p in me.parents()[:3]]}


def main():
    channel = WorkerChannel(os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0))  # windows-footgun: ok — binary frames
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    # This process's own stderr only (the owner discards it): never the channel, and never the
    # profile's log files, which the module contract keeps free of assignment secrets.
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.propagate = False
    try:
        channel.send('hello', **hello())
        frame = validate_bootstrap(read_frame(sys.stdin.buffer))
        execute(frame, channel)
    except Exception:
        # Runtime exceptions can contain credentials; the owner gets no raw traceback.
        logger.exception('managed worker failed')
        channel.send('error', reason='managed_worker_failed')
        return 1
    finally:
        channel.stream.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
