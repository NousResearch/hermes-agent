"""Admission-owned production exec; observers never own the worker lifetime."""
import asyncio
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import queue
import threading
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

from agent.managed_worker import accept_result, encode_frame, read_frame
from gateway.session_worker_reservation import reserve_admission_worker
from hermes_state_runtime import RuntimeStoreError

# The interpreter only imports psutil before it introduces itself; a child silent this long
# is wedged (loader/stdio stall), not slow.
HELLO_SECONDS = 60
# After Stop the child interrupts its agent and emits its result; the owner terminates a
# child that has not acknowledged within this window instead of waiting on the pipe forever.
STOP_ACK_SECONDS = 30


class ManagedExecutionUnknown(RuntimeError):
    """The committed unknown verdict stops the FIFO without forging failed settlement."""


def managed_policy(authority, ref):
    """Bypass (safe / config-only) sessions always execute out of process; other local
    sessions only under the explicit custom-provider opt-in.

    The frozen creation snapshot, not current profile config, chooses execution.
    """
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None:
        return None
    if policy.ignore_user_config or policy.kanban_json is not None:
        return policy
    if policy.config().get('gateway', {}).get('managed_workers') is not True:
        return None
    request = json.loads(policy.request_json)
    if policy.source != 'cli' or request.get('provider') != 'custom' or not request.get('base_url'):
        raise RuntimeStoreError('unsupported_managed_policy')
    return policy


def _bootstrap(authority, ref, row, policy, scope):
    from gateway.session_ingress import row_turn_author
    from gateway.session_policy import launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    terminal = json.loads(policy.terminal_json)
    if policy.config_secret_ref:
        for path, value in recover_config_secrets(authority, policy).items():
            if path[0] is None:
                terminal[path[1]] = value
    live = authority.sessions[ref.session_id]
    # This turn's facts ride the per-turn hydrated request (the bootstrap field set is closed):
    # the admission's one-shot flags and the route's YOLO as of now, never the frozen launch flag.
    request = dict(json.loads(policy.request_json), turn_v1={
        'finite': row['payload'].get('finite', False), 'unattended': row['payload'].get('unattended') is True,
        'yolo': _session_yolo(authority, live.route, policy)})
    hydrated = replace(policy, config_json=json.dumps(policy.config(authority)), request_json=json.dumps(request),
                       terminal_json=json.dumps(terminal), credential_ref=None, config_secret_ref=None)
    return {'version': 1, 'home': authority.profile_id, 'scope': scope,
            'policy': asdict(hydrated), 'api_key': launch_key(authority, policy),
            'text': row['payload']['text'], 'route': live.route,
            **({'attachments_v1': row['payload']['attachments_v1']} if 'attachments_v1' in row['payload'] else {}),
            'user_id': live.source.user_id, 'chat_id': live.source.chat_id,
            'turn_author': row_turn_author(policy, row),
            'safe_mode': policy.safe_mode, 'ignore_user_config': policy.ignore_user_config}


def _session_yolo(authority, route, policy):
    """The route's bypass as the in-process turn arms it on the owner, the one place a revocation
    is recorded: a ``--yolo`` launch seeded once per boundary, then the persisted ``/yolo`` copy."""
    from tools.approval import is_session_yolo_enabled
    from tools.approval_yolo import apply_launch_yolo, restore_gateway_yolo
    store = getattr(authority.runner, 'session_store', None)
    entry = store.lookup_by_session_key(route) if store is not None else None
    if entry is not None:
        restore_gateway_yolo(route, entry.yolo is True)
    elif policy.yolo:
        apply_launch_yolo(route)
    return is_session_yolo_enabled(route)


@contextmanager
def worker_turn_scope(frame):
    """Child side of ``turn_v1``: bind what the in-process turn binds on the owner
    (execute_finite_admission, the route's YOLO), so ``chat -q``/``-z`` never park a prompt and
    the session's current YOLO governs this child. Frames without it bind nothing."""
    from gateway.session_finite import finite_turn_scope
    turn = json.loads(frame['policy'].get('request_json') or '{}').get('turn_v1')
    if turn is None:
        yield
        return
    if (not isinstance(turn, dict) or set(turn) != {'finite', 'unattended', 'yolo'}
            or any(type(v) is not bool for v in turn.values()) or (turn['unattended'] and not turn['finite'])):
        raise ValueError('invalid_managed_worker_bootstrap')
    if turn['yolo']:
        from tools.approval import enable_session_yolo
        enable_session_yolo(frame['route'])
    with finite_turn_scope(turn['finite'], turn['unattended']):
        yield


class ManagedWorker:
    def __init__(self, process):
        self.process = process
        # Verified interpreter behind the handle (a launcher trampoline may sit between).
        self.worker = None
        self.write_lock = threading.Lock()
        self.commands = queue.Queue(maxsize=16)
        self.closed = threading.Event()
        # Latched the moment Stop is admitted, before any pipe write: the owner's read loop
        # supervises it even while the child has not yet said hello or read its bootstrap.
        self.stop = asyncio.Event()
        self.writer = threading.Thread(target=self._write_controls, name='managed-control-writer', daemon=True)

    def _write_controls(self):
        try:
            while not self.closed.is_set():
                try:
                    frame = self.commands.get(timeout=.5)
                except queue.Empty:
                    continue
                self.send(frame)
        except (OSError, ValueError):
            self.closed.set()

    def control(self, frame):
        if self.closed.is_set():
            raise RuntimeStoreError('managed_worker_lost')
        if frame == {'type': 'stop'}:
            self.stop.set()
        try:
            self.commands.put_nowait(frame)
        except queue.Full as exc:
            raise RuntimeStoreError('worker_control_backpressure') from exc

    async def next_frame(self, timeout, ack):
        """Read one frame. A requested Stop bounds the wait to ``ack`` seconds (zero before the
        child can receive controls) and, unanswered, escalates instead of leaving the turn
        started behind a silent-but-alive child."""
        reader = asyncio.ensure_future(asyncio.to_thread(read_frame, self.process.stdout))
        stopper = asyncio.ensure_future(self.stop.wait())
        try:
            done, _ = await asyncio.wait({reader, stopper}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if reader not in done and stopper in done and ack:
                done, _ = await asyncio.wait({reader}, timeout=ack)
            if reader in done:
                return reader.result()
            raise RuntimeStoreError('managed_worker_stopped' if self.stop.is_set() else 'managed_worker_hello_timeout')
        finally:
            stopper.cancel()
            reader.cancel()

    def interrupt(self):
        self.control({'type': 'stop'})

    def respond(self, kind, prompt_id, value):
        self.control({'type': kind, 'prompt_id': prompt_id, 'value': value})

    def send(self, frame):
        with self.write_lock:
            self.process.stdin.write(encode_frame(frame))
            self.process.stdin.flush()

    def _signal_worker(self, kill):
        """Signal the verified interpreter, not only the handle: a launcher that exec-chained
        or exited leaves the real worker outside the Popen's reach."""
        import psutil
        if self.worker is None or self.worker[0] == self.process.pid:
            return
        try:
            proc = psutil.Process(self.worker[0])
            if proc.create_time() == self.worker[1]:
                (proc.kill if kill else proc.terminate)()
        except psutil.Error:
            pass

    def close(self):
        self.closed.set()
        if self.process.poll() is None:
            self._signal_worker(kill=False)
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._signal_worker(kill=True)
            self.process.kill()
            self.process.wait(timeout=5)
        else:
            self._signal_worker(kill=True)
        if self.writer.ident is not None:
            self.writer.join(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()


def interrupt_managed(authority, actor, ref, generation):
    worker = getattr(authority, '_managed_workers', {}).get(ref.session_id)
    if worker is None:
        return False
    authority.authorize(actor, ref, 'session:control')
    authority.check_approval_generation(ref.session_id, generation)
    worker.control({'type': 'stop'})
    return True


def _prompt_frame(authority, ref, row, worker, frame):
    live = authority.sessions[ref.session_id]
    controls = live.controls
    kind = frame.get('type')
    if kind == 'prompt_settled' and set(frame) == {'type', 'prompt_id'}:
        prompt_id = frame['prompt_id']
        controls.remote_responders.pop(prompt_id, None)
        saved = controls.pending.pop(prompt_id, None)
        if saved:
            live.event_stream.publish(ref.session_id, {'prompt_id': prompt_id,
                'execution_generation': row['generation']}, event_type=saved[1]['kind'] + '.settled')
        return True
    if kind not in {'approval', 'clarify'}:
        return False
    if len(controls.remote_responders) >= 16:
        raise RuntimeStoreError('worker_control_backpressure')
    if kind == 'approval':
        fields = {'request_id', 'command', 'description', 'allow_session', 'allow_permanent', 'smart_denied', 'edit'}
        data = frame.get('data')
        if set(frame) != {'type', 'data'} or not isinstance(data, dict) or set(data) - fields:
            raise RuntimeStoreError('invalid_worker_frame')
        prompt_id = data.get('request_id')
    else:
        if (set(frame) != {'type', 'prompt_id', 'question', 'choices', 'multi_select'}
                or not isinstance(frame['question'], str) or not isinstance(frame['choices'], list)
                or any(not isinstance(c, str) for c in frame['choices']) or type(frame['multi_select']) is not bool):
            raise RuntimeStoreError('invalid_worker_frame')
        prompt_id = frame['prompt_id']
    if not isinstance(prompt_id, str) or not prompt_id or prompt_id in controls.pending:
        raise RuntimeStoreError('invalid_worker_frame')
    controls.remote_responders[prompt_id] = worker.respond
    if kind == 'approval':
        authority.register_approval(ref.session_id, row['generation'], live.route, data)
    else:
        entry = SimpleNamespace(clarify_id=prompt_id, question=frame['question'], choices=frame['choices'],
                                multi_select=frame['multi_select'], event=threading.Event())
        authority.register_clarify(ref.session_id, row['generation'], entry)
    return True


def _worker_env(authority):
    """Child env for the OWNING profile under multiplex: its HERMES_HOME plus its ``.env``
    secrets over a scrubbed base, never the launch profile's process environment (the same
    rule MCP stdio children and shell hooks follow). Single-profile gateways inherit the
    process env byte-for-byte, exactly as before."""
    from pathlib import Path
    from agent.secret_scope import is_multiplex_active
    from tools.environments.local import _is_routed_home
    home = Path(str(authority.profile_id))
    if not home.is_absolute():
        return None
    # Keyed on the worker's OWNING profile, not only the process-wide multiplex flag: a worker for
    # another profile must never inherit the launch environ even when that flag reads False.
    routed = _is_routed_home(home)
    if not routed and not is_multiplex_active():
        return None
    from agent.secret_scope import build_profile_secret_scope
    from tools.environments.local import _scrub_credentials, build_subprocess_env, strip_launch_profile_env
    # The scrub removes credentials, not settings: the launch profile's TERMINAL_* policy and
    # its ``.env`` settings would otherwise reach the secondary's worker (cron/kanban rule).
    env = strip_launch_profile_env(build_subprocess_env(scrub_secrets=True), home)
    if routed:
        # Same rule as served_profile_child_env: env_passthrough / first-party carve-outs must not
        # forward launch-process provider credentials that no .env or source snapshot recorded.
        _scrub_credentials(env, inherit_credentials=False)
    env.update({k: v for k, v in build_profile_secret_scope(home).items() if v is not None})
    env['HERMES_HOME'] = str(home)
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


async def execute_managed(authority, ref, row, policy):
    from gateway.run_turn_progress import publish_worker_tool_event
    env = await asyncio.to_thread(_worker_env, authority)
    cwd = (await asyncio.to_thread(Path(__file__).resolve)).parents[1]
    from gateway.session_worker_spawn import acquire_process
    process = await acquire_process(subprocess.Popen, [sys.executable, '-m', 'agent.managed_worker'],
        cwd=cwd, stdin=subprocess.PIPE, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, close_fds=True)
    worker = ManagedWorker(process)
    workers = getattr(authority, '_managed_workers', None)
    if workers is None:
        workers = authority._managed_workers = {}
    workers[ref.session_id] = worker
    # A Stop acknowledged during the env/spawn awaits found no worker and was latched for this
    # generation; consume it in the same step that makes the worker reachable to interrupt_managed.
    authority.adopt_agent(ref.session_id, row['generation'], worker)
    accepted = usage = None
    scope = None
    try:
        # The interpreter behind the handle introduces itself first; the owner verifies that
        # identity (alive, same birth, descends from the handle) before reserving for it.
        hello = await worker.next_frame(HELLO_SECONDS, ack=0)
        if worker.stop.is_set():
            # Stopped before the child could receive controls: hello raced the latch; never bootstrap.
            raise RuntimeStoreError('managed_worker_stopped')
        scope = reserve_admission_worker(authority, admission_id=row['admission_id'],
                    process=process, principal_id=row['principal_id'], hello=hello)
        worker.worker = (scope['pid'], scope['birth'])
        # The child reads nothing else until the exact reservation has committed.
        frame = await asyncio.to_thread(_bootstrap, authority, ref, row, policy, scope)
        await asyncio.to_thread(worker.send, frame)
        worker.writer.start()
        while True:
            frame = await worker.next_frame(None, ack=STOP_ACK_SECONDS)
            authority.check_approval_generation(ref.session_id, row['generation'])
            with authority.sessions[ref.session_id].event_stream.lock:
                if _prompt_frame(authority, ref, row, worker, frame):
                    continue
            if frame == {'type': 'error', 'reason': 'managed_worker_failed'}:
                raise RuntimeStoreError('managed_worker_failed')
            kind = frame.get('type')
            if kind == 'ready' and set(frame) == {'type', 'pid'} and frame['pid'] == scope['pid']:
                continue
            if kind == 'delta' and set(frame) == {'type', 'text'} and isinstance(frame['text'], str):
                authority.publish_execution(ref.session_id, row['generation'], 'message.delta', {'text': frame['text']})
                continue
            if publish_worker_tool_event(authority, ref.session_id, row['generation'], frame):
                continue
            if kind == 'result' and set(frame) == {'type', 'result'} and accepted is None:
                try:
                    accepted, usage = accept_result(frame['result'])
                except ValueError as exc:
                    raise RuntimeStoreError('invalid_worker_result') from exc
                authority.sessions[ref.session_id].controls.snapshot(ref.session_id, None)
                await asyncio.to_thread(worker.send, {'type': 'finish'})
                continue
            if frame == {'type': 'finished'} and accepted is not None:
                code = await asyncio.to_thread(process.wait, 10)
                if code != 0:
                    raise RuntimeStoreError('managed_worker_lost')
                # Like in-process execution, settlement belongs to the drain's stream lock.
                # The worker must acknowledge its durable finish before that boundary.
                authority.pending_results[row['admission_id']] = {'result': accepted, 'usage': usage}
                return accepted['final_response']
            raise RuntimeStoreError('invalid_worker_frame')
    except (Exception, asyncio.CancelledError) as exc:
        import logging
        logging.getLogger(__name__).warning('Managed worker lost: %s',
            exc.reason if isinstance(exc, RuntimeStoreError) else type(exc).__name__)
        if scope is None:
            if isinstance(exc, RuntimeStoreError) and exc.reason == 'managed_worker_stopped':
                # Stopped before the child ever received its bootstrap: nothing executed, so
                # this settles like an ordinary interrupted turn (the finally kills the child).
                authority.pending_results[row['admission_id']] = {
                    'result': {'final_response': '', 'interrupted': True}, 'usage': {}}
                return ''
            raise
        from gateway.session_worker_reservation import lose_admission_worker
        lose_admission_worker(authority, row, scope)
        live = authority.sessions[ref.session_id]
        with live.event_stream.lock:
            live.controls.snapshot(ref.session_id, None)
            authority._publish_pending(ref)
            live.event_stream.publish(ref.session_id, {'text': 'Worker execution is unknown.',
                'content': 'Worker execution is unknown.', 'admission_id': row['admission_id'], 'outcome': 'unknown'})
        waiter = authority.waiters.pop(row['admission_id'], None)
        if waiter is not None and not waiter.done():
            waiter.set_result('Worker execution is unknown.')
        # Stop this drain without its ordinary Exception→failed settlement. The
        # committed unknown row deliberately pauses every accepted follower.
        raise ManagedExecutionUnknown('managed_worker_unknown') from exc
    finally:
        workers.pop(ref.session_id, None)
        await asyncio.to_thread(worker.close)
