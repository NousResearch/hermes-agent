"""Admission-owned production exec; observers never own the worker lifetime."""
import asyncio
from dataclasses import asdict, replace
import json
import queue
import threading
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

from agent.managed_worker import encode_frame, read_frame
from gateway.session_worker_reservation import reserve_admission_worker
from hermes_state_runtime import RuntimeStoreError


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
    from gateway.session_policy import launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    terminal = json.loads(policy.terminal_json)
    if policy.config_secret_ref:
        for path, value in recover_config_secrets(authority, policy).items():
            if path[0] is None:
                terminal[path[1]] = value
    hydrated = replace(policy, config_json=json.dumps(policy.config(authority)),
                       terminal_json=json.dumps(terminal), credential_ref=None, config_secret_ref=None)
    live = authority.sessions[ref.session_id]
    return {'version': 1, 'home': authority.profile_id, 'scope': scope,
            'policy': asdict(hydrated), 'api_key': launch_key(authority, policy),
            'text': row['payload']['text'], 'route': live.route,
            **({'attachments_v1': row['payload']['attachments_v1']} if 'attachments_v1' in row['payload'] else {}),
            'user_id': live.source.user_id, 'chat_id': live.source.chat_id,
            'safe_mode': policy.safe_mode, 'ignore_user_config': policy.ignore_user_config}


class ManagedWorker:
    def __init__(self, process):
        self.process = process
        # Verified interpreter behind the handle (a launcher trampoline may sit between).
        self.worker = None
        self.write_lock = threading.Lock()
        self.commands = queue.Queue(maxsize=16)
        self.closed = threading.Event()
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
        try:
            self.commands.put_nowait(frame)
        except queue.Full as exc:
            raise RuntimeStoreError('worker_control_backpressure') from exc

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
    home = Path(str(authority.profile_id))
    if not is_multiplex_active() or not home.is_absolute():
        return None
    from agent.secret_scope import build_profile_secret_scope
    from tools.environments.local import build_subprocess_env
    env = build_subprocess_env(scrub_secrets=True)
    env.update({k: v for k, v in build_profile_secret_scope(home).items() if v is not None})
    env['HERMES_HOME'] = str(home)
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


async def execute_managed(authority, ref, row, policy):
    env = await asyncio.to_thread(_worker_env, authority)
    process = await asyncio.to_thread(subprocess.Popen, [sys.executable, '-m', 'agent.managed_worker'],
        cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, close_fds=True)
    worker = ManagedWorker(process)
    workers = getattr(authority, '_managed_workers', None)
    if workers is None:
        workers = authority._managed_workers = {}
    workers[ref.session_id] = worker
    accepted = None
    scope = None
    try:
        # The interpreter behind the handle introduces itself first; the owner verifies that
        # identity (alive, same birth, descends from the handle) before reserving for it.
        hello = await asyncio.to_thread(read_frame, process.stdout)
        scope = reserve_admission_worker(authority, admission_id=row['admission_id'],
                    process=process, principal_id=row['principal_id'], hello=hello)
        worker.worker = (scope['pid'], scope['birth'])
        # The child reads nothing else until the exact reservation has committed.
        await asyncio.to_thread(worker.send, _bootstrap(authority, ref, row, policy, scope))
        worker.writer.start()
        while True:
            frame = await asyncio.to_thread(read_frame, process.stdout)
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
            if (kind in {'tool.start', 'tool.complete'} and set(frame) == {'type', 'tool_call_id', 'name'}
                    and isinstance(frame['tool_call_id'], str) and isinstance(frame['name'], str)):
                authority.publish_execution(ref.session_id, row['generation'], kind,
                    {'tool_call_id': frame['tool_call_id'], 'name': frame['name']})
                continue
            if kind == 'result' and set(frame) == {'type', 'result'} and accepted is None:
                result = frame['result']
                if (not isinstance(result, dict) or set(result) - {'final_response', 'failed', 'interrupted'}
                        or not isinstance(result.get('final_response'), str)):
                    raise RuntimeStoreError('invalid_worker_result')
                accepted = result
                authority.sessions[ref.session_id].controls.snapshot(ref.session_id, None)
                await asyncio.to_thread(worker.send, {'type': 'finish'})
                continue
            if frame == {'type': 'finished'} and accepted is not None:
                code = await asyncio.to_thread(process.wait, 10)
                if code != 0:
                    raise RuntimeStoreError('managed_worker_lost')
                # Like in-process execution, settlement belongs to the drain's stream lock.
                # The worker must acknowledge its durable finish before that boundary.
                authority.pending_results[row['admission_id']] = {'result': accepted, 'usage': {}}
                return accepted['final_response']
            raise RuntimeStoreError('invalid_worker_frame')
    except (Exception, asyncio.CancelledError) as exc:
        import logging
        logging.getLogger(__name__).warning('Managed worker lost: %s',
            exc.reason if isinstance(exc, RuntimeStoreError) else type(exc).__name__)
        if scope is None:
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
        raise asyncio.CancelledError('managed_worker_unknown') from exc
    finally:
        workers.pop(ref.session_id, None)
        await asyncio.to_thread(worker.close)
