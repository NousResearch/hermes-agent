"""Admission-owned production exec; observers never own the worker lifetime."""
import asyncio
from contextlib import contextmanager
from collections import deque
from dataclasses import asdict, replace
import json
import os
import queue
import threading
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys
import time

from agent.managed_worker import accept_result, encode_frame, read_frame
from gateway.session_worker_reservation import reserve_admission_worker
from hermes_state_runtime import RuntimeStoreError

# The interpreter only imports psutil before it introduces itself; a child silent this long
# is wedged (loader/stdio stall), not slow.
HELLO_SECONDS = 60
# After Stop the child interrupts its agent and emits its result; the owner terminates a
# child that has not acknowledged within this window instead of waiting on the pipe forever.
STOP_ACK_SECONDS = 30
# Own session (POSIX): a worker whose owner died kills its whole group (managed_worker
# _die_with_owner); on Windows it exits itself.
WORKER_POPEN = {'start_new_session': True, 'close_fds': True}


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
    if policy.platform != 'cli' or request.get('provider') != 'custom' or not request.get('base_url'):
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
    # the admission's one-shot flags, the route's YOLO as of now (never the frozen launch flag) and
    # the committed surface (HUD / live voice / voice turn) when admission recorded one.
    request = dict(json.loads(policy.request_json), turn_v1={
        'finite': row['payload'].get('finite', False), 'unattended': row['payload'].get('unattended') is True,
        'yolo': _session_yolo(authority, live.route, policy),
        **({'surface_v1': row['payload']['surface_v1']} if row['payload'].get('surface_v1') else {}),
        **({'display_v1': row['payload']['display_v1']} if row['payload'].get('display_v1') else {})})
    from gateway.session_worker_construct import construct_inputs
    request['construct_v1'] = construct_inputs(authority, policy, live.source.chat_id)
    hydrated = replace(policy, config_json=json.dumps(_worker_config(authority, policy)), request_json=json.dumps(request),
                       terminal_json=json.dumps(terminal), credential_ref=None, config_secret_ref=None)
    return {'version': 1, 'home': authority.profile_id, 'scope': scope,
            'policy': asdict(hydrated), 'api_key': launch_key(authority, policy),
            'text': row['payload']['text'], 'route': live.route,
            **({'attachments_v1': row['payload']['attachments_v1']} if 'attachments_v1' in row['payload'] else {}),
            'user_id': live.source.user_id, 'chat_id': live.source.chat_id,
            'turn_author': row_turn_author(policy, row),
            'safe_mode': policy.safe_mode, 'ignore_user_config': policy.ignore_user_config}


def _worker_config(authority, policy):
    """The frozen session config the child binds as its only config (``bind_worker_policy``).
    ``command_allowlist`` is approval state rather than launch policy: an ``always`` grant an
    earlier worker of an ordinary session persisted, or one the operator removed by hand, comes
    from the live profile file, as it did before the child's config was frozen. Bypass sessions
    never read the profile."""
    config = policy.config(authority)
    home = Path(str(authority.profile_id))
    if policy.ignore_user_config or not home.is_absolute():
        return config
    from gateway.run import _load_gateway_config
    live = _load_gateway_config(home / 'config.yaml')
    config.pop('command_allowlist', None)
    if live.get('command_allowlist') is not None:
        config['command_allowlist'] = live['command_allowlist']
    return config


def _session_yolo(authority, route, policy):
    """The route's bypass as the in-process turn arms it on the owner, the one place a revocation
    is recorded: a ``--yolo`` launch seeded once per boundary, then the persisted ``/yolo`` copy."""
    from tools.approval import is_session_yolo_enabled
    from tools.approval_yolo import restore_gateway_yolo
    store = getattr(authority.runner, 'session_store', None)
    entry = store.lookup_by_session_key(route) if store is not None else None
    restore_gateway_yolo(route, getattr(entry, 'yolo', None), launch=bool(policy.yolo))
    return is_session_yolo_enabled(route)


@contextmanager
def worker_turn_scope(frame):
    """Child side of ``turn_v1``: bind what the in-process turn binds on the owner
    (execute_finite_admission, the route's YOLO, the committed surface), so ``chat -q``/``-z``
    never park a prompt, the session's current YOLO governs this child and a HUD / live-voice /
    voice turn reaches the model as it does in process. Frames without it bind nothing."""
    from gateway.session_finite import finite_turn_scope
    from gateway.session_display import display_turn_scope, restore_display
    from gateway.session_surface import restore_surface, surface_turn_scope
    turn = json.loads(frame['policy'].get('request_json') or '{}').get('turn_v1')
    if turn is None:
        yield
        return
    flags = {k: v for k, v in turn.items() if k not in ('surface_v1', 'display_v1')} if isinstance(turn, dict) else None
    if (flags is None or set(flags) != {'finite', 'unattended', 'yolo'}
            or any(type(v) is not bool for v in flags.values()) or (turn['unattended'] and not turn['finite'])):
        raise ValueError('invalid_managed_worker_bootstrap')
    try:
        surface = restore_surface(turn['surface_v1']) if 'surface_v1' in turn else None
        display = restore_display(turn['display_v1']) if 'display_v1' in turn else None
    except RuntimeStoreError as exc:
        raise ValueError('invalid_managed_worker_bootstrap') from exc
    if turn['yolo']:
        from tools.approval import enable_session_yolo
        enable_session_yolo(frame['route'])
    with finite_turn_scope(turn['finite'], turn['unattended']), surface_turn_scope(surface), \
            display_turn_scope(frame['scope']['execution_id'], display):
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
        # Monotonic instant of the FIRST Stop: one acknowledgment budget runs from here, never
        # renewed by the child's later output or by a repeated Stop.
        self.stopped_at = None
        self.writer = threading.Thread(target=self._write_controls, name='managed-control-writer', daemon=True)
        # One dedicated reader thread per worker (never the loop's shared default executor, which
        # parked pipe reads would exhaust). It reads one frame per demand, so the pipe keeps its
        # backpressure, and posts it here; a cancelled next_frame leaves the frame for the next.
        self.reader = None
        self._demand = threading.Semaphore(0)
        self._requested = False
        self._frames = deque()
        self._arrived = None
        self.stderr_drain = None
        # Exact secrets this worker was handed; the stderr sink scrubs them besides the patterns.
        self.secrets = []

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

    def _read_frames(self, loop):
        """Reader thread body. It owns ``stdout``: no other thread closes it under a blocked read."""
        try:
            while self._demand.acquire() and not self.closed.is_set():
                try:
                    item = read_frame(self.process.stdout)
                except (EOFError, ValueError, OSError, RecursionError) as exc:
                    item = exc  # EOF / invalid or oversized frame / closed pipe: the owner raises it
                try:
                    loop.call_soon_threadsafe(self._deliver, item)
                except RuntimeError:
                    return  # the owner loop has closed; nobody is left to read for
                if isinstance(item, Exception):
                    return
        finally:
            self.process.stdout.close()

    def _deliver(self, item):
        self._frames.append(item)
        self._arrived.set()

    def _take(self):
        item = self._frames[0]
        if isinstance(item, Exception):
            raise item  # stays queued: the reader has exited, every later read fails the same way
        self._frames.popleft()
        self._requested = False
        if not self._frames:
            self._arrived.clear()
        return item

    def drain_stderr(self, path):
        """Route the child's stderr (its tracebacks and redirected prints) into the profile's
        bounded, redacted, private log instead of discarding it."""
        from agent.memory_provider import spawn_context_thread
        from gateway.session_managed_worker_log import drain_worker_stderr
        self.stderr_drain = spawn_context_thread(drain_worker_stderr, name='managed-stderr-drain',
                                                 args=(self.process.stderr, path, self.process.pid, lambda: self.secrets))
        self.stderr_drain.start()

    def control(self, frame):
        if self.closed.is_set():
            raise RuntimeStoreError('managed_worker_lost')
        if frame == {'type': 'stop'}:
            if self.stopped_at is None:
                self.stopped_at = time.monotonic()
            self.stop.set()
        try:
            self.commands.put_nowait(frame)
        except queue.Full as exc:
            raise RuntimeStoreError('worker_control_backpressure') from exc

    def _stop_budget(self, ack):
        """Seconds left of the ``ack`` window that opened at the first Stop. Pure: only
        ``control`` stamps ``stopped_at``, so a read before any Stop (the hello) never starts
        the window a later Stop must receive in full."""
        return self.stopped_at + ack - time.monotonic()

    async def next_frame(self, timeout, ack):
        """Read one frame. A requested Stop bounds supervision to ``ack`` seconds TOTAL from the
        first Stop: a child that keeps emitting valid frames is escalated on the same deadline as a
        silent one, instead of leaving the turn started behind it."""
        if self.reader is None:
            self._arrived = asyncio.Event()
            from agent.memory_provider import spawn_context_thread
            self.reader = spawn_context_thread(self._read_frames, name='managed-frame-reader',
                                               args=(asyncio.get_running_loop(),))
            self.reader.start()
        if not self._frames and not self._requested:
            self._requested = True
            self._demand.release()
        arrived = asyncio.ensure_future(self._arrived.wait())
        stopper = asyncio.ensure_future(self.stop.wait())
        try:
            if not self._frames and not self.stop.is_set():
                await asyncio.wait({arrived, stopper}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if not self._frames and not self.stop.is_set():
                    raise RuntimeStoreError('managed_worker_hello_timeout')
            if not self._frames:
                # Only a latched Stop gets here; its window runs from the first Stop.
                remaining = self._stop_budget(ack)
                if remaining > 0:
                    await asyncio.wait({arrived}, timeout=remaining)
            if not self._frames:
                raise RuntimeStoreError('managed_worker_stopped')
            return self._take()
        finally:
            stopper.cancel()
            arrived.cancel()

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
        self._demand.release()  # a reader parked between frames exits; one mid-read ends at EOF
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
        if self.reader is None:
            self.process.stdout.close()
        else:
            self.reader.join(timeout=5)
        if self.stderr_drain is not None:
            self.stderr_drain.join(timeout=5)  # EOF once the child is gone; the thread closes it
        elif self.process.stderr is not None:
            self.process.stderr.close()


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
    rule MCP stdio children and shell hooks follow) — except for the launch profile's own
    worker, whose scope legitimately includes its frozen launch env. Single-profile gateways
    inherit the process env byte-for-byte, exactly as before."""
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
    # Strip the RAW environ before the constructor injects this turn's own session/bridge context:
    # a launch ``.env`` name (HERMES_SESSION_ID...) must not erase a value derived for this turn.
    env = build_subprocess_env(base=strip_launch_profile_env(os.environ.copy(), home), scrub_secrets=True)
    secrets = build_profile_secret_scope(home)
    if routed:
        # Same rule as served_profile_child_env: env_passthrough / first-party carve-outs must not
        # forward launch-process provider credentials that no .env or source snapshot recorded.
        _scrub_credentials(env, inherit_credentials=False)
    else:
        secrets = {**_launch_env_only_credentials(home, env), **secrets}
    env.update({k: v for k, v in secrets.items() if v is not None})
    env['HERMES_HOME'] = str(home)
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


def _launch_env_only_credentials(home, env):
    """Credentials the LAUNCH profile's owner resolves from its frozen launch env
    (``launch_secret_scope``: systemd ``Environment=`` / ``op run`` keys with no ``.env`` line) that
    the scrub removed from *env* — the same mapping the owner's own ``get_secret`` reads for this
    profile. Credential names only: non-credential names the constructor removed on purpose
    (unbound session context, venv markers) stay removed, and Hermes-internal secrets
    (``AUXILIARY_*_API_KEY``, relay auth) never reach a child. Only the launch home's worker calls
    this; a routed home's worker sees its own files only."""
    from tools.environments.local import _scrub_credentials
    from tools.environments.local_env_policy import _is_hermes_internal_secret
    from tui_gateway.launch_profile_policy import launch_secret_scope
    missing = {k: v for k, v in launch_secret_scope(home).items()
               if k not in env and not _is_hermes_internal_secret(k)}
    plain = _scrub_credentials(dict(missing), inherit_credentials=False)
    return {k: v for k, v in missing.items() if k not in plain}


def _interrupted_before_bootstrap(authority, row):
    authority.pending_results[row['admission_id']] = {
        'result': {'final_response': '', 'interrupted': True}, 'usage': {}}
    return ''


async def execute_managed(authority, ref, row, policy):
    from gateway.run_turn_progress import publish_worker_tool_event
    try:
        env = await asyncio.to_thread(_worker_env, authority)
        cwd = (await asyncio.to_thread(Path(__file__).resolve)).parents[1]
        from gateway.session_worker_spawn import acquire_process
        from gateway.session_managed_worker_log import worker_log_path
        log_path = worker_log_path(authority.profile_id)
        process = await acquire_process(subprocess.Popen, [sys.executable, '-m', 'agent.managed_worker'],
            cwd=cwd, stdin=subprocess.PIPE, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if log_path is None else subprocess.PIPE, **WORKER_POPEN)
    except asyncio.CancelledError:
        # acquire_process owns late-child cleanup; no bootstrap or reservation was sent.
        return _interrupted_before_bootstrap(authority, row)
    worker = ManagedWorker(process)
    if log_path is not None:
        worker.drain_stderr(log_path)
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
        worker.secrets.extend(v for v in (frame['api_key'], scope['secret']) if isinstance(v, str) and v)
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
            if isinstance(exc, asyncio.CancelledError) or (
                    isinstance(exc, RuntimeStoreError) and exc.reason == 'managed_worker_stopped'):
                # Stopped before the child ever received its bootstrap: nothing executed, so
                # this settles like an ordinary interrupted turn (the finally kills the child).
                return _interrupted_before_bootstrap(authority, row)
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
