"""Ordinary daemon plus delayed diagnostics (native-runner readiness stalls only).

No behaviour change: after ``UGW_STACK_DUMP_AFTER`` seconds every OS thread's stack
(faulthandler), every asyncio task's coroutine stack, the live runtime descriptor and a
locally answered ``identify`` are appended to ``$HERMES_HOME/logs/stacks.txt`` so a stall
on a runner without SIGUSR2/strace still names the awaiting frame.
"""
import asyncio
import faulthandler
import gc
import json
import os
from pathlib import Path
import runpy
import time

path = Path(os.environ['HERMES_HOME']) / 'logs' / 'stacks.txt'
path.parent.mkdir(parents=True, exist_ok=True)
delay = float(os.environ.get('UGW_STACK_DUMP_AFTER', '60'))
_sink = open(path, "w", encoding="utf-8")  # noqa: SIM115 - must outlive the dump
faulthandler.dump_traceback_later(delay, repeat=True, file=_sink)
_real_run = asyncio.run


def _describe(sink):
    from gateway.control_socket import GatewayControlServer, windows_pipe_name
    sink.write('\n=== runtime descriptors ===\n')
    for obj in gc.get_objects():
        # vars() only: ctypes CDLL objects turn arbitrary getattr into a library load.
        try:
            attrs = object.__getattribute__(obj, '__dict__')
        except Exception:
            continue
        if not isinstance(attrs, dict):
            continue
        descriptor = attrs.get('session_runtime_descriptor')
        if isinstance(descriptor, dict):
            sink.write(f'{type(obj).__name__} id={id(descriptor)} running={attrs.get("_running")} '
                       f'draining={attrs.get("_draining")} descriptor={json.dumps(descriptor, default=str)}\n')
    sink.write(f'pipe={windows_pipe_name(Path(os.environ["HERMES_HOME"]))}\n')
    for obj in gc.get_objects():
        if type(obj) is GatewayControlServer:  # noqa: E721 - never isinstance over arbitrary gc objects
            reply = obj.handle_request_line(b'{"protocol":1,"id":1,"verb":"identify"}', 'local')
            sink.write(f'control_server home={obj._home} handlers={sorted(obj._handlers)} '
                       f'pipe_server={obj._pipe_server!r} error={getattr(obj._pipe_server, "_error", None)!r}\n'
                       f'local identify={reply.decode("utf-8", "replace")}\n')


def _run(main, **kwargs):
    async def wrapped():
        loop = asyncio.get_running_loop()

        def dump():
            loop.call_later(delay, dump)
            with open(path, 'a', encoding='utf-8') as sink:
                sink.write(f'\n=== asyncio tasks @ {time.time():.0f} ===\n')
                for task in asyncio.all_tasks(loop):
                    sink.write(f'\n--- {task.get_name()} done={task.done()} ---\n')
                    task.print_stack(file=sink)
                try:
                    _describe(sink)
                except Exception as exc:  # diagnostics only
                    sink.write(f'describe failed: {exc!r}\n')
        loop.call_later(delay + 1, dump)
        return await main
    return _real_run(wrapped(), **kwargs)


asyncio.run = _run
runpy.run_module('gateway.run', run_name='__main__')
