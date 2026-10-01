"""Browser threading + transport runtime for Hermes under Pyodide.

Pyodide runs CPython on a single wasm thread with no OS threads. This module
provides the execution model upstream code expects, faithfully emulated:

- threading.Thread      -> eager execution; daemon loops suspend on blocking
                           primitives (SleepBreak) and are re-entered by the
                           scheduler when their deadline passes
- threading.Event/Condition/Lock/Timer, queue.Queue,
  concurrent.futures.ThreadPoolExecutor -> pump-based waits: while blocked,
  the interpreter drains inbound transport frames pulled from the page via
  a synchronous coincident proxy call (no event loop needed) and runs due
  scheduler callbacks
- time.sleep            -> pump for the duration, keeping transport live

Installed via install() before tui_gateway (and its deps) are imported.
"""

from __future__ import annotations

import heapq
import itertools
import sys
import threading as _real_threading
import time as _real_time
import types
from collections import deque


class SleepBreak(BaseException):
    """Raised by blocking primitives inside daemon-thread contexts.

    Propagates out of the daemon's target; the scheduler re-invokes the
    target after the requested delay (daemon targets are restartable loops:
    `while ...: body; sleep(n)` -> each re-entry runs the next iteration).
    """

    def __init__(self, delay: float):
        self.delay = delay


# ---------------------------------------------------------------- scheduler


class _Scheduler:
    def __init__(self):
        self.heap: list[tuple[float, int, object]] = []
        self.counter = itertools.count()
        self.inbound = deque()  # frames decoded from the transport ring
        self.frame_router = None  # callable(frame) -> None, set by transport glue

    def push(self, delay: float, fn) -> None:
        heapq.heappush(self.heap, (_real_time.monotonic() + delay, next(self.counter), fn))

    def due(self) -> list:
        now = _real_time.monotonic()
        out = []
        while self.heap and self.heap[0][0] <= now:
            _, _, fn = heapq.heappop(self.heap)
            out.append(fn)
        return out

    def next_deadline(self):
        return self.heap[0][0] if self.heap else None

    def due_count(self) -> int:
        now = _real_time.monotonic()
        return sum(1 for item in self.heap if item[0] <= now)


_sched = _Scheduler()
_in_daemon = _real_threading.local()
_orig_sleep = _real_time.sleep  # captured before install() patches time.sleep
_pump_ticks = 0
_last_dump_t = 0.0


def _js_bridge():
    try:
        import js  # type: ignore

        return js.hermesBridge
    except Exception:
        return None


def _js_bridge_log(msg: str):
    # console.log via the bridge is unbuffered; sys.stderr is batched by
    # Pyodide and anything in the buffer is lost when the interpreter wedges.
    try:
        b = _js_bridge()
        if b is not None:
            b.log("err", f"[pump] {msg}")
            return
    except Exception:
        pass
    try:
        import sys

        sys.stderr.write(f"[pump] {msg}\n")
    except Exception:
        pass


def _dump_pump_state() -> None:
    """_TRACE_PUMP aid: pending loop tasks, last run_sync wait point,
    and every interpreter stack — the data that named each bring-up wedge."""
    import sys as _s

    loop = _loop
    _js_bridge_log(
        f"pump#{_pump_ticks} sched={len(_sched.heap)} "
        f"ready={len(loop._ready) if loop else '-'} "
        f"lsched={len(loop._scheduled) if loop else '-'}")
    if loop is not None and loop._scheduled:
        _now = loop.time()
        for _h in list(loop._scheduled)[:4]:
            _cb = getattr(_h, "_callback", None)
            _js_bridge_log(
                f"  lsched when-now={_h._when - _now:.1f}s "
                f"cancelled={_h._cancelled} cb={getattr(_cb, '__qualname__', _cb)}")
    t = _last_sync_task
    if t is not None and not t.done():
        coro = t.get_coro()
        _js_bridge_log(f"  fut_waiter: {getattr(t, '_fut_waiter', None)!r:.200}")
        chain = []
        node = coro
        for _ in range(12):
            if node is None:
                break
            fr = getattr(node, "cr_frame", None) or getattr(node, "ag_frame", None)
            if fr is not None:
                chain.append(
                    f"{fr.f_code.co_filename.rsplit('/', 1)[-1]}:"
                    f"{fr.f_lineno}:{fr.f_code.co_name}")
            else:
                chain.append(f"<{type(node).__name__}>")
            nxt = getattr(node, "cr_await", None) or getattr(node, "ag_await", None)
            if nxt is node:
                break
            node = nxt
        _js_bridge_log("TASK-AWAIT " + " <- ".join(chain))
        try:
            import asyncio as _aio
            for tt in _aio.all_tasks(loop):
                if tt.done():
                    continue
                c2 = tt.get_coro()
                fr2 = getattr(c2, "cr_frame", None)
                where2 = (f"{fr2.f_code.co_filename.rsplit('/', 1)[-1]}:"
                          f"{fr2.f_lineno}:{fr2.f_code.co_name}") if fr2 else f"<{type(c2).__name__}>"
                _js_bridge_log(
                    f"  PENDING-TASK {tt.get_name()} @{where2} "
                    f"fut={tt._fut_waiter!r:.100}"[:220])
        except Exception as _ex:
            _js_bridge_log(f"  taskscan fail {_ex!r:.80}")
    for fr in _s._current_frames().values():
        stack = []
        while fr is not None and len(stack) < 80:
            stack.append(
                f"{fr.f_code.co_filename.rsplit('/', 1)[-1]}:"
                f"{fr.f_lineno}:{fr.f_code.co_name}")
            fr = fr.f_back
        _js_bridge_log("STACK " + " <- ".join(stack))


def pump(deadline: float | None):
    """Run one pump cycle.

    Each cycle: due scheduler callbacks run, inbound transport frames are
    drained and routed, the asyncio loop steps once, then — if no work is
    due — Atomics.wait sleeps once until the doorbell or the nearest
    deadline and arrived frames are queued for the next cycle.

    Returns after that single wait cycle — callers re-check their own
    predicates (event flag, queue contents) and re-enter. An earlier
    version looped internally until `deadline`; any perpetual timer churn
    (e.g. the logging QueueListener's 50ms resched) then starved the
    caller's predicate forever — a join() whose event was already set
    never returned.
    """
    global _pump_ticks, _last_dump_t
    bridge = _js_bridge()
    while True:
        _pump_ticks += 1
        if _TRACE_PUMP and _real_time.monotonic() - _last_dump_t > 10.0:
            _last_dump_t = _real_time.monotonic()
            _dump_pump_state()
        for fn in _sched.due():
            fn()
        while _sched.inbound:
            frame = _sched.inbound.popleft()
            if _sched.frame_router is not None:
                _sched.frame_router(frame)
        _loop_step()
        now = _real_time.monotonic()
        if deadline is not None and now >= deadline:
            return
        wait_until = deadline
        nd = _sched.next_deadline()
        if nd is not None and (wait_until is None or nd < wait_until):
            wait_until = nd
        loop = _loop
        if loop is not None:
            if loop._ready:
                wait_until = now
            elif loop._scheduled:
                when = loop._scheduled[0]._when
                if wait_until is None or when < wait_until:
                    wait_until = when
        if bridge is None:
            if deadline is None:
                return
            _orig_sleep(min(0.01, max(0.0, (deadline or now) - now)))
            return  # one cycle, as with the bridged path — callers re-check
        if wait_until is not None and wait_until <= now:
            continue  # work pending; don't sleep
        ms = 0.0 if wait_until is None else max(0.0, (wait_until - now) * 1000.0)
        frames = bridge.pump(ms if wait_until is not None else -1)
        if frames is not None:
            for raw in frames.to_py() if hasattr(frames, "to_py") else frames:
                _sched.inbound.append(raw)
        return


def wait_blocking(flag, timeout: float | None):
    """Wait for `flag()` truthiness, pumping transport meanwhile."""
    deadline = None if timeout is None else _real_time.monotonic() + timeout
    while True:
        if flag():
            return True
        if deadline is not None and _real_time.monotonic() >= deadline:
            return flag()
        pump(deadline)


# ----------------------------------------------------------- synchronization


class _LockEmu:
    """Cooperative lock. No true preemption exists, so contention only occurs
    if a holder's stack suspended — which cannot happen without greenlets.
    acquire therefore always succeeds; kept honest via owner bookkeeping."""

    def __init__(self):
        self._held = False

    def acquire(self, blocking: bool = True, timeout: float = -1):
        self._held = True
        return True

    def release(self):
        self._held = False

    def locked(self):
        return self._held

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()

    # threading internals probe these
    def acquire_lock(self, blocking: bool = True, timeout: float = -1):
        return self.acquire(blocking, timeout)

    def release_lock(self):
        self.release()

    def locked_lock(self):
        return self._held

    def _at_fork_reinit(self):
        self._held = False


class _RLockEmu(_LockEmu):
    def __init__(self):
        super().__init__()
        self._count = 0

    def acquire(self, blocking: bool = True, timeout: float = -1):
        self._held = True
        self._count += 1
        return True

    def release(self):
        self._count = max(0, self._count - 1)
        if self._count == 0:
            self._held = False


class _EventEmu:
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def clear(self):
        self._set = False

    def is_set(self):
        return self._set

    isSet = is_set

    def wait(self, timeout=None):
        if self._set:
            return True
        # timeout=0 is a poll, not a block: it must answer immediately even
        # inside a daemon body — raising SleepBreak there would abandon and
        # restart the thread's whole target.
        if timeout == 0:
            return False
        if _in_daemon.value:
            raise SleepBreak(timeout if timeout is not None else 0.05)
        return wait_blocking(lambda: self._set, timeout)


class _ConditionEmu:
    def __init__(self, lock=None):
        self._lock = lock or _RLockEmu()
        self._waiters = 0
        self._notify_flag = False

    def acquire(self, *a, **k):
        return self._lock.acquire(*a, **k)

    def release(self):
        return self._lock.release()

    def wait(self, timeout=None):
        self._notify_flag = False
        self._lock.release()
        try:
            if timeout == 0:
                return self._notify_flag
            if _in_daemon.value:
                raise SleepBreak(timeout if timeout is not None else 0.05)
            return wait_blocking(lambda: self._notify_flag, timeout)
        finally:
            self._lock.acquire()

    def notify(self, n=1):
        self._notify_flag = True

    def notify_all(self):
        self._notify_flag = True

    notifyAll = notify_all

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()


class _TimerEmu:
    def __init__(self, interval, function, args=(), kwargs=None):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs or {}
        self.finished = _EventEmu()
        self._cancelled = False

    def start(self):
        def _fire():
            if not self._cancelled:
                self.function(*self.args, **self.kwargs)
            self.finished.set()

        _sched.push(self.interval, _fire)

    def cancel(self):
        self._cancelled = True
        self.finished.set()

    def join(self, timeout=None):
        wait_blocking(lambda: self.finished.is_set(), timeout)


class _ThreadEmu:
    """Cooperative thread.

    start() runs the target eagerly. If the target suspends on a blocking
    primitive (SleepBreak) while marked daemon AND suspendable, it is
    rescheduled and re-invoked after the delay — matching upstream's
    `while ...: sleep(n)` daemon shapes (upstream tests already run
    Thread.start synchronously). Non-suspending targets complete inside
    start(), identical to upstream's synchronous test stub.

    Suspend→restart is only correct for `while` idle loops: a linear body
    (agent turn, subagent turn, wake retry, hook runner, deadline guard)
    that suspends mid-flight replays completed work on every reschedule —
    a turn that suspended at ``ticker.join()`` re-sent the same prompt
    forever. So suspension is opt-IN: only threads whose name or target
    qualname marks them as idle loops get SleepBreak; every other daemon
    pump-waits (wait_blocking) inside its stack, which is semantically
    safe for ANY body shape. An unclassified loop daemon parks its
    spawner in start() — a loud boot-time stall a smoke run catches —
    where an unclassified linear daemon under the inverse policy would
    silently replay work."""

    _ids = itertools.count(1)
    # Thread-name / target-qualname substrings that mark an idle-loop
    # daemon. Grounded in the observed runtime set: QueueListener._monitor,
    # *_ensure_*_watcher._loop, *_start_*_reaper/ticker._loop,
    # PeriodicScheduler._run, HostedRoomRuntime._worker_loop,
    # _notification_poller_loop, _cleanup_thread_worker, banner._daemon.
    # A linear body must NEVER match — replay is the failure mode.
    _LOOP_HINTS = (
        "_loop",                # *_loop targets — watchers, tickers, pollers
        "serve_forever",        # http/ws server accept loops
        "._monitor",            # logging.handlers.QueueListener._monitor
        "periodicscheduler._run",
        "_cleanup_thread_worker",
        "_drain",               # queue drainers (not -drain one-shots)
        "event.wait",           # target=stop.wait park-until-signal daemons
        "._daemon.",            # e.g. banner._daemon.<locals>.<lambda> (dot-bound)
        "ticker", "watcher", "heartbeat", "keepalive", "poller",
        "drainer", "sweeper", "janitor", "housekeep",
    )

    def __init__(self, target=None, name=None, args=(), kwargs=None, daemon=None, **_):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.name = name or f"Thread-{next(_ThreadEmu._ids)}"
        self.daemon = daemon
        qual = getattr(target, "__qualname__", "") or ""
        self._suspendable = any(
            h in f"{self.name} {qual}".lower() for h in _ThreadEmu._LOOP_HINTS)
        self._finished = _EventEmu()
        self._result = None
        self._exc = None
        self._started = False

    @property
    def ident(self):
        return 1

    def is_alive(self):
        return self._started and not self._finished.is_set()

    def _run(self):
        if self._target is None:
            self._finished.set()
            return
        # Save/restore, not set/clear: one real thread hosts every emulated
        # thread, so a nested _run's finally would otherwise clobber the
        # enclosing daemon's flag — a daemon body that loses it silently
        # switches to wait_blocking and can wedge its own start() forever
        # (poller spawned inside agent build).
        prev_daemon = getattr(_in_daemon, "value", False)
        _in_daemon.value = bool(self.daemon) and self._suspendable
        tgt = self._target
        if _TRACE_THREADS:
            _js_bridge_log(
                f"thr> {self.name} run (daemon={bool(self.daemon)}) "
                f"target={getattr(tgt, '__module__', '?')}.{getattr(tgt, '__qualname__', getattr(tgt, '__name__', repr(tgt)))}")
        try:
            self._result = self._target(*self._args, **self._kwargs)
            self._finished.set()
        except SleepBreak as sb:
            if _TRACE_THREADS:
                import traceback as _tb
                _js_bridge_log(
                    f"thr~ {self.name} suspend delay={sb.delay:.2f} at "
                    + " <- ".join(
                        f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}"
                        for f in _tb.extract_tb(sb.__traceback__)[-6:]))
            _sched.push(sb.delay, self._run)
        except BaseException as e:  # noqa: BLE001 - record like threading.excepthook
            self._exc = e
            self._finished.set()
            _js_bridge_log(f"thr! {self.name} exc {e}")
        finally:
            _in_daemon.value = prev_daemon

    def start(self):
        self._started = True
        self._run()

    def run(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        # join is a completion wait — pump until the other unit finishes.
        # The daemon Event.wait path would raise SleepBreak and restart the
        # JOINING thread's body, replaying completed work (turn restart loop).
        wait_blocking(lambda: self._finished.is_set(), timeout)


# ------------------------------------------------------------- queue module


def _make_queue_shim():
    """queue.Queue over cooperative primitives."""
    import collections as _c
    import queue as _q

    class _QueueEmu:
        def __init__(self, maxsize=0):
            self._items = _c.deque()
            self._maxsize = maxsize

        def put(self, item, block=True, timeout=None):
            self._items.append(item)

        put_nowait = lambda self, item: self.put(item)  # noqa: E731

        def get(self, block=True, timeout=None):
            # Non-blocking calls answer immediately in every context — a
            # SleepBreak in a daemon body abandons and restarts the thread's
            # entire target, which must only happen on a real wait.
            if not block or timeout == 0:
                if not self._items:
                    raise _q.Empty
                return self._items.popleft()
            if _in_daemon.value:
                if self._items:
                    return self._items.popleft()
                raise SleepBreak(timeout if timeout is not None else 0.05)
            deadline = None if timeout is None else _real_time.monotonic() + timeout
            wait_blocking(lambda: bool(self._items), timeout)
            if not self._items:
                raise _q.Empty
            return self._items.popleft()

        def get_nowait(self):
            return self.get(block=False)

        def empty(self):
            return not self._items

        def qsize(self):
            return len(self._items)

        def full(self):
            return self._maxsize > 0 and len(self._items) >= self._maxsize

        def task_done(self):
            pass

        def join(self):
            pass

    return _QueueEmu


# ----------------------------------------------------- concurrent.futures


class _FutureEmu:
    def __init__(self):
        self._done = False
        self._result = None
        self._exc = None
        self._cbs = []

    def set_result(self, r):
        self._done = True
        self._result = r
        for cb in self._cbs:
            cb(self)

    def set_exception(self, e):
        self._done = True
        self._exc = e
        for cb in self._cbs:
            cb(self)

    def result(self, timeout=None):
        if not self._done:
            wait_blocking(lambda: self._done, timeout)
        if not self._done:
            raise TimeoutError()
        if self._exc is not None:
            raise self._exc
        return self._result

    def exception(self, timeout=None):
        if not self._done:
            wait_blocking(lambda: self._done, timeout)
        return self._exc

    def done(self):
        return self._done

    def cancelled(self):
        return False

    def running(self):
        return False

    def cancel(self):
        return False

    def set_running_or_notify_cancel(self):
        return True

    def add_done_callback(self, cb):
        if self._done:
            cb(self)
        else:
            self._cbs.append(cb)


class _InlineExecutor:
    """ThreadPoolExecutor that executes callables on the caller's stack.

    Semantics match upstream's synchronous test stub; pooled handlers that
    block on server-requests pump transport instead of a worker thread."""

    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *args, **kwargs):
        fut = _FutureEmu()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut

    def map(self, fn, *iterables, **_k):
        return map(fn, *iterables)

    def shutdown(self, wait=True, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.shutdown()


# ------------------------------------------------------------------ asyncio

import asyncio  # noqa: E402


class _BrowserLoop(asyncio.BaseEventLoop):
    """Single-threaded asyncio loop driven by pump().

    BaseEventLoop supplies call_soon/call_later/create_task/run_in_executor
    over _ready + _scheduled; this subclass only replaces the selector wait
    (pump handles it) and the write-to-self pipe (cooperative — the next
    pump iteration always observes _ready).
    """

    def _write_to_self(self):
        pass

    def _run_once(self):
        raise RuntimeError("browser loop is driven by browser_runtime.pump")

    def _process_events(self, event_list):
        pass

    def run_in_executor(self, executor, func, *args):
        # The inline executor runs func on the caller's stack and returns a
        # completed future; wrap_future chains it onto an asyncio.Future.
        self._check_closed()
        return asyncio.futures.wrap_future(
            _InlineExecutor().submit(func, *args), loop=self)

    def _check_default_executor(self):
        if getattr(self, "_executor", None) is None:
            self._executor = _InlineExecutor()


_loop = None
_last_sync_task = None
_last_sync_done = None
# Dev aids — flip to True locally to trace loop/pump/thread scheduling.
# Intentionally constants, not env vars: browser env has no secure way to
# scope "debug me" to a developer session.
_TRACE_LOOP = False
_TRACE_PUMP = False
_TRACE_THREADS = False


def get_loop():
    """Return (creating) the process-wide browser event loop."""
    global _loop
    if _loop is None:
        _loop = _BrowserLoop()
        asyncio.set_event_loop(_loop)
        # Code asks get_running_loop() from cooperative contexts that aren't
        # inside a Task step (WSTransport._on_loop, async utils); the loop is
        # logically always "running" under the pump.
        asyncio.events._set_running_loop(_loop)
    return _loop


def _loop_step() -> None:
    """Move due asyncio timers to ready and run all ready callbacks once."""
    loop = _loop
    if loop is None:
        return
    now = loop.time()
    scheduled = loop._scheduled
    while scheduled and not scheduled[0]._cancelled and scheduled[0]._when <= now:
        handle = heapq.heappop(scheduled)
        loop._ready.append(handle)
    # Prune cancelled timer heads (BaseEventLoop keeps them until due).
    while scheduled and scheduled[0]._cancelled:
        heapq.heappop(scheduled)
    ntodo = len(loop._ready)
    _trace = _TRACE_LOOP
    for _ in range(ntodo):
        # A handle's callback may re-enter pump()/_loop_step (nested
        # wait_blocking/run_sync) and drain _ready itself — stop rather
        # than pop an empty deque.
        if not loop._ready:
            break
        handle = loop._ready.popleft()
        if not handle._cancelled:
            if _trace:
                _cb = handle._callback
                _js_bridge_log(
                    f"looprun {getattr(_cb, '__qualname__', type(_cb).__name__)}")
            handle._run()


def run_until_complete(coro):
    """Drive a coroutine to completion under the pump. Returns its result."""
    loop = get_loop()
    task = asyncio.ensure_future(coro, loop=loop)
    while not task.done():
        pump(None)
    return task.result()


def run_sync(coro, timeout: float | None = None):
    """Reentrant-safe sync entry into the loop.

    Safe nested inside a running task's pump: the new task is scheduled on
    the loop and steps during the same pump that is servicing the caller's
    wait. Returns the coroutine's result or raises its exception.
    """
    global _last_sync_done
    box: dict = {}
    done = _EventEmu()
    _last_sync_done = done

    async def _wrap():
        try:
            box["result"] = await coro
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            done.set()

    global _last_sync_task
    _last_sync_task = asyncio.Task(_wrap(), loop=get_loop())
    if not wait_blocking(done.is_set, timeout):
        raise TimeoutError("run_sync deadline exceeded")
    if "error" in box:
        raise box["error"]
    return box.get("result")


# ------------------------------------------------------------------ install


_in_daemon.value = False
_installed = False


def install():
    global _installed
    if _installed:
        return
    _installed = True

    T = _real_threading
    T.Thread = _ThreadEmu
    T._Thread__bootstrap_inner = lambda self: None  # noqa: SLF001
    T.Timer = _TimerEmu
    T.Event = _EventEmu
    T.Condition = _ConditionEmu
    T.Lock = _LockEmu
    T.RLock = _RLockEmu
    T._allocate_lock = _LockEmu  # noqa: SLF001

    import _thread

    _thread.allocate_lock = _LockEmu
    _thread.start_new_thread = lambda fn, args=(), kwargs=None: (
        _ThreadEmu(target=fn, args=args, kwargs=kwargs).start(),
        1,
    )[1]

    import queue

    _Q = _make_queue_shim()
    queue.Queue = _Q
    # SimpleQueue is a C type in Pyodide whose blocking get() acquires a real
    # _thread lock — a hard wedge, no bytecode boundary for the pump. Every
    # queue class must route through the cooperative shim.
    queue.SimpleQueue = _Q
    queue.LifoQueue = _Q
    queue.PriorityQueue = _Q

    import concurrent.futures

    concurrent.futures.ThreadPoolExecutor = _InlineExecutor
    concurrent.futures.Future = _FutureEmu
    import concurrent.futures.thread as _cft

    _cft.ThreadPoolExecutor = _InlineExecutor

    _real_time.sleep = _sleep_pump

    # The C-accelerated Task (_CTask) parks without a wakeup when an awaited
    # async-generator-asend yields through the send chain — the pure-Python
    # _PyTask handles it correctly under Pyodide.
    import asyncio.tasks as _at

    if hasattr(_at, "_PyTask"):
        _at.Task = _at._PyTask
        asyncio.Task = _at._PyTask

    # NOTE: never wrap Task.__step/coro.send for tracing — an extra send()
    # per step resumes __await__ while the future is still pending, raising
    # 'await wasn't used with future'. Debug via heartbeat dumps instead.

    # Cooperative task nesting: a Task's coroutine can block in pump()
    # (inline threadpool work, Event/Queue waits), and that pump must be
    # able to step OTHER tasks — e.g. a REST frame arriving mid-boot is
    # routed inside the lifespan task's own __step. Stock _enter_task
    # hard-errors on nested entry ("Cannot enter into task ... while
    # another task is being executed") and Handle._run swallows it into
    # call_exception_handler — the task dies before its coroutine starts:
    # 'never awaited' GC noise, then 'task destroyed pending'. Single-
    # threaded semantics require save/restore instead of refusal.
    _at._coop_task_stack = {}
    _ct = _at._current_tasks
    _stack = _at._coop_task_stack
    if isinstance(_ct, dict):
        def _coop_enter(loop, task):
            _stack.setdefault(loop, []).append(_ct.get(loop))
            _ct[loop] = task

        def _coop_leave(loop, task):
            stack = _stack.get(loop)
            prev = stack.pop() if stack else None
            if prev is None:
                _ct.pop(loop, None)
            else:
                _ct[loop] = prev
    else:  # ContextVar form (older asyncio)
        def _coop_enter(loop, task):
            cur = dict(_ct.get() or {})
            _stack.setdefault(loop, []).append(cur.get(loop))
            cur[loop] = task
            _ct.set(cur)

        def _coop_leave(loop, task):
            stack = _stack.get(loop)
            prev = stack.pop() if stack else None
            cur = dict(_ct.get() or {})
            if prev is None:
                cur.pop(loop, None)
            else:
                cur[loop] = prev
            _ct.set(cur)

    _at._enter_task = _coop_enter
    _at._leave_task = _coop_leave

    # anyio's to_thread worker loses work items under the cooperative
    # thread model: SleepBreak re-invokes the thread target from the top,
    # so a func that suspends mid-body drops its queue item and the result
    # future pends forever. A single-threaded interpreter can only run
    # threadpool work inline — the same model upstream's own tests use
    # (Thread.start synchronous). func still gets cooperative semantics:
    # any sleep/wait inside it pumps transport while blocked.
    try:
        from anyio._backends import _asyncio as _anyio_asyncio
    except Exception:
        _anyio_asyncio = None
    if _anyio_asyncio is not None:
        _backend = _anyio_asyncio.AsyncIOBackend

        @classmethod
        async def _inline_run_sync_in_worker_thread(
            cls, func, args, abandon_on_cancel=False, limiter=None
        ):
            return func(*args)

        _backend.run_sync_in_worker_thread = _inline_run_sync_in_worker_thread


def _sleep_pump(seconds: float):
    if _in_daemon.value:
        if seconds and seconds > 0:
            raise SleepBreak(seconds)
        return  # sleep(0): a yield — restarting the body would discard work
    end = _real_time.monotonic() + seconds
    while _real_time.monotonic() < end:
        pump(end)


def deliver_inbound(frame) -> None:
    """JS bridge -> here: enqueue a decoded transport frame."""
    _sched.inbound.append(frame)


def set_frame_router(fn) -> None:
    _sched.frame_router = fn


def pump_once(ms: float = 0.0) -> int:
    """One pump cycle; returns pending work so callers can drain fully.

    The worker's JS main loop wakes on the ring doorbell — Python-side
    tasks/timers only advance when this runs, so the JS loop calls it until
    it reports quiescence."""
    pump(None if ms < 0 else _real_time.monotonic() + ms / 1000.0)
    loop = _loop
    return (len(_sched.inbound)
            + (len(loop._ready) if loop is not None else 0)
            + _sched.due_count())
