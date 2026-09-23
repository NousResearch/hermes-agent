
from __future__ import annotations

import base64
import errno
import json
import os
import time
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import pytest

from plugins.platforms.a2a import protocol, security
from plugins.platforms.a2a import tools as a2a_tools
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.a2a.protocol import A2AResultValidationError, TaskStore
from gateway.config import PlatformConfig

# Helper to make a valid Task
def _valid_task(task_id="task-abc", context_id="ctx-1", state=protocol.STATE_COMPLETED, text="hello"):
    return protocol.build_task(task_id, context_id, state, text)

def _valid_message(msg_id="msg-1", context_id="ctx-1", text="hello"):
    return protocol.text_message(protocol.ROLE_AGENT, text, context_id)


import contextlib, asyncio as _aio_l, threading as _thr_l, sys, concurrent.futures as _cf

_REAL_RUN_COROUTINE_THREADSAFE = _aio_l.run_coroutine_threadsafe

@contextlib.contextmanager
def _a2a_managed_loop(primary_adapter, monkeypatch, *, timeout=5, additional_adapters=(), application_scheduler=_REAL_RUN_COROUTINE_THREADSAFE, cleanup_scheduler=_REAL_RUN_COROUTINE_THREADSAFE):
    loop = None
    th = None
    ready = None
    captured: list[_cf.Future] = []
    ctx = None
    m = None
    handle = None
    primary_exc = None
    primary_tb = None
    try:
        loop = _aio_l.new_event_loop()
        ready = _thr_l.Event()
        def _runner():
            _aio_l.set_event_loop(loop)
            ready.set()
            loop.run_forever()
        th = _thr_l.Thread(target=_runner, daemon=True)
        th.start()
        ready.wait(timeout)
        if not th.is_alive() or not ready.is_set():
            raise AssertionError("managed loop failed to start")
        def _schedule_owned(coro, tgt_loop):
            retained = coro
            try:
                fut = application_scheduler(retained, tgt_loop)
            except BaseException as sched_exc:
                try:
                    retained.close()
                except BaseException as close_exc:
                    raise BaseExceptionGroup("schedule rejection and coroutine close failure", [sched_exc, close_exc])
                raise
            captured.append(fut)
            return fut
        def _cap(coro, tgt_loop):
            return _schedule_owned(coro, tgt_loop)
        def _schedule(coro, loop_arg=None):
            return _schedule_owned(coro, loop)
        class _Handle:
            __slots__ = ("loop", "thread", "captured_futures", "schedule")
            def __init__(self, loop, thread, captured_futures, schedule_fn):
                self.loop = loop
                self.thread = thread
                self.captured_futures = captured_futures
                self.schedule = schedule_fn
            def __iter__(self):
                return iter((self.loop, self.thread, self.captured_futures, self.schedule))
            def __getitem__(self, idx):
                return (self.loop, self.thread, self.captured_futures, self.schedule)[idx]
        handle = _Handle(loop, th, captured, _schedule)
        ctx = monkeypatch.context()
        m = ctx.__enter__()
        m.setattr(_aio_l, "run_coroutine_threadsafe", _cap)
        try:
            import plugins.platforms.a2a.adapter as _mod
            m.setattr(_mod.asyncio, "run_coroutine_threadsafe", _cap)
        except BaseException:
            pass
        primary_adapter._loop = loop
        async def _no_op(_e):
            return None
        primary_adapter._message_handler = object()
        try:
            primary_adapter.handle_message = _no_op  # type: ignore[attr-defined]
        except BaseException:
            pass
        try:
            yield handle
        except BaseException as e:
            primary_exc = e
            primary_tb = sys.exc_info()[2]
    except BaseException as e:
        if primary_exc is None:
            primary_exc = e
            primary_tb = sys.exc_info()[2]
    finally:
        cleanup_failures: list[BaseException] = []
        for _f in list(captured):
            try:
                try:
                    is_done = _f.done()
                except BaseException as e:
                    cleanup_failures.append(BaseExceptionGroup("drain.settle.done", [e]))
                    continue
                if is_done:
                    try:
                        _f.result(timeout=0)
                    except _cf.CancelledError:
                        pass
                    except BaseException as e:
                        cleanup_failures.append(BaseExceptionGroup("drain.settle.result", [e]))
                else:
                    try:
                        _f.cancel()
                    except BaseException as e:
                        cleanup_failures.append(BaseExceptionGroup("drain.settle.cancel", [e]))
            except BaseException as e:
                cleanup_failures.append(BaseExceptionGroup("drain.settle.outer", [e]))
        drain_coro = None
        drain_future = None
        if loop is not None:
            async def _drain_impl():
                import asyncio as _a2
                failures: list[BaseException] = []
                known_tasks: set = set()
                self_task = None
                initial_tasks = None
                initial_unknown = False
                try:
                    try:
                        self_task = _a2.current_task()  # type: ignore[call-arg]
                    except TypeError:
                        self_task = _a2.current_task(loop=loop)  # type: ignore[call-arg]
                except BaseException as e:
                    failures.append(BaseExceptionGroup("drain.current_task", [e]))
                    self_task = None
                try:
                    try:
                        initial_tasks = set(_a2.all_tasks(loop))  # type: ignore[call-arg]
                    except TypeError:
                        initial_tasks = set(_a2.all_tasks())  # type: ignore[call-arg]
                except BaseException as e:
                    failures.append(BaseExceptionGroup("drain.initial_all_tasks", [e]))
                    initial_unknown = True
                    initial_tasks = None
                else:
                    initial_unknown = False
                    if initial_tasks is not None:
                        known_tasks.update(initial_tasks)
                todo: list = []
                if self_task is None or initial_unknown:
                    if self_task is None:
                        failures.append(AssertionError("drain.cancel_skipped_self_unknown"))
                    elif initial_unknown:
                        failures.append(AssertionError("drain.cancel_skipped_tasks_unknown"))
                else:
                    for t in list(initial_tasks):  # type: ignore[union-attr]
                        if t is self_task:
                            continue
                        try:
                            is_done = t.done()
                        except BaseException as e:
                            failures.append(BaseExceptionGroup("drain.cancel_done", [e]))
                            is_done = False
                        if is_done:
                            continue
                        todo.append(t)
                        try:
                            t.cancel()
                        except BaseException as e:
                            failures.append(BaseExceptionGroup("drain.cancel", [e]))
                            continue
                if todo:
                    gather_coro = None
                    try:
                        gather_coro = _a2.gather(*todo, return_exceptions=True)
                    except BaseException as e:
                        failures.append(BaseExceptionGroup("drain.gather", [e]))
                    else:
                        try:
                            results = await gather_coro  # type: ignore[assignment]
                        except BaseException as e:
                            failures.append(BaseExceptionGroup("drain.gather_await", [e]))
                        else:
                            for r in results:
                                if isinstance(r, BaseException) and not isinstance(r, _a2.CancelledError):
                                    failures.append(BaseExceptionGroup("drain.task_exception", [r]))
                try:
                    await _a2.sleep(0)
                except BaseException as e:
                    failures.append(BaseExceptionGroup("drain.yield", [e]))
                if self_task is None:
                    try:
                        try:
                            self_task_retry = _a2.current_task()  # type: ignore[call-arg]
                        except TypeError:
                            self_task_retry = _a2.current_task(loop=loop)  # type: ignore[call-arg]
                    except BaseException as e:
                        failures.append(BaseExceptionGroup("drain.final_current_task", [e]))
                        self_task_retry = None
                    else:
                        if self_task_retry is not None:
                            self_task = self_task_retry
                final_tasks = None
                final_unknown = False
                try:
                    try:
                        final_tasks = set(_a2.all_tasks(loop))  # type: ignore[call-arg]
                    except TypeError:
                        final_tasks = set(_a2.all_tasks())  # type: ignore[call-arg]
                except BaseException as e:
                    failures.append(BaseExceptionGroup("drain.final_all_tasks", [e]))
                    final_unknown = True
                    final_tasks = None
                else:
                    final_unknown = False
                    if final_tasks is not None:
                        known_tasks.update(final_tasks)
                pending_survivors: list = []
                if not final_unknown and final_tasks is not None:
                    for t in list(final_tasks):
                        if t is self_task:
                            continue
                        try:
                            is_done = t.done()
                        except BaseException as e:
                            failures.append(BaseExceptionGroup("drain.survivor_done", [e]))
                            is_done = False
                        if not is_done:
                            pending_survivors.append(t)
                            failures.append(AssertionError(f"drain.survivor {t!r}"))
                elif final_unknown:
                    pass
                salvage_tasks: list = []
                for t in list(known_tasks):
                    if t is self_task:
                        continue
                    try:
                        is_done = t.done()
                    except BaseException as e:
                        failures.append(BaseExceptionGroup("drain.salvage_done", [e]))
                        is_done = False
                    if not is_done:
                        salvage_tasks.append(t)
                        try:
                            t.cancel()
                        except BaseException as e:
                            failures.append(BaseExceptionGroup("drain.salvage_cancel", [e]))
                if salvage_tasks:
                    try:
                        salvage_gather = _a2.gather(*salvage_tasks, return_exceptions=True)
                    except BaseException as e:
                        failures.append(BaseExceptionGroup("drain.salvage_gather", [e]))
                    else:
                        try:
                            s_results = await salvage_gather  # type: ignore[assignment]
                        except BaseException as e:
                            failures.append(BaseExceptionGroup("drain.salvage_gather_await", [e]))
                        else:
                            for r in s_results:
                                if isinstance(r, BaseException) and not isinstance(r, _a2.CancelledError):
                                    failures.append(BaseExceptionGroup("drain.salvage_task_exception", [r]))
                try:
                    try:
                        proof_tasks = set(_a2.all_tasks(loop))  # type: ignore[call-arg]
                    except TypeError:
                        proof_tasks = set(_a2.all_tasks())  # type: ignore[call-arg]
                except BaseException as e:
                    failures.append(BaseExceptionGroup("drain.proof_all_tasks", [e]))
                else:
                    for t in list(proof_tasks):
                        if t is self_task:
                            continue
                        try:
                            is_done = t.done()
                        except BaseException as e:
                            failures.append(BaseExceptionGroup("drain.proof_done", [e]))
                            is_done = False
                        if not is_done:
                            failures.append(AssertionError(f"drain.proof_survivor {t!r}"))
                if failures:
                    raise BaseExceptionGroup("drain failed", failures)
                return []
            try:
                drain_coro = _drain_impl()
                try:
                    drain_future = cleanup_scheduler(drain_coro, loop)
                except BaseException as sched_exc:
                    try:
                        drain_coro.close()
                    except BaseException as close_exc:
                        cleanup_failures.append(BaseExceptionGroup("drain.schedule_and_close", [sched_exc, close_exc]))
                    else:
                        try:
                            is_closed = getattr(drain_coro, "cr_frame", None) is None
                        except BaseException:
                            is_closed = False
                        if not is_closed:
                            cleanup_failures.append(BaseExceptionGroup("drain.schedule_not_closed", [sched_exc, AssertionError("coroutine not closed after schedule rejection")]))
                        else:
                            cleanup_failures.append(BaseExceptionGroup("drain.schedule", [sched_exc]))
                    drain_future = None
                    drain_coro = None
            except BaseException as e:
                cleanup_failures.append(BaseExceptionGroup("drain.setup", [e]))
                if drain_coro is not None:
                    try:
                        drain_coro.close()
                    except BaseException as ce:
                        cleanup_failures.append(BaseExceptionGroup("drain.setup_close", [ce]))
                drain_future = None
            if drain_future is not None:
                try:
                    drain_future.result(timeout=timeout)
                except _cf.TimeoutError as e:
                    cleanup_failures.append(BaseExceptionGroup("drain.timeout", [e]))
                    try:
                        cancelled = drain_future.cancel()
                    except BaseException as ce:
                        cleanup_failures.append(BaseExceptionGroup("drain.cancel", [ce]))
                    else:
                        if not cancelled:
                            cleanup_failures.append(AssertionError("drain.cancel_not_accepted"))
                except BaseException as e:
                    if isinstance(e, BaseExceptionGroup):
                        for sub in e.exceptions:
                            cleanup_failures.append(sub)
                    else:
                        cleanup_failures.append(BaseExceptionGroup("drain.future_result", [e]))
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except BaseException as e:
                cleanup_failures.append(BaseExceptionGroup("drain.stop", [e]))
        if th is not None:
            try:
                th.join(timeout=timeout)
                if th.is_alive():
                    cleanup_failures.append(AssertionError(f"drain.join_timeout thread still alive after {timeout}s"))
            except BaseException as e:
                cleanup_failures.append(BaseExceptionGroup("drain.join", [e]))
                try:
                    if th.is_alive():
                        cleanup_failures.append(AssertionError("drain.join_thread_still_alive"))
                except BaseException as ee:
                    cleanup_failures.append(BaseExceptionGroup("drain.join_alive_check", [ee]))
        if loop is not None:
            try:
                try:
                    loop.close()
                except BaseException as e:
                    cleanup_failures.append(BaseExceptionGroup("drain.close", [e]))
                try:
                    is_closed = loop.is_closed()
                except BaseException as e:
                    cleanup_failures.append(BaseExceptionGroup("drain.is_closed", [e]))
                else:
                    if not is_closed:
                        cleanup_failures.append(AssertionError("drain.loop_not_closed"))
            except BaseException as e:
                cleanup_failures.append(BaseExceptionGroup("drain.close_outer", [e]))
        seen_ids = set()
        owned_adapters = []
        for ad in (primary_adapter,) + tuple(additional_adapters):
            if ad is None:
                continue
            oid = id(ad)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            owned_adapters.append(ad)
        for ad in owned_adapters:
            try:
                ad._unregister_adapter()
            except BaseException as e:
                cleanup_failures.append(BaseExceptionGroup(f"drain.unregister {ad!r}", [e]))
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except BaseException as e:
                cleanup_failures.append(BaseExceptionGroup("drain.restore", [e]))
        if primary_exc is None and not cleanup_failures:
            pass
        elif primary_exc is None and cleanup_failures:
            raise BaseExceptionGroup("managed-loop cleanup failed", cleanup_failures)
        elif primary_exc is not None and not cleanup_failures:
            raise primary_exc.with_traceback(primary_tb) if primary_tb is not None else primary_exc  # type: ignore[union-attr]
        else:
            cleanup_group = BaseExceptionGroup("managed-loop cleanup failed", cleanup_failures)
            raise BaseExceptionGroup("managed-loop primary and cleanup failed", [primary_exc, cleanup_group])
