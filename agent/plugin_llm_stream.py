"""Incremental plugin inference. No agent loop, history, tools or raw SDK events."""

from __future__ import annotations

import asyncio
import copy
import contextvars
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Resolution can run key commands or OAuth refresh. Bound both running work and
# admission: cancelled callers cannot grow an executor's unbounded pending queue.
_RESOLVERS = ThreadPoolExecutor(max_workers=4, thread_name_prefix='plugin-llm-route')
_RESOLVER_SLOTS = threading.BoundedSemaphore(4)


def _prepare(loop, home, call_kw):
    from agent import auxiliary_client as ac
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.left_core_migration import installs_held
    token = set_hermes_home_override(home)
    asyncio.set_event_loop(loop)  # Host async-client cache must retain its real owner loop.
    try:
        route = {}
        with installs_held():
            req, _, _ = ac._plan_aux_call(
                call_kw.pop('task'), async_mode=True, base_url=None, api_key=None,
                main_runtime=None, tools=None, reasoning_config=None, extra_headers=None,
                api_mode=None, strict_resolution=True, **{**call_kw, 'route_info': route})
        return req, route
    finally:
        asyncio.set_event_loop(None)
        reset_hermes_home_override(token)


async def _prepare_bounded(home, call_kw):
    if not _RESOLVER_SLOTS.acquire(blocking=False):
        raise PluginLlmStreamError('Host stream routing is busy; wait for pending resolutions before trying again')
    context = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    try:
        future = _RESOLVERS.submit(context.run, _prepare, loop, home, call_kw)
    except BaseException:
        _RESOLVER_SLOTS.release()
        raise
    future.add_done_callback(lambda _: _RESOLVER_SLOTS.release())
    # Shield keeps admission held until the actual worker has exited on cancellation.
    return await asyncio.shield(asyncio.wrap_future(future))


def _observe_close(task):
    if not task.cancelled() and task.exception() is not None:
        logger.warning('Plugin stream deadline cleanup failed: %s', type(task.exception()).__name__)


class PluginLlmStreamError(RuntimeError):
    """The selected host transport cannot satisfy the incremental contract."""


def _positive_bound(name, value, *, integer=False):
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or value <= 0:
        raise ValueError(f'{name} must be a positive finite {"integer" if integer else "number"}')
    if not integer and not math.isfinite(value):
        raise ValueError(f'{name} must be a positive finite number')
    return value


@dataclass(frozen=True)
class PluginLlmStreamEvent:
    """Only text deltas or a genuine terminal result; never provider payloads."""

    type: str
    text: str = ""
    finish_reason: str | None = None
    provider: str = ""
    model: str = ""
    agent_id: str = "default"
    usage: Any = None
    audit: dict[str, Any] = field(default_factory=dict)


def _sdk_error_type(error):
    from openai import APIError
    from httpx import HTTPError
    pending, seen = [error], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (APIError, HTTPError)):
            return type(item).__name__
        if isinstance(item, BaseException):
            pending.extend((item.__cause__, item.__context__, item.args))
            if isinstance(item, BaseExceptionGroup):
                pending.extend(item.exceptions)
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        elif isinstance(item, dict):
            pending.extend(item.values())
    return None


async def _public_sdk_call(operation):
    try:
        return await operation
    except Exception as exc:
        error_type = _sdk_error_type(exc)
        if error_type is None:
            raise
    # Raise outside the handler: SDK exceptions retain requests with auth headers.
    raise PluginLlmStreamError(
        f'Host stream failed ({error_type}); check provider readiness and host logs')


class PluginLlmAsyncStream:
    """Single-use, pull-based async context manager; closing never closes a host client."""

    def __init__(self, facade, agent, kw, purpose, max_output_chars):
        _positive_bound('max_output_chars', max_output_chars, integer=True)
        if kw['max_tokens'] is not None:
            _positive_bound('max_tokens', kw['max_tokens'], integer=True)
        if kw['timeout'] is not None:
            _positive_bound('timeout', kw['timeout'])
        self._facade, self._agent = facade, agent
        self._kw, self._purpose = copy.deepcopy(kw), purpose
        self._limit = max_output_chars
        self._home = get_hermes_home().resolve()
        self._raw = self._iterator = None
        self._entered = self._closed = False
        self._reason = None
        self._usage = None
        self._provider = self._model = ""
        self._chars = 0
        self._deadline = None
        self._pending = None
        self._timer = self._expiry_task = None
        self._expired = False
        self._semaphore = None
        self._close_done = None
        self._close_owner = None

    def _check_home(self):
        if get_hermes_home().resolve() != self._home:
            raise PluginLlmStreamError("Plugin stream profile changed; create a new stream in its owning profile")

    async def __aenter__(self):
        from agent import auxiliary_client as ac
        from agent.relay_runtime import active_turn
        self._check_home()
        if active_turn() is not None:
            raise PluginLlmStreamError(
                'Plugin streaming within a managed agent turn is not supported; call outside the turn or use acomplete')
        if self._entered or self._closed:
            raise PluginLlmStreamError("Plugin streams are single-use")
        self._entered = True
        if self._kw['timeout'] is None:
            configured = ac._get_auxiliary_task_config(self._kw['task']).get('timeout')
            if configured is not None:
                _positive_bound('timeout', configured)
        timeout = _positive_bound('timeout', ac._effective_aux_timeout(self._kw['task'], self._kw['timeout']))
        self._kw['timeout'] = timeout
        self._deadline = asyncio.get_running_loop().time() + timeout
        self._pending = asyncio.create_task(_public_sdk_call(self._open()))
        try:
            async with asyncio.timeout_at(self._deadline):
                return await self._pending
        except BaseException:
            await self._close_after_failure()
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError from None
            raise
        finally:
            self._pending = None

    def _check_access(self):
        self._check_home()
        self._facade._validate_access(
            self._kw['provider_override'], self._kw['model_override'], self._agent,
            self._kw['profile_override'], self._kw['task'])

    async def _open(self):
        from agent import auxiliary_client as ac
        self._check_access()
        call_kw, _ = self._facade._host_kwargs(self._kw)
        semaphore = ac._acquire_async_aux_semaphore(self._kw['task'])
        if semaphore is not None:
            await semaphore.acquire()
            self._semaphore = semaphore
        req, route = await _prepare_bounded(self._home, call_kw)
        from openai._client import AsyncOpenAI
        if not isinstance(req.client, AsyncOpenAI):
            raise PluginLlmStreamError(
                'Host streaming is not supported for this transport; select a Chat Completions auxiliary slot')
        self._provider = route.get('provider', req.request_provider)
        self._model = req.final_model or ''
        from agent.auxiliary_wire import prepare_chat_messages
        kwargs = prepare_chat_messages(req.client, dict(req.kwargs, stream=True, stream_options={'include_usage': True}, n=1))
        kwargs = ac.bypass_chat_sdk_request_transform(kwargs, req.client)
        # Both admission and routing can suspend: no await between live gates and dispatch.
        self._check_access()
        self._raw = await req.client.chat.completions.create(**kwargs)
        if not hasattr(self._raw, '__aiter__'):
            raise PluginLlmStreamError("Host route did not return an incremental async stream")
        self._iterator = self._raw.__aiter__()
        self._timer = asyncio.get_running_loop().call_at(self._deadline, self._expire)
        return self

    def _expire(self):
        self._expired = True
        self._expiry_task = asyncio.create_task(self.aclose())
        self._expiry_task.add_done_callback(_observe_close)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._expired:
            raise TimeoutError('Plugin stream exceeded its overall timeout')
        if self._pending is not None:
            raise PluginLlmStreamError('Only one read may be active on a plugin stream')
        self._pending = asyncio.create_task(_public_sdk_call(self._next()))
        try:
            return await self._pending
        except BaseException:
            if self._expired:
                raise TimeoutError('Plugin stream exceeded its overall timeout') from None
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError from None
            raise
        finally:
            self._pending = None

    async def _next(self):
        from agent.plugin_llm import _extract_usage
        try:
            self._check_home()
        except PluginLlmStreamError:
            await self.aclose()
            raise
        if self._closed:
            raise StopAsyncIteration
        if not self._entered:
            raise PluginLlmStreamError("Use async with ctx.llm.astream(...) before iterating")
        try:
            while True:
                try:
                    async with asyncio.timeout_at(self._deadline):
                        chunk = await anext(self._iterator)
                except StopAsyncIteration:
                    if not self._reason:
                        raise PluginLlmStreamError("Provider stream ended without a finish reason") from None
                    event = self._end()
                    await self.aclose()
                    return event
                if getattr(chunk, 'usage', None) is not None:
                    self._usage = _extract_usage(chunk)
                if isinstance(getattr(chunk, 'model', None), str):
                    self._model = chunk.model
                choices = getattr(chunk, 'choices', [])
                if len(choices) > 1 or (choices and choices[0].index != 0):
                    raise PluginLlmStreamError("Plugin streams require exactly one choice")
                if not choices:
                    continue
                choice = choices[0]
                if choice.finish_reason:
                    self._reason = choice.finish_reason
                text = getattr(choice.delta, 'content', None)
                if isinstance(text, str) and text:
                    self._chars += len(text)
                    if self._chars > self._limit:
                        raise PluginLlmStreamError("Plugin stream exceeded max_output_chars")
                    return PluginLlmStreamEvent('text_delta', text=text)
        except BaseException:
            await self._close_after_failure()
            raise

    def _end(self):
        from agent.plugin_llm import PluginLlmUsage
        usage = self._usage or PluginLlmUsage()
        audit = dict(plugin_id=self._facade._plugin_id, purpose=self._purpose or '',
                     profile=self._kw['profile_override'] or '', task=self._kw['task'] or '')
        logger.info('plugin_llm.astream plugin=%s provider=%s model=%s task=%s tokens=%d',
                    self._facade._plugin_id, self._provider, self._model, self._kw['task'], usage.total_tokens)
        return PluginLlmStreamEvent('end', finish_reason=self._reason, provider=self._provider,
                                    model=self._model, agent_id=self._agent or 'default', usage=usage, audit=audit)

    async def aclose(self):
        await _public_sdk_call(self._close())

    async def _close_after_failure(self):
        try:
            await self.aclose()
        except (Exception, asyncio.CancelledError) as exc:  # health: allow BLE001 -- preserve primary failure without logging credential-bearing tracebacks
            logger.warning('Plugin stream cleanup failed: %s', type(exc).__name__)

    async def _close_raw(self):
        if self._raw is not None:
            async with asyncio.timeout(2):
                await self._raw.close()

    async def _close(self):
        current = asyncio.current_task()
        if self._closed:
            if self._close_done is not None and current not in (self._close_owner, self._pending):
                await asyncio.shield(self._close_done)
            return
        self._closed = True
        self._close_owner = current
        self._close_done = asyncio.get_running_loop().create_future()
        if self._timer is not None:
            self._timer.cancel()
        try:
            try:
                pending = self._pending
                if pending is not None and not pending.done() and pending is not current:
                    pending.cancel()
                    try:
                        await pending
                    except asyncio.CancelledError:
                        if current.cancelling():
                            raise
            except BaseException:
                try:
                    await self._close_raw()
                except (Exception, asyncio.CancelledError) as exc:  # health: allow BLE001 -- preserve primary failure without logging credential-bearing tracebacks
                    logger.warning('Plugin stream cleanup failed: %s', type(exc).__name__)
                raise
            else:
                await self._close_raw()
        finally:
            if self._semaphore is not None:
                self._semaphore.release()
                self._semaphore = None
            self._close_done.set_result(None)

    async def __aexit__(self, *exc):
        if exc and exc[0] is not None:
            await self._close_after_failure()
        else:
            await self.aclose()
