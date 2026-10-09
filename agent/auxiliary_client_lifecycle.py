"""Teardown of cached auxiliary-client transports."""

import contextlib
import inspect
from typing import Any


def _force_close_async_httpx(client: Any) -> None:
    """Mark the httpx AsyncClient inside an AsyncOpenAI client as closed so ``__del__`` won't
    schedule ``aclose()`` on a dead loop. Skips the full async close — the OS drops connections."""
    with contextlib.suppress(Exception):
        from httpx._client import ClientState
        inner = getattr(client, "_client", None)
        if inner is not None and not getattr(inner, "is_closed", True):
            inner._state = ClientState.CLOSED



def _schedule_async_close(close_result: Any, client: Any) -> None:
    """Finish an async close without leaking an unawaited coroutine."""
    async def _await_close() -> None:
        try:
            await close_result
        except Exception:
            pass
        finally:
            _force_close_async_httpx(client)
    runner = _await_close()
    try:
        import asyncio as _aio
        try:
            loop = _aio.get_running_loop()
        except RuntimeError:
            _aio.run(runner)
        else:
            task = loop.create_task(runner)

            def _consume(completed_task) -> None:
                with contextlib.suppress(BaseException):
                    completed_task.exception()
            task.add_done_callback(_consume)
            runner = None
    except Exception:
        if runner is not None:
            with contextlib.suppress(Exception):
                runner.close()
        _force_close_async_httpx(client)



def _close_cached_client(client: Any, *, close_async: bool = False) -> None:
    """Close one cached client, awaiting async transports only when safe."""
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if not callable(close_fn):
        _force_close_async_httpx(client)
        return
    try:
        close_result = close_fn()
    except Exception:
        _force_close_async_httpx(client)
        return
    if inspect.isawaitable(close_result):
        if close_async:
            _schedule_async_close(close_result, client)
        else:
            # Never await a client owned by another live loop; close the coroutine (no
            # unawaited warning) and neuter the transport.
            with contextlib.suppress(Exception):
                close_result.close()
            _force_close_async_httpx(client)
        return
    _force_close_async_httpx(client)

