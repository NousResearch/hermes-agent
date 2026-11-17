"""Serialize automation commands with topic handoff, before they capture a session id."""
import asyncio
from contextlib import asynccontextmanager
from functools import wraps


@asynccontextmanager
async def topic_boundary(runner, source):
    from gateway.fresh_session.settlement import is_group_topic

    if not is_group_topic(source):
        yield
        return
    key = runner._session_key_for_source(source)
    locks = runner.__dict__.setdefault('_fresh_session_locks', {})
    lock = locks.setdefault(key, asyncio.Lock())
    async with lock:
        yield


def serialize_command(handler):
    @wraps(handler)
    async def dispatch(runner, event):
        async with topic_boundary(runner, event.source):
            return await handler(runner, event)
    return dispatch
