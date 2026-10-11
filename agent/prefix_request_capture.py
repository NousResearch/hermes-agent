"""Capture ordinary requests at the final provider boundary."""

from contextlib import contextmanager
from contextvars import ContextVar
import time

from agent.gemini_native_adapter import is_native_gemini_base_url
from agent.sdk_transform_bypass import bypass_chat_sdk_request_transform

_MAIN_CAPTURE_SCOPE = ContextVar("hermes_main_prefix_capture", default=None)


@contextmanager
def main_capture_scope(agent):
    """Limit capture to this ordinary call, including its copied worker context."""
    from agent.prefix_request import enabled

    if not enabled(agent):
        yield
        return
    scope = {"owner": agent, "active": True}
    token = _MAIN_CAPTURE_SCOPE.set(scope)
    try:
        yield
    finally:
        scope["active"] = False
        _MAIN_CAPTURE_SCOPE.reset(token)


def capture_main_request(agent, kwargs):
    """Return a capture token only inside this agent's active ordinary call."""
    from agent.prefix_request import begin_capture

    scope = _MAIN_CAPTURE_SCOPE.get()
    if not isinstance(scope, dict) or scope.get("owner") is not agent or not scope.get("active"):
        return None
    return begin_capture(agent, kwargs)


def open_main_chat_stream(driver, stream_kwargs, capture_store):
    """Open a chat stream and keep its own final request token for response assembly."""
    agent = driver.agent
    if not is_native_gemini_base_url(agent.base_url) and not getattr(agent, "_stream_options_unsupported", False):
        stream_kwargs["stream_options"] = {"include_usage": True}
    request_client = driver._attempt_request_client = driver.clients.set_client(
        agent._create_request_openai_client(reason="chat_completion_stream_request", api_kwargs=stream_kwargs))
    driver.last_chunk_time["t"] = time.time()
    agent._touch_activity("waiting for provider response (streaming)")
    stream_kwargs = bypass_chat_sdk_request_transform(stream_kwargs, request_client)
    capture_store.update(kwargs=stream_kwargs, token=capture_main_request(agent, stream_kwargs))
    return request_client.chat.completions.create(**stream_kwargs)


def bind_stream_capture(agent, capture_store, response):
    """Bind the assembled response to this stream's physical request, if one was sent."""
    from agent.prefix_request import capture_response

    return capture_response(agent, capture_store.get("kwargs", {}), response, capture=capture_store.get("token"))
