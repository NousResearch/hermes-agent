"""
AsyncGenerator Streaming for Hermes-Agent.

This module provides synchronous generator-based streaming that adapts
the existing callback-based streaming pattern in run_agent.py.

Usage:
    from agent.hermes.streaming import stream_conversation, Delta

    for delta in stream_conversation(agent, "Hello"):
        print(delta.type, delta.content)
        if delta.done:
            break
"""

from dataclasses import dataclass, field
from typing import Generator, Optional, Any, Callable
import queue
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class Delta:
    """
    Represents a single unit of streamed content.

    Attributes:
        type: The type of delta - "text", "tool_call", "tool_result", "done", "error"
        content: The text content (for "text" type)
        tool_call: Tool call data (for "tool_call" type)
        done: True if this is the final delta (stream complete)
    """
    type: str  # "text", "tool_call", "tool_result", "done", "error"
    content: Optional[str] = None
    tool_call: Optional[dict] = None
    done: bool = False


def stream_conversation(
    agent,
    user_message: str,
    **kwargs
) -> Generator[Delta, None, None]:
    """
    Synchronous generator that adapts existing callback-based streaming.

    This function converts the callback-based streaming in run_agent.py to a
    synchronous generator pattern. It uses a queue + thread adapter to convert
    the callback calls into generator yields.

    IMPORTANT: This is a `def` function, NOT `async def`. It returns a
    Generator[Delta, None, None], not a coroutine.

    Args:
        agent: The AIAgent instance
        user_message: The user's message
        **kwargs: Additional arguments passed to _interruptible_streaming_api_call
            - stream_callback: Optional callback to chain with the generator

    Yields:
        Delta objects with type, content, tool_call, and done fields

    Backward Compatibility:
        The existing stream_delta_callback on the agent continues to work.
        If a stream_callback is provided in kwargs, it will be chained so
        both the generator yields AND the callback receive the deltas.

    Example:
        for delta in stream_conversation(agent, "Hello", async_stream=True):
            if delta.type == "text" and delta.content:
                print(delta.content, end="", flush=True)
            if delta.done:
                break
    """
    q: queue.Queue = queue.Queue()

    def stream_wrapper(content: str, delta_type: str, **extra):
        """Wrapper that puts deltas into the queue for generator yields."""
        tool_call = extra.get('tool_call')
        q.put(Delta(type=delta_type, content=content, tool_call=tool_call, done=False))

    # Handle callback chaining for backward compatibility
    stream_callback = kwargs.pop('stream_callback', None)
    if stream_callback:
        # Chain: our wrapper calls the user's callback too
        original_callback = stream_callback
        def chained_wrapper(content: str, delta_type: str, **extra):
            stream_wrapper(content, delta_type, **extra)
            try:
                original_callback(content, delta_type, **extra)
            except Exception as e:
                logger.warning(f"stream_callback error: {e}")
        kwargs['stream_callback'] = chained_wrapper
    else:
        kwargs['stream_callback'] = stream_wrapper

    # Capture the streaming kwargs
    api_kwargs = kwargs

    def run_in_thread():
        """Run the existing callback-based streaming in a separate thread."""
        try:
            agent._interruptible_streaming_api_call(
                user_message,
                **api_kwargs
            )
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            q.put(Delta(type="error", content=str(e)))
        finally:
            q.put(Delta(type="done", done=True))  # Sentinel to signal completion

    t = threading.Thread(target=run_in_thread)
    t.start()

    while True:
        delta = q.get()
        yield delta
        if delta.done:
            break