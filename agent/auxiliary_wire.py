"""Request and codec shapes at the resolved auxiliary client boundary.

Auxiliary callers own clients and requests; these stateless transformations keep
Relay's intercepted chat surface separate from the adapter's native wire mode.
"""

from openai import AsyncOpenAI, OpenAI

from agent.transports.chat_completions import ChatCompletionsTransport


def prepare_chat_messages(client, kwargs: dict) -> dict:
    """Sanitize actual Chat Completions SDK requests, not native adapter replay.

    Auxiliary and MoA callers can retain a prepared request before the virtual
    transport sanitizes its copy. The resolved SDK client identifies the wire;
    native Messages/Responses adapters must retain their reasoning sidecars.
    """
    if not isinstance(client, (OpenAI, AsyncOpenAI)) or "messages" not in kwargs:
        return kwargs
    messages = ChatCompletionsTransport().convert_messages(
        kwargs["messages"], model=kwargs.get("model"), base_url=str(getattr(client, "base_url", "") or ""),
    )
    return {**kwargs, "messages": messages}


def relay_boundary_api_mode(client: object, api_mode: str | None) -> str:
    """Select a codec for the intercepted surface, not the adapter's native wire.

    Native mode still selects the client upstream. Responses/Messages adapters
    expose chat completions here, so Relay must decode their chat-shaped body.
    Non-chat clients retain their supplied mode.
    """
    if getattr(getattr(client, "chat", None), "completions", None) is not None:
        return "chat_completions"
    return api_mode or "chat_completions"
