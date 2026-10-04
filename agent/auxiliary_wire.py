"""Message hygiene at the resolved auxiliary client boundary."""

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
    """Adapters can expose chat completions while using a different native protocol."""
    if getattr(getattr(client, "chat", None), "completions", None) is not None:
        return "chat_completions"
    return api_mode or "chat_completions"
