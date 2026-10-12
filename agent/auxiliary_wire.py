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
    from agent.reasoning_carriers import shape_wire_carriers

    base_url = str(getattr(client, "base_url", "") or "")
    messages = ChatCompletionsTransport().convert_messages(
        shape_wire_carriers(kwargs["messages"], model=kwargs.get("model"), base_url=base_url),
        model=kwargs.get("model"), base_url=base_url,
    )
    return {**kwargs, "messages": messages}
