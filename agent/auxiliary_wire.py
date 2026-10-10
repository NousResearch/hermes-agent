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
    # The private ``_reasoning_config`` sidecar is consumed only by
    # ``AnthropicAuxiliaryClient.create``; a plain OpenAI SDK client forwards it
    # into ``Completions.create()`` and raises TypeError (#123194). The producer
    # gate (``_build_call_kwargs``) is intentionally wider than the wrap rule —
    # a declared anthropic_messages demoted to chat_completions (#76836) and
    # compat-set members leak here — so strip where the resolved wire is known
    # instead of re-deriving it.
    kwargs = {k: v for k, v in kwargs.items() if k != "_reasoning_config"}
    messages = ChatCompletionsTransport().convert_messages(
        kwargs["messages"], model=kwargs.get("model"), base_url=str(getattr(client, "base_url", "") or ""),
    )
    return {**kwargs, "messages": messages}
