"""OpenAI-client-compatible auxiliary clients over the AWS Bedrock Converse API.

The Bedrock adapter family (R1-C11) extracted from agent/auxiliary_client.py.
"""

import logging

from typing import Any

logger = logging.getLogger(__name__)


class _BedrockCompletionsAdapter:
    """Translates ``chat.completions.create(**kwargs)`` into Bedrock Converse."""

    def __init__(self, region: str, model: str):
        self._region = region
        self._model = model

    def create(self, **kwargs) -> Any:
        from agent.bedrock_adapter import call_converse

        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        # OpenAI accepts ``stop`` as str or list; Converse requires a list.
        stop = kwargs.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if kwargs.get("tool_choice") is not None:
            # Converse's toolChoice isn't wired through call_converse();
            # no in-tree auxiliary caller passes tool_choice today. Surface
            # the drop instead of silently ignoring it.
            logger.debug(
                "BedrockAuxiliaryClient: tool_choice=%r not supported by the "
                "Converse shim — ignored.", kwargs.get("tool_choice"),
            )
        if kwargs.get("stream"):
            # Converse streaming isn't wired through this shim. Return a
            # complete response instead — call_llm's streaming consumer
            # detects a final object and downgrades to non-live output.
            logger.debug(
                "BedrockAuxiliaryClient: stream=True requested for %s — "
                "returning a complete response (Converse shim does not "
                "stream); caller downgrades to non-streaming.",
                model,
            )
        return call_converse(
            region=self._region,
            model=model,
            messages=messages,
            tools=kwargs.get("tools"),
            max_tokens=int(max_tokens) if max_tokens else 4096,
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            stop_sequences=stop,
        )


class _BedrockChatShim:
    def __init__(self, adapter: "_BedrockCompletionsAdapter"):
        self.completions = adapter


class BedrockAuxiliaryClient:
    """OpenAI-client-compatible wrapper over AWS Bedrock Converse API."""

    def __init__(self, region: str, model: str):
        self._region = region
        self._model = model
        adapter = _BedrockCompletionsAdapter(region, model)
        self.chat = _BedrockChatShim(adapter)
        self.api_key = "aws-sdk"
        self.base_url = f"https://bedrock-runtime.{region}.amazonaws.com"

    def close(self):
        pass


class _AsyncBedrockCompletionsAdapter:
    def __init__(self, sync_adapter: _BedrockCompletionsAdapter):
        self._sync = sync_adapter

    async def create(self, **kwargs) -> Any:
        import asyncio
        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncBedrockChatShim:
    def __init__(self, adapter: _AsyncBedrockCompletionsAdapter):
        self.completions = adapter


class AsyncBedrockAuxiliaryClient:
    def __init__(self, sync_wrapper: "BedrockAuxiliaryClient"):
        sync_adapter = sync_wrapper.chat.completions
        async_adapter = _AsyncBedrockCompletionsAdapter(sync_adapter)
        self.chat = _AsyncBedrockChatShim(async_adapter)
        self.api_key = sync_wrapper.api_key
        self.base_url = sync_wrapper.base_url
