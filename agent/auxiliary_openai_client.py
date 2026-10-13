"""Lazy construction of usage-validated auxiliary OpenAI clients."""

from typing import Any, TYPE_CHECKING

from agent.process_bootstrap import _OpenAIProxy

if TYPE_CHECKING:
    from openai import OpenAI


class _AuxiliaryOpenAIProxy(_OpenAIProxy):
    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> "OpenAI":
        from agent.native_usage_openai import UsageValidatedOpenAI

        return UsageValidatedOpenAI(*args, **kwargs)
