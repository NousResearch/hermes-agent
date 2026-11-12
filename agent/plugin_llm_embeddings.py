"""Validation and trust boundary for the plugin embedding capability."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

MAX_INPUTS = 128
MAX_INPUT_BYTES = 1024 * 1024
MAX_DIMENSIONS = 65536


def _request(llm: Any, inputs: Sequence[str], *, input_type: str, provider: str | None,
             model: str | None, dimensions: int | None, timeout: float | None,
             profile: str | None, purpose: str | None) -> dict[str, Any]:
    from agent.plugin_llm import _check_overrides

    # Gate before resolving a provider or reading any of its credentials.
    policy = llm._policy_loader(llm._plugin_id)
    provider, model, _, profile = _check_overrides(
        policy, requested_provider=provider, requested_model=model,
        requested_agent_id=None, requested_profile=profile)
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence) or not 0 < len(inputs) <= MAX_INPUTS:
        raise ValueError("Embeddings require between 1 and 128 text inputs")
    if any(not isinstance(item, str) or not item.strip() for item in inputs):
        raise ValueError("Embedding inputs must be non-empty text")
    try:
        input_bytes = sum(len(item.encode("utf-8")) for item in inputs)
    except UnicodeEncodeError:
        raise ValueError("Embedding inputs must be valid UTF-8 text") from None
    if input_bytes > MAX_INPUT_BYTES:
        raise ValueError("Embedding inputs exceed the 1 MiB request limit")
    if input_type not in {"query", "document"}:
        raise ValueError("Embedding input_type must be query or document")
    if dimensions is not None and (type(dimensions) is not int or not 0 < dimensions <= MAX_DIMENSIONS):
        raise ValueError("Embedding dimensions must be an integer between 1 and 65536")
    timeout = 60.0 if timeout is None else timeout
    if not _finite_number(timeout) or not 0 < timeout <= 300:
        raise ValueError("Embedding timeout must be finite and between 0 and 300 seconds")
    return dict(inputs=list(inputs), input_type=input_type, provider=provider, model=model,
                dimensions=dimensions, timeout=float(timeout), profile=profile)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _invalid() -> None:
    from agent.embedding_client import EmbeddingError
    raise EmbeddingError("embedding_invalid_response")


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _vectors(response: Any, count: int, requested: int | None) -> list[list[float]]:
    data = _field(response, "data")
    if not isinstance(data, (list, tuple)) or len(data) != count:
        _invalid()
    ordered: list[Any] = [None] * count
    for row in data:
        index = _field(row, "index")
        vector = _field(row, "embedding")
        if type(index) is not int or not 0 <= index < count or ordered[index] is not None:
            _invalid()
        if not isinstance(vector, (list, tuple)) or not 0 < len(vector) <= MAX_DIMENSIONS:
            _invalid()
        if any(not _finite_number(v) for v in vector):
            _invalid()
        ordered[index] = [float(v) for v in vector]
    dimensions = len(ordered[0])
    if (requested is not None and dimensions != requested) or any(len(v) != dimensions for v in ordered):
        _invalid()
    return ordered


def _finish(llm: Any, kw: dict[str, Any], invoked: tuple[str, str, Any], purpose: str | None) -> Any:
    from agent.plugin_llm import PluginLlmEmbeddingResult, _extract_usage, logger

    provider, model, response = invoked
    if not isinstance(provider, str) or not provider or not isinstance(model, str) or not model:
        _invalid()
    vectors = _vectors(response, len(kw["inputs"]), kw["dimensions"])
    handling = _field(response, "_hermes_input_type_handling")
    if handling not in ("shared", "input_type"):
        _invalid()
    audit = {"plugin_id": llm._plugin_id, "purpose": purpose or "", "profile": kw["profile"] or "",
             "input_type": kw["input_type"], "input_type_handling": handling,
             "normalization": "none"}
    # Model identity only: text inputs and provider exception bodies never enter audit logs.
    logger.info("plugin_llm.embed plugin=%s provider=%s model=%s dimensions=%d count=%d",
                llm._plugin_id, provider, model, len(vectors[0]), len(vectors))
    return PluginLlmEmbeddingResult(vectors=vectors, provider=provider, model=model,
                                    dimensions=len(vectors[0]), usage=_extract_usage(response), audit=audit)


def embedding_call(llm: Any, inputs: Sequence[str], **kwargs: Any) -> Any:
    from agent.embedding_client import call_embeddings

    kw = _request(llm, inputs, **kwargs)
    return _finish(llm, kw, call_embeddings(**kw), kwargs["purpose"])


async def async_embedding_call(llm: Any, inputs: Sequence[str], **kwargs: Any) -> Any:
    from agent.embedding_client import async_call_embeddings

    kw = _request(llm, inputs, **kwargs)
    return _finish(llm, kw, await async_call_embeddings(**kw), kwargs["purpose"])
