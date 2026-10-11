"""Apply plugin policies at the shared auxiliary provider-attempt boundary."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from agent.auxiliary_hooks import _parent_turn_identity, arun_with_aux_hooks, run_with_aux_hooks
from hermes_cli.middleware import apply_llm_request_middleware, run_llm_execution_middleware
from hermes_cli.middleware_async import run_llm_execution_middleware_async


def _prepare_attempt(
    *, aux_task: str, metadata: dict[str, Any], client: Any, kwargs: dict[str, Any],
    provider: str, model: str, api_mode: str, streaming: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    context = dict(
        _parent_turn_identity(), task=aux_task, aux_task=aux_task,
        api_request_id=str(metadata.get("api_request_id") or ""),
        retry_count=int(metadata.get("retry_count") or 0),
        api_call_count=int(metadata.get("retry_count") or 0) + 1,
        provider=provider, model=model, api_mode=api_mode, streaming=streaming,
        base_url=str(getattr(client, "base_url", "") or ""),
    )
    result = apply_llm_request_middleware(kwargs, **context)
    context.update(original_request=result.original_payload, middleware_trace=list(result.trace))
    context["model"] = str(result.payload.get("model") or model)
    hook_context = dict(
        aux_task=aux_task, metadata={**metadata, "middleware_trace": list(result.trace)},
        client=client, kwargs=result.payload, provider=provider,
        model=context["model"], api_mode=api_mode,
    )
    return result.payload, context, hook_context


def run_auxiliary_attempt(
    call: Callable[[dict[str, Any]], Any], *, aux_task: str, metadata: dict[str, Any],
    client: Any, kwargs: dict[str, Any], provider: str, model: str, api_mode: str,
    streaming: bool = False,
) -> Any:
    request, context, hooks = _prepare_attempt(
        aux_task=aux_task, metadata=metadata, client=client, kwargs=kwargs,
        provider=provider, model=model, api_mode=api_mode, streaming=streaming,
    )
    return run_with_aux_hooks(
        lambda: run_llm_execution_middleware(request, call, **context),
        **hooks, streaming=streaming,
    )


async def arun_auxiliary_attempt(
    call: Callable[[dict[str, Any]], Awaitable[Any]], *, aux_task: str,
    metadata: dict[str, Any], client: Any, kwargs: dict[str, Any],
    provider: str, model: str, api_mode: str,
) -> Any:
    request, context, hooks = _prepare_attempt(
        aux_task=aux_task, metadata=metadata, client=client, kwargs=kwargs,
        provider=provider, model=model, api_mode=api_mode,
    )
    return await arun_with_aux_hooks(
        lambda: run_llm_execution_middleware_async(request, call, **context), **hooks,
    )
