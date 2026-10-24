"""Keep fallback diagnostics aligned without weakening the caller's failure policy."""

from typing import Any


def run_attributed_recovery(
    first_error: Exception, request: Any, retry_kwargs: dict, candidate_kwargs: dict, *,
    task: str | None, route_info: dict[str, str] | None, attributed: bool,
) -> Any:
    from agent import auxiliary_client as aux

    swallowed_errors: list[Exception] = []
    candidate_raised = False
    final_route: dict[str, str] = {}

    def perform(step):
        nonlocal candidate_raised
        kind, args, kwargs = aux._ladder_step_call(step, request, retry_kwargs, candidate_kwargs)
        if kind == "call":
            return aux._validate_llm_response(aux._relay_sync_completion(*args, **kwargs), task)
        if kind == "retry":
            return aux._retry_same_provider_sync(**kwargs)
        try:
            result = aux._call_fallback_candidate_sync(*args, terminal_errors=swallowed_errors, **kwargs)
            if route_info is not None:
                final_route.update(route_info)
            return result
        except Exception:
            candidate_raised = True
            raise

    try:
        return aux._drive_ladder(
            aux._start_recovery_ladder(
                first_error, request, retry_kwargs, task=task, async_mode=False, route_info=route_info,
            ),
            perform,
        )
    except Exception as policy_error:
        if candidate_raised or not swallowed_errors or not attributed:
            raise
        # The ladder restores the primary route/error at exhaustion for legacy callers.
        # Attributed callers report the final wire, but retain that original policy error
        # independently so e.g. a primary 401 followed by a 429 cannot discard context.
        terminal_error = swallowed_errors[-1]
        terminal_error._auxiliary_preservation_error = policy_error
        if route_info is not None:
            route_info.update(final_route)
        raise terminal_error from first_error
