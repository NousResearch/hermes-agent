"""Auxiliary relay codec metadata, concrete served-route recording and the served-dispatch observer."""
from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Iterator, Optional

if TYPE_CHECKING:
    from agent.auxiliary_client import _LadderRoute


def _record_route_info(
    route_info: Optional[dict[str, str]], provider: Optional[str], model: Optional[str]
) -> None:
    """Expose the concrete route selected for one auxiliary call."""
    if route_info is not None:
        route_info["provider"] = provider or "auto"
        route_info["model"] = model or "default"


# The dispatch that answered one auxiliary call. An observer installs a fresh dict; every dispatch that
# returns a response overwrites it, after that response, so once ``call_llm`` returns it names the
# dispatch whose response came back (a response rejected by validation is overwritten by the dispatch
# that then serves). A ContextVar holding a mutable dict: a worker thread that copied the context
# writes into the same dict, and calls outside an observer cost one lookup.
_SERVED_DISPATCH: contextvars.ContextVar[Optional[dict[str, Any]]] = (
    contextvars.ContextVar("hermes_aux_served_dispatch", default=None)
)


@contextlib.contextmanager
def observe_served_dispatch() -> Iterator[dict[str, Any]]:
    """Collect the dispatch that answers the auxiliary calls made inside this block (the last one wins)."""
    observed: dict[str, Any] = {}
    token = _SERVED_DISPATCH.set(observed)
    try:
        yield observed
    finally:
        _SERVED_DISPATCH.reset(token)


def _note_served_dispatch(
    response: Any, client: Any, request: dict[str, Any], provider: Optional[str], api_mode: Optional[str],
) -> Any:
    """Record the dispatch that returned ``response`` and hand the response back unchanged.

    Host and credential are read from the ONE client that answered, so they always belong together;
    model and timeout are those of the request it was sent. Only the non-streaming completion seams
    record (``_relay_sync_completion`` / ``_relay_async_completion``, through ``_ServedDispatch``):
    every recovery-ladder lane dispatches through them, and the caller-consumed stream path has no
    complete response to name (it only clears the observation).
    """
    observed = _SERVED_DISPATCH.get()
    if observed is not None:
        from agent import auxiliary_client as owner
        api_key = getattr(client, "api_key", None)
        # Ladder "call" rungs pass the route label ("auto", "fallback_chain[0](x)"): keep the named backend,
        # else the one an auto route tagged on its client; an unnamed lane leaves the host to identify it.
        concrete = owner._fallback_provider_from_label(provider or "")
        if concrete.lower() in ("", "auto"):
            concrete = owner._effective_provider_for_client(client, "")
        observed.clear()
        observed.update({
            "provider": concrete if concrete.lower() not in ("", "auto") else None,
            "model": request.get("model") or None,
            "base_url": str(getattr(client, "base_url", "") or "") or None,
            "api_key": api_key if isinstance(api_key, str) and api_key else None,
            "api_mode": api_mode or None, "timeout": request.get("timeout"),
        })
    return response


class _ServedDispatch:
    """One completion-seam dispatch: clears the observation before sending, names it once it answers.

    Clearing first means a dispatch that fails, or a serving path that bypasses the seams, leaves the
    observation empty (the compressor then fails closed) instead of a stale earlier dispatch. The
    wrapped provider callback keeps the request actually sent, after any Relay rewrite and retry.
    """

    __slots__ = ("api_mode", "client", "provider", "request")

    def __init__(self, client: Any, request: dict[str, Any], provider: Optional[str], api_mode: Optional[str]):
        _clear_served_dispatch()
        self.client, self.request, self.provider, self.api_mode = client, request, provider, api_mode

    def sent(self, callback: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        def send(request: dict[str, Any]) -> Any:
            self.request = request
            return callback(request)
        return send

    def asent(self, callback: Callable[[dict[str, Any]], Awaitable[Any]]) -> Callable[[dict[str, Any]], Awaitable[Any]]:
        async def send(request: dict[str, Any]) -> Any:
            self.request = request
            return await callback(request)
        return send

    def answered(self, response: Any) -> Any:
        return _note_served_dispatch(response, self.client, self.request, self.provider, self.api_mode)


def _clear_served_dispatch() -> None:
    """A new dispatch starts: forget the previous one (also called by the unrecorded stream seam)."""
    observed = _SERVED_DISPATCH.get()
    if observed is not None:
        observed.clear()


def _relay_codec_api_mode(api_mode: str, request: dict[str, Any]) -> str:
    """Relay codec label for the body this attempt actually hands to Relay.

    The auxiliary helpers below always hand Relay Chat Completions kwargs (``prepare_chat_messages``
    plus a ``chat.completions.create`` callback); the Codex adapter translates them to Responses inside
    that callback and answers in chat shape. A ``codex_responses`` destination label (a Codex fallback
    or recovery rung) made Relay's Responses codec refuse such a body before dispatch ("request is
    missing input"), so the attempt meant to rescue a failed summary never reached the provider.
    """
    if api_mode == "codex_responses" and "messages" in request and "input" not in request:
        return "chat_completions"
    return api_mode


def _relay_auxiliary_metadata(
    *, provider: str | None = None, api_mode: str | None = None, request: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]] | None:
    from agent import auxiliary_client as owner
    context = owner._RELAY_AUX_CALL_CONTEXT.get()
    if context is None:
        return None
    attempt_count = int(context.get("attempt_count") or 0)
    context["attempt_count"] = attempt_count + 1
    provider_name = str(provider or context.get("provider") or "auxiliary")
    model_name = str(context.get("model") or "unknown")
    return provider_name, model_name, {
        "api_mode": _relay_codec_api_mode(
            str(api_mode or context.get("api_mode") or "chat_completions"), request or {}),
        "api_request_id": str(context["request_id"]),
        "call_role": f"auxiliary:{context['task']}",
        "retry_count": attempt_count,
        "auxiliary_task": str(context["task"]),
    }


def _record_swapped_model(route: _LadderRoute, model: str) -> None:
    """A rung that swaps ``kwargs['model']`` serves another route: record it like a fallback lane.

    Keeps the provider already recorded for this call; writing ``model`` is what tells a
    ``_SummaryRouteRecord`` that the ladder left the prepared route.
    """
    from agent import auxiliary_client as owner
    recorded = route.route_info.get("provider") if route.route_info is not None else None
    owner._record_route_info(route.route_info, recorded or owner._fallback_provider_from_label(route.resolved_provider), model)
