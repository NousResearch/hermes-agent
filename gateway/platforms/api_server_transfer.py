"""Bearer-bound user instruction provenance for API chat transfer tools."""

import hmac
from contextlib import contextmanager, suppress
from contextvars import ContextVar

request_transfer_auth: ContextVar[tuple[str, str]] = ContextVar(
    "api_server_transfer_auth", default=("", ""))


def bind_authenticated_request(adapter, request, profile):
    """Room grants and no-key listeners never mint a transfer identity."""
    expected_key = adapter._expected_api_key()
    authorization = request.headers.get("Authorization", "")
    bearer = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
    verified = bool(expected_key and hmac.compare_digest(bearer.encode(), expected_key.encode()))
    identity = (
        adapter._derive_browser_control_principal(profile or "default"),
        adapter._browser_control_transport_family(request)) if verified else ("", "")
    return request_transfer_auth.set(identity)


def clear_authenticated_request(token):
    request_transfer_auth.reset(token)


def current():
    return request_transfer_auth.get()


def bind_legacy_turn(identity, user_message, session_id):
    principal, family = identity
    if not principal or not family or not session_id or not isinstance(user_message, str):
        return
    from gateway.session_context import set_api_transfer_identity, set_turn_user_message
    set_turn_user_message(user_message)
    set_api_transfer_identity(principal, family, session_id)


@contextmanager
def admitted_serialized_tool_turn(session_key, session_id, agent):
    """Journal a bearer-bound API worker before its tools can use serialized services."""
    from gateway.session_context import get_api_transfer_identity, set_session_run_generation
    principal, family, conversation = get_api_transfer_identity()
    actual = getattr(agent, "session_id", None)
    if (not principal or not family or not conversation or not session_key or not session_id
            or not isinstance(actual, str) or actual != session_id):
        yield False
        return
    from agent.chat_execution_context import bind_chat_execution
    from gateway.chat_execution_recovery import ChatExecutionStore
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    store = ChatExecutionStore(home)
    generation = store.start_local_worker(session_key, actual)
    if generation is None:
        yield False
        return
    set_session_run_generation(generation)
    with bind_chat_execution(home, session_key, generation):
        try:
            yield True
        finally:
            store.worker_finished(session_key, generation)


def browser_control_transport_family(request):
    """A remote controller never satisfies a local-only browser dispatch."""
    host = None
    with suppress(Exception):
        transport = request.transport
        peer = transport.get_extra_info("peername") if transport is not None else None
        if isinstance(peer, tuple) and peer:
            host = peer[0]
        elif isinstance(peer, str):
            host = peer
    return "local-api" if host in ("127.0.0.1", "::1", "localhost") else "remote-api"
