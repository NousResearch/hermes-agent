"""The ``manage_connections`` card on the session authority: ``connection.respond``, the operation's ``status``/``wake`` and the card's Try again (``connectors.connect`` with ``reconnect``), plus the open card a resume restores (``pending_connection``).

The operation lives in ``tools/connectors/live.py`` under the turn's session key (the live session's route) and profile, so a card can only drive its own session's operation. Account-owned calls (the Connectors page) are not the session's and stay on the legacy handlers (``tui_gateway/methods_connectors.py``). Answers and refusals share that file's code (``tools.connectors.run.respond``, ``tui_gateway/connector_refusals.py``); ``ConnectorRefusal`` carries a refusal through the authority's error path.
"""
import asyncio
from functools import partial

from hermes_state_runtime import RuntimeStoreError

METHODS = ('connection.respond', 'connectors.operation.status', 'connectors.operation.wake', 'connectors.connect')


class ConnectorRefusal(RuntimeStoreError):
    """A refusal on the legacy connector handlers' wire shape: their code, their human message, their ``ConnectorErrorReason`` as ``data.reason``."""

    def __init__(self, code, reason, message):
        super().__init__(str(reason), detail=message)
        self.wire_code, self.wire_message = code, message


def session_owned(params) -> bool:
    owner = params.get('owner') if isinstance(params, dict) else None
    return isinstance(owner, dict) and owner.get('type') == 'session'


def handlers(connection):
    return {method: partial(_handle, connection, method) for method in METHODS}


def pending_connection(authority, session_id):
    """The open operation of a live session as its ``connection.request`` payload, or None."""
    from tools.connectors import live
    operation = live.current(authority.sessions[session_id].route, profile_home=authority.profile_id)
    return operation.request_payload() if operation is not None else None


def _request(method, params):
    """The validated envelope and, for ``connection.respond``, the answer validated on its own as the legacy handler does: a bad answer is ``INVALID_ANSWER``, not a bad envelope."""
    from pydantic import ValidationError
    from tui_gateway.connector_refusals import INVALID_ANSWER, INVALID_PARAMS
    from tui_gateway.contracts.connectors_operation import ConnectionAnswer, ConnectionOperationParams
    from tui_gateway.contracts.registry import METHODS as CONTRACTS
    respond = method == 'connection.respond'
    contract = ConnectionOperationParams if respond else CONTRACTS[method].params
    try:
        request = contract.model_validate({k: v for k, v in params.items() if not (respond and k == 'result')})
    except ValidationError as exc:
        raise ConnectorRefusal(*INVALID_PARAMS) from exc
    if not respond:
        return request, None
    try:
        return request, ConnectionAnswer.model_validate(params.get('result'))
    except ValidationError as exc:
        raise ConnectorRefusal(*INVALID_ANSWER) from exc


def _view(operation):
    from tui_gateway.connector_payload import connector_ui_payload
    snapshot = operation.snapshot()
    return connector_ui_payload({**snapshot, 'settled': snapshot.get('settled_at') is not None})


async def _handle(connection, method, _ref, params):
    from gateway.session_busy_controls import authorize
    from gateway.session_contract import SessionRef
    from tools.connectors import live
    from tui_gateway import connector_refusals as refusals
    request, answer = _request(method, params)
    authority = connection.authority
    ref = SessionRef(authority.profile_id, request.owner.session_id)
    # Capability before existence, as every authority verb: a refused caller learns nothing about the id.
    authorize(connection, ref, params, 'session:read' if method == 'connectors.operation.status' else 'session:control')
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    route = authority.sessions[ref.session_id].route
    if method == 'connectors.connect':
        operation = live.current(route, profile_home=authority.profile_id)
    else:
        operation = live.get(route, request.op_id, profile_home=authority.profile_id)
    if operation is None:
        raise ConnectorRefusal(*refusals.UNKNOWN_OPERATION)
    if method == 'connectors.operation.status':
        return _view(operation)
    if method == 'connectors.operation.wake':
        operation.wake.set()
        return {'status': 'ok'}
    if method == 'connection.respond':
        return await asyncio.to_thread(_respond, operation, answer)
    return await asyncio.to_thread(_retry, authority, ref, operation, request)


def _respond(operation, answer):
    from tools.connectors.operation import IllegalTransition
    from tools.connectors.run import respond
    from tui_gateway.connector_refusals import INVALID_ANSWER
    try:
        return {'status': 'ok', 'settled': respond(operation, answer.model_dump_json(exclude_none=True))}
    except IllegalTransition as exc:
        raise ConnectorRefusal(*INVALID_ANSWER) from exc


def _retry(authority, ref, operation, request):
    """The card's Try again, under the session's own context, as the sidecar ran it for its session."""
    from tools.connectors import run
    from tools.connectors.gateway.errors import GatewayAuthError
    from tui_gateway.connector_refusals import (INVALID_PARAMS, REISSUE_REFUSALS, REISSUE_REFUSED, REQUEST_FAILED,
                                                UNKNOWN_TARGET_REFUSAL, auth_refusal)
    if not request.reconnect:
        raise ConnectorRefusal(*INVALID_PARAMS)
    for name in request.connectors:
        target = operation.target(name)
        # A retry naming another account must not re-mint this one.
        if target is not None and ((request.alias is not None and target.alias != request.alias)
                                   or request.connection_id not in (None, target.repair_id, target.connection_id)):
            raise ConnectorRefusal(*UNKNOWN_TARGET_REFUSAL)
    with _session_context(authority, ref):
        _require_connectors(authority, ref)
        try:
            reason = run.reissue(operation, request.connectors)
        except RuntimeStoreError:
            raise
        except GatewayAuthError as exc:
            raise ConnectorRefusal(*auth_refusal(exc)) from exc
        except Exception as exc:  # health: allow BLE001 -- the sidecar's _connector_guard: one named failure
            raise ConnectorRefusal(*REQUEST_FAILED) from exc
    if reason is not None:
        raise ConnectorRefusal(*REISSUE_REFUSALS.get(reason, REISSUE_REFUSED))
    return _view(operation)


def _require_connectors(authority, ref):
    """The sidecar's ``_session_connector_gate``: connectors on, and ``manage_connections`` in the session's tools."""
    import model_tools
    from tools.connectors import connectors_available
    from tui_gateway import connector_refusals as refusals
    enabled, disabled = _session_toolsets(authority, ref)
    if 'manage_connections' not in model_tools._select_tool_names(enabled, disabled, quiet_mode=True) or not connectors_available():
        raise ConnectorRefusal(*refusals.CONNECTORS_UNAVAILABLE)


def _session_toolsets(authority, ref):
    agent = authority.agent(ref)
    if agent is not None:
        return getattr(agent, 'enabled_toolsets', None), getattr(agent, 'disabled_toolsets', None)
    from agent.skill_utils import parse_config_string_list
    from gateway.session_policy import policy_for_source
    from toolsets import session_disabled_toolsets
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    config = policy.config(authority, secrets=False) if policy is not None else {}
    enabled = list(policy.current_toolsets(config)) if policy is not None else None
    disabled = session_disabled_toolsets(parse_config_string_list((config.get('agent') or {}).get('disabled_toolsets')) or None,
                                         policy.platform if policy is not None else 'cli',
                                         oneshot=policy is not None and policy.source == 'oneshot')
    return enabled, disabled


def _session_context(authority, ref):
    """The session vars a turn of this session binds (``GatewayRunner._session_env_scope``), so a re-mint carries the turn's caller identity and session key."""
    from gateway.session import SessionContext
    from gateway.session_ancillary import physical_target
    live = authority.sessions[ref.session_id]
    context = SessionContext(source=live.source, connected_platforms=[], home_channels={},
                             session_key=live.route, session_id=physical_target(authority, ref))
    return authority.runner._session_env_scope(context)
