"""Connector refusals both connector handlers send: the legacy one (``methods_connectors``) and the session authority's (``gateway/session_connections``). Each entry is ``(code, ConnectorErrorReason, message)``."""
from tools.connectors.run import LINK_STILL_VALID, MIXED_KINDS, NOT_ALLOWED, REFUSED, SETTLED, UNKNOWN_TARGET

from tui_gateway.contracts.connectors import ConnectorErrorReason as Reason

INVALID_PARAMS = (4000, Reason.invalid_params, "Connector parameters are invalid.")
INVALID_ANSWER = (4002, Reason.invalid_answer, "Connection answer is invalid.")
UNKNOWN_OPERATION = (4004, Reason.unknown_operation, "No open operation with that op_id.")
UNKNOWN_TARGET_REFUSAL = (4004, Reason.unknown_target, "No such target on the open operation.")
CONNECTORS_UNAVAILABLE = (4031, Reason.connectors_unavailable, "Connectors are not available in this session.")
REQUEST_FAILED = (5034, Reason.connector_request_failed, "Connector request failed. Try again explicitly.")
REISSUE_REFUSED = (4002, Reason.reissue_refused, "The target cannot be run again.")
REISSUE_REFUSALS = {
    UNKNOWN_TARGET: UNKNOWN_TARGET_REFUSAL,
    MIXED_KINDS: (4000, Reason.invalid_params, "One target kind per request."),
    LINK_STILL_VALID: (4002, Reason.link_still_valid, "Reopen the stored link."),
    SETTLED: (4002, Reason.reissue_refused, "The operation has settled."),
    NOT_ALLOWED: (4002, Reason.reissue_refused, "This target cannot be run again."),
    REFUSED: REISSUE_REFUSED,
}


def auth_refusal(exc):
    """The refusal for a ``GatewayAuthError``."""
    if exc.code in {"no_access", "ORG_ACCESS_DENIED"}:
        return 4030, Reason.org_access_denied, "This account cannot manage connectors for this organization."
    if exc.status == 401 or exc.code in {"invalid_token", "INVALID_TOKEN", "NO_TOKEN"}:
        return 4032, Reason.needs_nous_auth, "Sign in to use connectors."
    return 4030, Reason.forbidden_scope, "Connector access is not permitted for this account."
