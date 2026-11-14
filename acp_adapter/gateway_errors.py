"""Translate expected owner refusals at ACP entry points, not inside recovery logic."""
from contextlib import suppress
from functools import wraps
from inspect import signature

import acp

from hermes_cli.gateway_client import (
    GatewayClientError, GatewayRPCError, GatewayUnavailableError,
    _bounded_error_data, _bounded_reason,
)


class GatewayRequestError(acp.RequestError, GatewayClientError):
    """Keep local reason comparisons intact; give editors the recovery guidance too."""

    def to_error_obj(self):
        return {'code': self.code, 'message': f"{self.data['reason']}: {self.data['recovery']}",
                'data': self.data}


class GatewayRPCRequestError(GatewayRequestError, GatewayRPCError):
    """A definite owner refusal stays distinct from an ambiguous transport failure."""

    def __init__(self, code, message, data):
        Exception.__init__(self, message)
        self.code, self.data = code, data


def gateway_errors(method):
    """Keep owner reasons/data visible through the SDK's RequestError-only boundary."""
    params = signature(method)

    @wraps(method)
    async def dispatch(self, *args, **kwargs):
        try:
            return await method(self, *args, **kwargs)
        except acp.RequestError:
            raise
        except (GatewayClientError, OSError, TimeoutError) as exc:
            session_id = params.bind(self, *args, **kwargs).arguments.get('session_id')
            error = request_error(exc, session_id)
            if method.__name__ == 'cancel' and self._conn is not None:
                # session/cancel is a notification: it has no JSON-RPC error reply.
                with suppress(Exception):
                    await self._conn.session_update(session_id=session_id,
                        update=acp.update_agent_message_text(f'Cancel refused: {error}\n'))
            raise error from exc

    return dispatch


def request_error(exc, session_id=None, *, recovery=None):
    text = str(exc) or ('gateway_timeout' if isinstance(exc, TimeoutError) else 'gateway_unavailable')
    data = _bounded_error_data(exc.data) if isinstance(exc, GatewayRPCError) else {}
    if isinstance(exc, GatewayRPCError) and exc.code is not None:
        data['gatewayCode'] = exc.code
    reason = _bounded_reason(text.partition(':')[0])
    if reason is None:
        reason = (getattr(exc, 'reason_code', None) if isinstance(exc, GatewayUnavailableError)
                  else 'gateway_unavailable')
        reason = _bounded_reason(reason) or 'gateway_unavailable'
    data['reason'] = reason
    if session_id is not None:
        data['sessionId'] = session_id
    if recovery is None:
        recovery = recovery_text(reason, text)
    data['recovery'] = recovery
    code = {'invalid_params': -32602, 'not_found': -32002}.get(reason, -32001)
    error_type = GatewayRPCRequestError if isinstance(exc, GatewayRPCError) else GatewayRequestError
    return error_type(code, text, data)


def recovery_text(reason, text):
    if reason == 'unknown_execution':
        recovery = (text if text.startswith('unknown_execution:') else
                    'The turn outcome is unknown; do not resend accepted work. Inspect the session and '
                    'use /discard <admission-id> only after checking the lost turn; then continue.')
        return recovery
    return {
        'queue_full': ('The session queue is full; this input was not admitted. Inspect the pending queue '
                       'and cancel unwanted queued inputs. If a lost turn blocks it, explicitly /discard '
                       'that admission after checking its outcome; do not resend accepted work.'),
        'session_busy': ('This control requires an idle session. Wait for the active turn to finish, '
                         'or cancel it, then retry.'),
        'not_found': 'Reload an existing session or explicitly create a new session.',
    }.get(reason, text)
