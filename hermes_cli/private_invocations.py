"""Opt-in plugin facade for native admitted origins, not saved source claims."""
import time
import json
import secrets
from typing import Any
from contextlib import contextmanager

from hermes_cli.private_invocation_store import PrivateReference
from tools.registry import _HostContextLease, _lease_value, get_current_host_context, _handler_context

_SEAL = object()


class _NativeOrigin:
    __slots__ = ('origin', 'active')

    def __init__(self, seal, origin):
        if seal is not _SEAL:
            raise PermissionError('stock native admission required')
        self.origin, self.active = origin, True

    def __repr__(self):
        return '<native invocation origin>'

    def __reduce__(self):
        raise TypeError('private capability is not serializable')


@contextmanager
def admitted_origin(runner, event, source, session_key):
    """Only called by the authorized/routed native post-admission dispatch seam."""
    from gateway.session_ingress_context import capture_provenance
    from gateway.session_identity import identity_of
    from tools.registry import bind_host_context
    provenance = capture_provenance(runner, event)
    identity = identity_of(source)
    if (provenance is None or identity is None or identity.transport is None
            or source.platform.value != 'slack' or source.is_bot
            or getattr(event, 'internal', False)
            or not all((source.scope_id, source.user_id, source.chat_id, event.message_id))
            or source.chat_type not in ('dm', 'group', 'channel')):
        yield
        return
    origin = {'platform': 'slack', 'workspace': source.scope_id,
              'author': source.user_id, 'channel': source.chat_id,
              'channelKind': source.chat_type, 'thread': source.thread_id,
              'message': event.message_id, 'profile': identity.runtime_profile,
              'transportProfile': identity.transport_profile, 'admission': secrets.token_hex(16), 'session': session_key,
              'provenance': provenance}
    capability = _NativeOrigin(_SEAL, origin)
    try:
        with bind_host_context(capability):
            yield
    finally:
        capability.active = False


class InvocationLease:
    __slots__ = ('_api', '_owner', '_reference', '_operation', '_anchor')

    def __init__(self, seal, api, reference, operation=None):
        if seal is not _SEAL:
            raise PermissionError('private invocation lease required')
        self._api, self._reference, self._operation = api, reference, operation
        self._owner = _HostContextLease(self)
        self._anchor = _handler_context.get()
        if (self._anchor is None or not self._anchor.active
                or self._anchor.thread_id != self._owner.thread_id
                or self._anchor.task is not self._owner.task):
            raise PermissionError('owned host handler execution required')

    def __repr__(self):
        return '<private invocation lease>'

    def __reduce__(self):
        raise TypeError('private capability is not serializable')

    def _check(self):
        if _lease_value(self._owner) is not self or not self._anchor.active:
            raise PermissionError('invocation belongs to another execution')
        if self._reference is None:
            current = get_current_host_context()
            if type(current) is not _NativeOrigin or not current.active:
                raise PermissionError('native admission lease closed')
            return current
        return None

    def retain(self, *, job_id, actions, expires_at):
        origin = self._check()
        if origin is None:
            raise PermissionError('retained delegations cannot renew or widen')
        return self._api._issuer._retain_native(origin, job_id, actions, expires_at)

    def call_mcp(self, server, tool, public_args, *, operation_id):
        from tools.mcp_tool_private import PrivateCall
        origin = self._check()
        if server not in self._api._ctx._mcp_allowlist(self._api._ctx.plugin_id):
            raise PermissionError('MCP server not granted to plugin')
        target = self._api._issuer.targets.get(server)
        if target is None or tool != 'call':
            raise PermissionError('only configured exact direct call supported')
        action = target['action']
        reference = self._reference
        if reference is None:
            reference = self.retain(job_id=operation_id, actions=[action], expires_at=int(time.time()) + 60)
        elif self._operation != (server, action, operation_id):
            raise PermissionError('rebound operation mismatch')
        request = PrivateCall(self._api._issuer, reference, server, operation_id, action)
        from tools.mcp_tool_handlers import _make_tool_handler
        # Arguments enter the normal final handler/result path. No bearer in kwargs.
        raw = _make_tool_handler(server, tool, 60)(public_args, _private_call=request)
        return self._api._ctx._mcp_envelope(raw)


class Invocations:
    def __init__(self, ctx):
        self._ctx = ctx

    @property
    def _issuer(self):
        issuer = getattr(self._ctx, '_private_invocation_issuer', None)
        if issuer is None:
            raise PermissionError('private invocation provider not registered')
        return issuer

    def register(self, issuer):
        """Trusted plugin setup: construct issuer from sanitized operator configuration.

        Plugin Python is process-trusted. This is not a hostile-plugin sandbox.
        """
        if getattr(self._ctx, '_private_invocation_issuer', None) is not None:
            raise ValueError('issuer already registered')
        self._ctx._private_invocation_issuer = issuer
        return self._ctx._track('private_invocation', self._ctx.plugin_id,
                               lambda: setattr(self._ctx, '_private_invocation_issuer', None))

    def current(self):
        origin = get_current_host_context()
        if type(origin) is not _NativeOrigin or not origin.active:
            return None
        self._issuer
        return InvocationLease(_SEAL, self, None)

    def rebind(self, reference, *, operation):
        if type(reference) is not PrivateReference or set(operation) != {'server', 'action', 'operationId'}:
            raise PermissionError('private reference and exact operation required')
        self._issuer.rebind(reference, operation['server'], operation['action'])
        lease = InvocationLease(_SEAL, self, reference, (operation['server'], operation['action'], operation['operationId']))
        return lease


def mcp_envelope(raw, cap):
    """Existing plugin ordinary-result normalization shared by the facade."""
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    truncated = len(raw) > cap
    if truncated:
        raw = raw[: cap] + "… [truncated]"
    try:
        parsed: Any = json.loads(raw)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and "error" in parsed:
        envelope: dict[str, Any] = {"ok": False, "error": parsed["error"]}
    elif isinstance(parsed, dict) and "result" in parsed:
        envelope = {"ok": True, "result": parsed["result"]}
        if "structuredContent" in parsed:
            envelope["structuredContent"] = parsed["structuredContent"]
    else:
        envelope = {"ok": True, "result": parsed if parsed is not None else raw}
    return {**envelope, "truncated": True} if truncated else envelope
