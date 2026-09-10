"""Validated webhook destinations live in the owner's admitted input, not a TTL cache."""
from copy import deepcopy
import hashlib
import json

from hermes_state_runtime import RuntimeStoreError


def route_digest(adapter, chat_id):
    adapter._reload_dynamic_routes()
    route_name = chat_id.split(':', 2)[1]
    route = adapter._routes.get(route_name)
    if route is None:
        raise RuntimeStoreError('not_found')
    return hashlib.sha256(json.dumps(route, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_destination(delivery):
    from gateway.platforms.webhook import _is_known_platform, _REPO_RE
    if not isinstance(delivery, dict) or set(delivery) != {'deliver', 'deliver_extra'}:
        raise RuntimeStoreError('invalid_params')
    target, extra = delivery['deliver'], delivery['deliver_extra']
    if not isinstance(target, str) or not isinstance(extra, dict):
        raise RuntimeStoreError('invalid_params')
    if target == 'github_comment':
        repo, number = extra.get('repo'), extra.get('pr_number')
        if not isinstance(repo, str) or not _REPO_RE.fullmatch(repo) or not str(number).isdigit() or int(number) <= 0:
            raise RuntimeStoreError('invalid_params')
    elif target != 'log' and not _is_known_platform(target):
        raise RuntimeStoreError('invalid_params')
    return deepcopy(delivery)


def retained_destination(adapter, chat_id):
    """Readback also checks today's connector and authorization; retained data is not a grant."""
    from gateway.session_envelope import restore_native, _validate_native
    runner = adapter.gateway_runner or getattr(adapter._message_handler, '__self__', None)
    authority = getattr(runner, 'session_authority', None)
    if authority is None:
        return validate_destination(adapter._delivery_info.get(chat_id))
    rows = authority.db._read_all("""SELECT payload_json FROM session_admissions
        WHERE json_extract(payload_json, '$.native_text_v1.source.chat_id')=?
          AND json_extract(payload_json, '$.native_text_v1.source.platform')='webhook'""", (chat_id,))
    if len(rows) != 1:
        raise RuntimeStoreError('not_found')
    payload = json.loads(rows[0]['payload_json'])
    envelope = payload['native_text_v1']
    event = restore_native(payload, runner)
    _validate_native(runner, event, envelope.get('provenance'))
    if runner._adapter_for_source(event.source) is not adapter:
        raise RuntimeStoreError('permission_denied')
    if envelope.get('webhook_route') != route_digest(adapter, chat_id):
        raise RuntimeStoreError('admission_conflict')
    return validate_destination(envelope.get('webhook_delivery'))
