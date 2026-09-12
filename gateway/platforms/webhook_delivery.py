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
    if (not isinstance(delivery, dict)
            or not {'deliver', 'deliver_extra'} <= set(delivery)
            or set(delivery) - {'deliver', 'deliver_extra', 'profile'}
            or (delivery.get('profile') is not None and not isinstance(delivery['profile'], str))):
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
    # Configuration extras may contain credentials or whole rendered payloads.
    # Persist only fields the selected sender actually consumes.
    fields = {'log': (), 'github_comment': ('repo', 'pr_number')}.get(
        target, ('chat_id', 'message_thread_id', 'thread_id'))
    result = {'deliver': target, 'deliver_extra': deepcopy({key: extra[key] for key in fields if key in extra})}
    # Omit the unbound default to preserve pre-profile receipt fingerprints.
    if delivery.get('profile'):
        result['profile'] = delivery['profile']
    return result


def snapshot_destination(adapter, delivery):
    from gateway.config import Platform

    result = validate_destination(delivery)
    if result['deliver'] not in {'log', 'github_comment'} and not result['deliver_extra'].get('chat_id'):
        profile = result.get('profile')
        with adapter._profile_scope(profile):
            home = adapter._delivery_config(profile).get_home_channel(Platform(result['deliver']))
        if home is None:
            raise RuntimeStoreError('not_found')
        result['deliver_extra']['chat_id'] = home.chat_id
    return result


def retained_destination(adapter, chat_id):
    """Readback also checks today's connector and authorization; retained data is not a grant."""
    from gateway.session_envelope import restore_native, _validate_native
    runner = adapter.gateway_runner or getattr(adapter._message_handler, '__self__', None)
    from gateway.session_authorities import active_authority, all_authorities
    if not all_authorities(runner):
        return validate_destination(adapter._delivery_info.get(chat_id))
    # The reply leg runs in the routed profile's scope; that ledger holds the destination. A
    # delivery id is unique per route, so an unscoped caller may search every served ledger.
    scoped = active_authority(runner)
    candidates = [scoped] if scoped is not None else all_authorities(runner)
    query = """SELECT payload_json FROM session_admissions
        WHERE json_extract(payload_json, '$.native_text_v1.source.chat_id')=?
          AND json_extract(payload_json, '$.native_text_v1.source.platform')='webhook'"""
    rows = [row for authority in candidates for row in authority.db._read_all(query, (chat_id,))]
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
