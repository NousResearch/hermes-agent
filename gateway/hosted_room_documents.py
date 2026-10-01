"""Bounded, immutable document inputs for proof-v2 room attempts.

Generic attachments remain unsupported: this extension transfers file/PDF bytes,
never images, remote paths, or executable custody handles.
"""
import base64
import binascii
import hashlib

DOCUMENT_MAX_BYTES = 5_000_000
DOCUMENT_BATCH_MAX_BYTES = 6_000_000
DOCUMENT_MAX_COUNT = 8
DOCUMENT_HTTP_MAX_BYTES = ((DOCUMENT_BATCH_MAX_BYTES + 2) // 3) * 4 + 1_000_000
DOCUMENT_CAPABILITY = {'version': 1, 'kinds': ['file', 'pdf'], 'max_count': DOCUMENT_MAX_COUNT,
                       'max_file_bytes': DOCUMENT_MAX_BYTES, 'max_batch_bytes': DOCUMENT_BATCH_MAX_BYTES}


def manifest(value, *, member_id=None, capability=None):
    from gateway.hosted_room_driver import validate_bound_task_manifest
    from gateway.hosted_room_attachments import _SHA256_RE
    if not isinstance(value, list) or not 0 < len(value) <= DOCUMENT_MAX_COUNT:
        raise ValueError('document batch exceeded its bound')
    if any(not isinstance(item, dict) or set(item) != {
            'event_id', 'attachment_id', 'recipient_member_id', 'kind', 'name', 'mime', 'size', 'sha256'}
            for item in value):
        raise ValueError('invalid document manifest')
    bound = validate_bound_task_manifest([
        {k: v for k, v in item.items() if k not in {'sha256', 'recipient_member_id'}} for item in value])
    for item in value:
        if (item['kind'] not in {'file', 'pdf'} or item['mime'].startswith(('image/', 'audio/', 'video/'))
                or type(item['size']) is not int or not 0 < item['size'] <= DOCUMENT_MAX_BYTES
                or not isinstance(item['sha256'], str) or not _SHA256_RE.fullmatch(item['sha256'])
                or not isinstance(item['recipient_member_id'], str) or not item['recipient_member_id']
                or (member_id is not None and item['recipient_member_id'] != member_id)):
            raise ValueError('unsupported or mismatched document input')
    if sum(item['size'] for item in value) > DOCUMENT_BATCH_MAX_BYTES:
        raise ValueError('document batch exceeded its bound')
    if capability is not None and (len(value) > capability['max_count']
            or any(item['size'] > capability['max_file_bytes'] for item in value)
            or sum(item['size'] for item in value) > capability['max_batch_bytes']):
        raise ValueError('document batch exceeds receiver capability')
    return [{**item, 'sha256': source['sha256'], 'recipient_member_id': source['recipient_member_id']}
            for item, source in zip(bound, value)]


def decode_batch(inputs, value):
    """Verify the entire transfer before publishing even the first private copy."""
    if not isinstance(value, list) or len(value) != len(inputs):
        raise ValueError('document transfer does not match manifest')
    documents = []
    for item, encoded in zip(inputs, value):
        if not isinstance(encoded, str) or len(encoded) != ((item['size'] + 2) // 3) * 4:
            raise ValueError('document transfer size mismatch')
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('invalid document transfer') from exc
        if len(data) != item['size'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('document digest mismatch')
        documents.append({'name': item['name'], 'data': data, 'sha256': item['sha256'], 'size': item['size']})
    from gateway.session_ingress_media import validate_media_batch_size
    validate_media_batch_size(item['size'] for item in inputs)
    return documents


def check_capability(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != set(DOCUMENT_CAPABILITY)
            or type(value['version']) is not int or value['version'] != 1 or value['kinds'] != ['file', 'pdf']
            or any(type(value[key]) is not int or not 0 < value[key] <= DOCUMENT_CAPABILITY[key]
                   for key in ('max_count', 'max_file_bytes', 'max_batch_bytes'))):
        raise ValueError('unsupported document capability')
    return dict(value)


def advertised_capability(adapter):
    from gateway.session_authorities import active_authority
    from gateway.platforms.base import get_inbound_media_max_bytes
    if active_authority(getattr(adapter, 'gateway_runner', None)) is None:
        return None
    limit = get_inbound_media_max_bytes()
    if limit <= 0:
        return dict(DOCUMENT_CAPABILITY)
    return {**DOCUMENT_CAPABILITY, 'max_file_bytes': min(DOCUMENT_MAX_BYTES, limit),
            'max_batch_bytes': min(DOCUMENT_BATCH_MAX_BYTES, limit)}
