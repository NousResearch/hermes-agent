"""Explicit document-output consent and exact scope for a canonical peer attempt."""
import hashlib
import json

from gateway.hosted_room_documents import DOCUMENT_CAPABILITY
from gateway.hosted_room_artifacts import RoomArtifactScope, validate_terminal_artifact_manifest

OUTPUT_FEATURE = 'document-output-v1'
OUTPUT_CAPABILITY = dict(DOCUMENT_CAPABILITY)


def output_contract(value):
    if value is None:
        return None
    if value != OUTPUT_CAPABILITY or not isinstance(value, dict):
        raise ValueError('unsupported peer document output contract')
    return dict(value)


def output_scope(dispatch):
    return RoomArtifactScope.from_mapping({key: getattr(dispatch, key) for key in (
        'room_id', 'task_id', 'execution_generation', 'member_id', 'target_profile',
        'home_install_id', 'target_install_id', 'authority_gateway_id', 'authority_epoch')})


def output_manifest(value):
    items = validate_terminal_artifact_manifest(value)
    if (len(items) > OUTPUT_CAPABILITY['max_count'] or
            sum(item['size'] for item in items) > OUTPUT_CAPABILITY['max_batch_bytes'] or
            any(item['kind'] not in OUTPUT_CAPABILITY['kinds'] or item['size'] > OUTPUT_CAPABILITY['max_file_bytes']
                or item['mime'].startswith(('image/', 'audio/', 'video/')) for item in items)):
        raise ValueError('peer output exceeds its frozen document contract')
    return items


def dispatch_digest(dispatch):
    """Commit the complete immutable peer attempt, excluding only transfer bytes."""
    return hashlib.sha256(json.dumps(dispatch.as_mapping(), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()
