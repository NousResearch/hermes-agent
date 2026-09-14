"""One public-input normalizer shared by authority submission and private preparation."""
from gateway.config import Platform
from gateway.session_finite import admit_finite
from gateway.session_ingress_media import admit_attachments
from gateway.session_surface import admit_surface


def normalize_submission_payload(authority, actor, request):
    payload = {'text': request.payload['text'], **admit_finite(request.payload),
               **admit_surface(request.payload), **admit_attachments(request.payload.get('attachments'))}
    source = authority.sessions[request.ref.session_id].source
    if source is not None and source.platform == Platform.LOCAL and source.user_id != actor.subject:
        payload['local_operator_v1'] = {'profile_id': authority.profile_id,
            'session_id': request.ref.session_id, 'principal_id': actor.subject}
    return payload
