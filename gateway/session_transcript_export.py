"""The filtered Desktop export, separate from the raw /save backup contract."""
import asyncio
import json

from gateway.session_local_recovery import transcript_target
from gateway.session_persistence import authority_persistence
from hermes_cli.session_export import default_save_filename, render_session_for_save
from hermes_cli.session_transcript import (
    TranscriptTooLargeError,
    project_transcript,
    redact_transcript,
    validate_options,
)
from persistence.conversation.messages import ResumeTooLarge, TranscriptRead, TranscriptView
from persistence.core.errors import AuthorityRefusal


def _workspace_roots(authority, ref, home):
    from gateway.session_policy import policy_for_source
    from tui_gateway import server

    live = authority.sessions.get(ref.session_id)
    policy = policy_for_source(authority.runner, live.source) if live is not None and live.source is not None else None
    if not policy or not policy.cwd:
        return ()
    staged = server._session_home_dir({'cwd': policy.cwd, 'profile_home': str(home)}, 'attachments')
    return (staged,) if staged != home / 'attachments' else ()


def _read_rows(store, target):
    from storage_limits import resolved_max_export_messages

    with store.messages.read_snapshot():
        session = store.sessions.get(target)
        if session is None:
            raise AuthorityRefusal('not_found')
        limit = resolved_max_export_messages()
        if limit:
            store.messages.assert_resume_safe(target, max_messages=limit)
        request = TranscriptRead(view=TranscriptView.DISPLAY, include_lineage=True)
        rows = store.messages.read_transcript(target, request)
    return {'id': target, 'title': session.title, 'started_at': session.started_at, 'model': session.model,
            'messages': [row.as_wire_row() for row in rows]}


def _render(authority, ref, target, options, redact, home):
    from hermes_cli.session_transcript import MAX_SERIALIZED_BYTES

    try:
        data = _read_rows(authority_persistence(authority), target)
        transcript = project_transcript(data, options, home=home, trusted_roots=_workspace_roots(authority, ref, home))
    except (ResumeTooLarge, TranscriptTooLargeError) as exc:
        raise AuthorityRefusal('export_too_large') from exc
    if redact:
        transcript = redact_transcript(transcript)
    result = {
        'format': 'json',
        'filename': default_save_filename(target, 'json'),
        'content': render_session_for_save(transcript, 'json'),
        'transcript_version': 1,
    }
    if len(json.dumps(result, ensure_ascii=True).encode('utf-8')) > MAX_SERIALIZED_BYTES:
        raise AuthorityRefusal('export_too_large')
    return result


async def export_transcript(authority, ref, options, redact, home):
    try:
        validate_options(options)
    except ValueError:
        raise AuthorityRefusal('invalid_params') from None
    target = transcript_target(authority, ref)
    return await asyncio.to_thread(_render, authority, ref, target, options, redact, home)
