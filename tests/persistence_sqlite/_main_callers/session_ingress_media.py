"""Main's ``gateway/session_ingress_media.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

import json
from pathlib import Path
import re


_API_IMAGE_NAME = re.compile(r'api_[0-9a-f]{32}\.(png|jpg|gif|webp)')


_SHA256_NAME = re.compile(r'[0-9a-f]{64}')


def native_inputs_store():
    """The store's path as the profile names it, unresolved. The one value ``make_user_dir`` gets:
    ``resolve()`` erases a link planted below the Hermes home (``cache/documents`` -> elsewhere),
    and an elevated grant on the resolved path would land outside the home with no home fence."""
    from gateway.platforms.base import get_document_cache_dir
    return get_document_cache_dir() / 'native-inputs'


def native_inputs_root():
    """This profile's content-addressed store of retained native inputs (``<root>/<sha256>/<name>``),
    resolved: what the identity checks compare against."""
    return native_inputs_store().resolve()


_media_root = native_inputs_root


HOSTED_DOCUMENT_LINE = '\n[Shared attachment] file: '


_HOSTED_DOCUMENT_RE = re.compile(re.escape(HOSTED_DOCUMENT_LINE) + r'([^\n]+)\n')


def hosted_document_references(request_id, payload):
    """Retained document references a hosted admission's prompt names (``[]`` for other rows).

    Only content-addressed names under this profile's retained root count, and never an API
    image name: those are held by ``api_turn_v1`` in every status, so a member's prompt text
    can name one without gaining deletion authority over it."""
    if not str(request_id).startswith('hosted:') or not isinstance(payload.get('text'), str):
        return []
    root = _media_root()
    references = []
    for value in _HOSTED_DOCUMENT_RE.findall(payload['text']):
        path = Path(value)
        if (path.is_absolute() and path.parent.parent == root and _SHA256_NAME.fullmatch(path.parent.name)
                and not _API_IMAGE_NAME.fullmatch(path.name)):
            references.append({'path': value, 'sha256': path.parent.name})
    return references


def _held_media_paths(conn):
    # Project only references, not potentially large inline-image/history payloads; a live hosted
    # row's prompt is projected too, because its documents are held by text, not by a reference.
    rows = conn.execute("""SELECT status, request_id, json_extract(payload_json,
            '$.attachments_v1.media', '$.native_text_v1.media', '$.api_turn_v1.media'),
            CASE WHEN status!='terminal' AND request_id LIKE 'hosted:%'
                 THEN json_extract(payload_json, '$.text') END
            FROM session_admissions WHERE status!='terminal'
            OR json_type(payload_json, '$.api_turn_v1.media') IS NOT NULL""").fetchall()
    held = set()
    for status, request_id, encoded, text in rows:
        attachments, native, api = json.loads(encoded)
        # API images remain canonical history context after the turn completes.
        references = list(api or ())
        if status != 'terminal':
            references.extend(attachments or ())
            references.extend(native or ())
            references.extend(hosted_document_references(request_id, {'text': text}))
        held.update(reference['path'] for reference in references)
    return held


def _history_mentions(conn, references, after_id=0):
    """Candidate paths any transcript row (id > after_id) mentions raw or JSON-escaped, in ONE
    messages pass. Rows are prefiltered in SQL on the literal text before each candidate's digest
    directory, so only rows that could name a candidate reach Python's exact substring test."""
    # Matched by the content-addressed tail ``<sha256>/<name>`` with either separator, so every
    # spelling a note can carry (the host path, a sandbox backend's ``to_agent_visible_cache_path``)
    # counts as a mention.
    needles = {}
    for reference in references:
        raw = reference['path']
        for tail in (reference['sha256'] + sep + Path(raw).name for sep in ('/', '\\')):
            needles[tail] = needles[json.dumps(tail)[1:-1]] = raw
    prefixes = sorted({reference['sha256'] for reference in references})
    # The digests are ONE bound JSON array, never one ``instr`` term each: the statement's size
    # must not grow with the candidate count (SQLite's expression-depth limit fails at ~1000 terms).
    query = ('WITH digests(value) AS MATERIALIZED (SELECT value FROM json_each(?)) SELECT content FROM messages '
             'WHERE EXISTS (SELECT 1 FROM digests WHERE instr(content, digests.value))' + (' AND id>?' if after_id else ''))
    mentioned = set()
    for (content,) in conn.execute(query, (json.dumps(prefixes), *((after_id,) if after_id else ()))):
        text = content.decode('utf-8', 'replace') if isinstance(content, bytes) else str(content)
        mentioned.update(raw for needle, raw in needles.items() if needle in text)
    return mentioned
