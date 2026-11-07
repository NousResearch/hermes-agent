"""Readable, explicitly filtered projection of a session snapshot."""
import base64
import binascii
import json
import os
import re
from pathlib import Path
from datetime import UTC, datetime

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_SERIALIZED_BYTES = 32 * 1024 * 1024


class TranscriptTooLargeError(ValueError):
    pass

def _image_data_url(value):
    if not isinstance(value, str):
        return None
    header, sep, encoded = value.partition(',')
    signatures = {'data:image/png;base64': b'\x89PNG\r\n\x1a\n',
                  'data:image/jpeg;base64': b'\xff\xd8\xff',
                  'data:image/gif;base64': b'GIF8', 'data:image/webp;base64': b'RIFF'}
    signature = signatures.get(header)
    if not sep or not signature:
        return None
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3) or len(encoded) * 3 // 4 - (len(encoded) - len(encoded.rstrip('='))) > MAX_IMAGE_BYTES:
        raise TranscriptTooLargeError('image budget exceeded')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return None
    if not raw.startswith(signature) or (header.endswith('webp;base64') and raw[8:12] != b'WEBP'):
        return None
    return value

OPTION_KEYS = frozenset({'include_tools', 'include_reasoning', 'include_images', 'include_videos',
                         'include_files', 'include_links', 'include_asides', 'collapse_details'})


def validate_options(options):
    if not isinstance(options, dict) or set(options) != OPTION_KEYS or any(type(v) is not bool for v in options.values()):
        raise ValueError('invalid transcript options')
    return dict(options)


def _timestamp(value):
    try:
        if type(value) in (int, float):
            return datetime.fromtimestamp(value, UTC).isoformat()
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            return parsed.isoformat() if parsed.tzinfo is not None else None
    except (ValueError, OverflowError, OSError):
        pass
    return None


def _opened_path(handle):
    if os.name == 'nt':
        import ctypes
        import msvcrt
        from ctypes import wintypes
        api = ctypes.WinDLL('kernel32', use_last_error=True).GetFinalPathNameByHandleW
        api.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
        api.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(32768)
        length = api(msvcrt.get_osfhandle(handle.fileno()), buf, len(buf), 0)
        if not length or length >= len(buf):
            return None
        value = buf.value
        if value.startswith('\\\\?\\UNC\\'):
            value = '\\\\' + value[8:]
        elif value.startswith('\\\\?\\'):
            value = value[4:]
        return Path(value).resolve(strict=True)
    fd = Path('/proc/self/fd') / str(handle.fileno())
    return fd.resolve(strict=True) if fd.exists() else None


def _managed_image(ref, roots):
    if not isinstance(ref, str) or ref.startswith(('\\\\', '//')):
        return None
    try:
        path = Path(ref)
        if not path.is_absolute():
            return None
        allowed = []
        for item in roots:
            try:
                root, owner = item if isinstance(item, tuple) else (item, item)
                root, owner = Path(root), Path(owner)
                if not path.is_relative_to(root) or '..' in path.parts or not root.is_relative_to(owner):
                    continue
                pinned = owner.resolve(strict=True)
                current = owner
                for component in path.relative_to(owner).parts:
                    current = current / component
                    stat = current.lstat()
                    if current.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
                        break
                else:
                    resolved = root.resolve(strict=True)
                    if resolved.is_relative_to(pinned):
                        allowed.append((resolved, pinned))
            except (OSError, ValueError):
                continue
        if not allowed or path.is_symlink():
            return None
        expected = path.resolve(strict=True)
        if not any(expected.is_relative_to(root) and expected != root and expected.is_relative_to(owner) for root, owner in allowed) or not path.is_file():
            return None
        with path.open('rb') as handle:
            if _opened_path(handle) != expected:
                return None
            if os.fstat(handle.fileno()).st_size > MAX_IMAGE_BYTES:
                raise TranscriptTooLargeError('image budget exceeded')
            raw = handle.read(MAX_IMAGE_BYTES + 1)
        if len(raw) > MAX_IMAGE_BYTES:
            raise TranscriptTooLargeError('image budget exceeded')
        for mime in ('png', 'jpeg', 'gif', 'webp'):
            result = _image_data_url('data:image/' + mime + ';base64,' + base64.b64encode(raw).decode('ascii'))
            if result:
                return result
    except TranscriptTooLargeError:
        raise
    except (OSError, ValueError):
        pass
    return None


def _content(content, options, roots):
    if isinstance(content, str):
        lines, parts = [], []
        leading = True
        for line in content.split('\n'):
            ref_match = re.fullmatch(r'''@(file|folder|image):(`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|[^\s]+)\s*''', line) if leading else None
            image_match = re.fullmatch(r'\[Image attached(?: at)?: (.+)\]', line)
            media_match = re.fullmatch(r'''\s*MEDIA:\s*([`"']?(?:/|[A-Za-z]:[\\/]).+?)\s*''', line)
            if ref_match:
                ref = ref_match.group(2).strip('`\"\'')
                ext = Path(ref).suffix.lower()
                category = ('image' if ref_match.group(1) == 'image' else
                            'file' if ref_match.group(1) == 'folder' else
                            'image' if ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp') else
                            'video' if ext in ('.mp4', '.mov', '.webm', '.mkv', '.avi') else 'file')
            elif image_match:
                ref, category = image_match.group(1), 'image'
            elif media_match:
                ref = media_match.group(1).strip('`"\'')
                ext = Path(ref).suffix.lower()
                category = 'image' if ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp') else 'video' if ext in ('.mp4', '.mov', '.webm', '.mkv', '.avi') else 'file'
            else:
                leading = False
                def inline_media(match):
                    ref = match.group(1).rstrip('.,;!?')
                    suffix = match.group(1)[len(ref):]
                    ext = Path(ref).suffix.lower()
                    category = 'image' if ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp') else 'video' if ext in ('.mp4', '.mov', '.webm', '.mkv', '.avi') else 'file'
                    if options['include_' + category + 's']:
                        parts.append({'kind': category, 'name': Path(ref).name or category, 'reference': ref,
                                      'local': True, 'unavailable': 'Reference only; local files may be unavailable on another computer'})
                    return suffix
                lines.append(re.sub(r'MEDIA:((?:/|[A-Za-z]:[\\/])[^\s]+)', inline_media, line))
                continue
            if options['include_' + category + 's']:
                parts.append({'kind': category, 'name': Path(ref).name or category, 'reference': ref,
                              'local': not ref.startswith(('https://', 'http://')),
                              'unavailable': 'Reference only; local files may be unavailable on another computer'})
        text = '\n'.join(lines)
        return text, parts
    texts, attachments = [], []
    if not isinstance(content, list):
        return '', []
    for part in content:
        if isinstance(part, str):
            text, found = _content(part, options, roots)
            texts.append(text)
            attachments.extend(found)
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get('type')
        if kind in ('text', 'input_text', 'output_text') and isinstance(part.get('text'), str):
            text, found = _content(part['text'], options, roots)
            texts.append(text)
            attachments.extend(found)
        elif kind in ('image_url', 'input_image', 'image', 'video_url', 'input_video', 'video', 'file', 'input_file'):
            category = 'image' if 'image' in kind else 'video' if 'video' in kind else 'file'
            if not options['include_' + category + 's']:
                continue
            ref = part.get(kind) or part.get('file') or part.get('url') or part.get('file_url') or part.get('path') or ''
            if isinstance(ref, dict):
                nested = ref
                ref = nested.get('url') or nested.get('path') or ''
            else:
                nested = {}
            if not isinstance(ref, str):
                ref = ''
            entry = {'kind': category, 'name': str(part.get('name') or part.get('filename') or nested.get('filename') or category)}
            if ref and not ref.startswith('data:'):
                entry['reference'] = ref
                entry['local'] = not ref.startswith(('https://', 'http://'))
            entry['unavailable'] = 'Image unavailable' if category == 'image' else 'Reference only; local files may be unavailable on another computer'
            if category == 'image' and (url := (_image_data_url(ref) or _managed_image(ref, roots))):
                entry.pop('unavailable')
                entry['mime_type'] = url[5:url.index(';')]
                entry['data_url'] = url
            attachments.append(entry)
    return '\n'.join(texts), attachments


def _redact_transcript_url(value, *, standalone=False):
    from agent.redact import redact_sensitive_text
    excess = {close: value.count(close) - value.count(opening)
              for close, opening in ((')', '('), (']', '['), ('}', '{'))}
    end = len(value)
    while end and not standalone:
        final = value[end - 1]
        if final in '.,;!?':
            end -= 1
        elif excess.get(final, 0) > 0:
            excess[final] -= 1
            end -= 1
        else:
            break
    url, suffix = value[:end], value[end:]
    head, question, rest = url.partition('?')
    if question:
        query, fragment, tail = rest.partition('#')
        parts = (piece.partition('=') for piece in re.split(r'([&;])', query))
        query = ''.join(key + '=***' if sep else key for key, sep, _ in parts)
        url = head + question + query + fragment + tail
    return redact_sensitive_text(url, force=True, redact_url_credentials=True) + suffix


def _redact_transcript_text(value):
    from agent.redact import redact_sensitive_text

    pieces = []
    end = 0
    pattern = re.compile(r"(?:https?|wss?|ftp)://", re.IGNORECASE)
    while match := pattern.search(value, end):
        pieces.append(redact_sensitive_text(value[end:match.start()], force=True,
                                            redact_url_credentials=True))
        markdown = value[max(0, match.start() - 2):match.start()] == ']('
        level = 0
        escaped = False
        url_end = match.end()
        while url_end < len(value):
            char = value[url_end]
            if char.isspace() or char in "\"'<>`":
                break
            if markdown:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '(':
                    level += 1
                elif char == ')':
                    if level == 0:
                        break
                    level -= 1
            url_end += 1
        pieces.append(_redact_transcript_url(value[match.start():url_end]))
        end = url_end
    pieces.append(redact_sensitive_text(value[end:], force=True, redact_url_credentials=True))
    return ''.join(pieces)


def _redact_decoded_json_text(value):
    # A decoded URI value has no enclosing JSON quotes or trailing prose.
    if re.fullmatch(r'(?:https?|wss?|ftp)://\S+', value, re.IGNORECASE):
        return _redact_transcript_url(value, standalone=True)
    return _redact_transcript_text(value)


def redact_transcript(value):
    if isinstance(value, str):
        from hermes_cli.session_transcript_redact import redact_json_text
        structured = redact_json_text(value, _redact_decoded_json_text)
        return _redact_transcript_text(value) if structured is None else structured
    if isinstance(value, list):
        return [redact_transcript(item) for item in value]
    if isinstance(value, dict):
        image_url = _image_data_url(value.get('data_url')) if value.get('kind') == 'image' else None
        result = {key: item if key == 'data_url' and image_url else redact_transcript(item)
                  for key, item in value.items()}
        # Redacted identifiers must not become identical placeholders and imply a false association.
        identity_keys = ('id', 'call_id', 'uid') if 'name' in value and 'arguments' in value else ()
        if value.get('role') == 'tool':
            identity_keys = ('tool_call_id', 'tool_call_uid')
        for key in identity_keys:
            if key in value and result[key] != value[key]:
                del result[key]
        return result
    return value


def _safe_tool_identity(value):
    return isinstance(value, str) and 0 < len(value) <= 512 and not any(ord(c) < 32 or ord(c) == 127 for c in value)


def _tool_calls(row):
    from agent.message_sanitization import coalesce_tool_call_id
    from hermes_state_identity import _tool_call_uid_map

    calls = []
    uids = _tool_call_uid_map(row)
    occurrences = {}
    for call in row['tool_calls']:
        if not isinstance(call, dict):
            continue
        key = coalesce_tool_call_id(call)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        uid = uids.get(key)
        if isinstance(uid, list):
            uid = uid[occurrence] if occurrence < len(uid) else None
        function = call.get('function') if isinstance(call.get('function'), dict) else call
        name, args = function.get('name'), function.get('arguments', '')
        if isinstance(name, str):
            item = {'name': name, 'arguments': args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}
            for field in ('id', 'call_id'):
                if _safe_tool_identity(call.get(field)):
                    item[field] = call[field]
            if _safe_tool_identity(uid):
                item['uid'] = uid
            calls.append(item)
    return calls


def _add_tool_fields(turn, row, role, options):
    if options['include_tools'] and isinstance(row.get('tool_calls'), list):
        calls = _tool_calls(row)
        if calls:
            turn['tool_calls'] = calls
    if options['include_tools'] and role == 'tool':
        from hermes_state_identity import _tool_call_uid_or_none
        for field, value in (('tool_call_id', row.get('tool_call_id')),
                             ('tool_call_uid', _tool_call_uid_or_none(row))):
            if _safe_tool_identity(value):
                turn[field] = value


def project_transcript(data, options, *, home=None, trusted_roots=()):
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    home = Path(home)
    roots = [(home / name, home) for name in ('images', 'image_cache', 'attachments', 'cache/images')]
    roots.extend((Path(root), Path(root).parent.parent) for root in trusted_roots)
    sid = str(data.get('id') or data.get('session_id') or '')
    messages = []
    image_bytes = 0
    for row in data.get('messages', []):
        if row.get('role') in ('system', 'developer') or (row.get('role') == 'tool' and not options['include_tools']):
            continue
        role = row.get('role') or 'unknown'
        kind = row.get('display_kind', row.get('kind'))
        if kind == 'aside' and not options['include_asides']:
            continue
        content, attachments = _content(row.get('content'), options, roots)
        for attachment in attachments:
            if url := attachment.get('data_url'):
                encoded = url.partition(',')[2]
                image_bytes += len(encoded) * 3 // 4 - len(encoded) + len(encoded.rstrip('='))
                if image_bytes > MAX_IMAGE_BYTES:
                    raise TranscriptTooLargeError('image budget exceeded')
        turn = {'role': role, 'speaker': str(row.get('name') or row.get('tool_name') or role),
                'text': content}
        if kind in ('message', 'aside', 'notice'):
            turn['kind'] = kind
        if attachments:
            turn['attachments'] = attachments
        if row.get('id') is not None:
            turn['id'] = str(row['id'])
        if stamp := _timestamp(row.get('timestamp')):
            turn['timestamp'] = stamp
        _add_tool_fields(turn, row, role, options)
        if options['include_reasoning']:
            reasoning = row.get('reasoning') or row.get('reasoning_content')
            if isinstance(reasoning, str) and reasoning:
                turn['reasoning'] = reasoning
        messages.append(turn)
    doc = {'version': 1, 'session_id': sid, 'title': str(data.get('title') or sid),
           'exported_at': datetime.now(UTC).isoformat(), 'options': dict(options),
           'messages': messages, 'warnings': [
               'Transient /btw conversations are not stored in normal session history and cannot be exported.'
           ] if options['include_asides'] else []}
    if stamp := _timestamp(data.get('started_at')):
        doc['created_at'] = stamp
    if isinstance(data.get('model'), str) and data['model']:
        doc['model'] = data['model']
    return doc
