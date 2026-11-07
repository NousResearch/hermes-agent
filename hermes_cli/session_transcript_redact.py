"""Format-aware redaction of JSON embedded in transcript strings."""
import json
import re


_MAX_NESTING = 64
_MAX_JSON_LENGTH = 32 * 1024 * 1024
_MARKER = '[redaction unavailable]'
_STANDALONE_URI_RE = re.compile(r'(?i)(?:https?|wss?|ftp)://\S+\Z')
# A decoded quote is credential data. The raw-text userinfo pattern excludes it
# so a JSON boundary such as `"https://a.com","u:pw@y"` is not one credential.
_DECODED_USERINFO_RE = re.compile(r'//([^/\s?#]*)@')


def _mask_decoded_quoted_userinfo(value):
    """Mask colon-form userinfo the raw-text redactor skipped because of a quote."""
    if not isinstance(value, str) or '"' not in value or not _STANDALONE_URI_RE.fullmatch(value):
        return value
    match = _DECODED_USERINFO_RE.search(value)
    if match is None or '"' not in match.group(1) or ':' not in match.group(1):
        return value
    return value[:match.start()] + '//***:***@' + value[match.end():]


def _redact_json_assignment(key, value):
    from agent.redact import _JSON_FIELD_RE, _mask_token, _should_redact_assignment

    # Match only the key; masking escaped JSON can leave a dangling backslash.
    field = _JSON_FIELD_RE.fullmatch(json.dumps(key) + ': "value"')
    if field and _should_redact_assignment(key, value, check_keyword=False):
        return _mask_token(value)
    return value


def redact_json_text(text, redact_text, depth=0):
    """Return None for non-JSON; otherwise preserve all unchanged lexical spans."""
    stripped = text.lstrip()
    if not stripped or stripped[0] not in '[{"':
        return None
    if depth >= _MAX_NESTING or len(text) > _MAX_JSON_LENGTH:
        return json.dumps(_MARKER)
    try:
        json.loads(text)
    except RecursionError:
        return json.dumps(_MARKER)
    except (ValueError, TypeError):
        return None
    nesting = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            nesting += 1
            if nesting > _MAX_NESTING:
                return json.dumps(_MARKER)
        elif char in ']}':
            nesting -= 1

    output = []
    pos = 0
    key = None
    key_end = 0
    try:
        while pos < len(text):
            if text[pos] != '"':
                output.append(text[pos])
                pos += 1
                continue
            start = pos
            value, pos = json.decoder.scanstring(text, pos + 1, True)
            following = pos
            while following < len(text) and text[following] in ' \t\r\n':
                following += 1
            is_key = text[following:following + 1] == ':'
            if is_key:
                key, key_end = value, pos
            nested = None if is_key else redact_json_text(value, redact_text, depth + 1)
            replacement = _mask_decoded_quoted_userinfo(redact_text(value)) if nested is None else nested
            if not is_key and key is not None and text[key_end:start].strip() == ':':
                replacement = _redact_json_assignment(key, replacement)
            output.append(text[start:pos] if replacement == value else json.dumps(replacement, ensure_ascii=True))
        return ''.join(output)
    except (ValueError, TypeError, RecursionError):
        return json.dumps(_MARKER)
