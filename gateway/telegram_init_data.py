"""Strict Telegram Mini App bot-token verification (host credential boundary)."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from urllib.parse import parse_qsl


class InitDataDenied(ValueError):
    """No untrusted payload in error text."""


class InitDataExpired(InitDataDenied):
    """The signature was valid but the bounded credential age elapsed."""


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InitDataDenied("invalid initialization data")
        result[key] = value
    return result


def verify_init_data(raw, bot_token, *, now, max_age=300, future_skew=30):
    """Return only verified actor/bot/date. Never return token or user payload.

    Telegram's bot-token algorithm includes every received field except hash,
    including the newer signature field. Ed25519 third-party validation is a
    different algorithm and is deliberately not used here.
    """
    try:
        if (type(max_age) is not int or not 30 <= max_age <= 900
                or type(future_skew) is not int or not 0 <= future_skew <= 30
                or not isinstance(raw, str) or not 1 <= len(raw.encode('utf-8')) <= 8192
                or not raw.isascii() or re.search(r"%(?![0-9A-Fa-f]{2})", raw)
                or not isinstance(bot_token, str)
                or not re.fullmatch(r"[1-9][0-9]{0,15}:[A-Za-z0-9_-]{20,128}", bot_token)):
            raise InitDataDenied("invalid initialization data")
        fields = _unique(parse_qsl(raw, keep_blank_values=True, strict_parsing=True,
                                   encoding="utf-8", errors="strict", max_num_fields=32))
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", k)
               or '\n' in v or '\r' in v or '\x00' in v for k, v in fields.items()):
            raise InitDataDenied("invalid initialization data")
        # Reject similarly-named identity/auth fields rather than allowing two interpretations.
        if set(fields) - {'auth_date', 'hash', 'signature', 'user', 'receiver', 'chat',
                          'query_id', 'chat_type', 'chat_instance', 'start_param', 'can_send_after'}:
            raise InitDataDenied("invalid initialization data")
        received = fields.pop('hash', '')
        if not re.fullmatch(r"[0-9a-f]{64}", received):
            raise InitDataDenied("invalid initialization data")
        check = '\n'.join(f'{k}={v}' for k, v in sorted(fields.items())).encode('utf-8')
        key = hmac.digest(b'WebAppData', bot_token.encode('utf-8'), 'sha256')
        expected = hmac.new(key, check, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, received):
            raise InitDataDenied("invalid initialization data")
        stamp = fields.get('auth_date', '')
        if not re.fullmatch(r'[1-9][0-9]{0,11}', stamp):
            raise InitDataDenied("invalid initialization data")
        stamp = int(stamp)
        if stamp < now - max_age:
            raise InitDataExpired("expired")
        if stamp > now + future_skew:
            raise InitDataDenied("invalid initialization data")
        def reject_constant(_):
            raise InitDataDenied("invalid initialization data")
        user = json.loads(fields.get('user', ''), object_pairs_hook=_unique,
                          parse_constant=reject_constant)
        if (not isinstance(user, dict) or type(user.get('id')) is not int
                or not 0 < user['id'] < 2**53 or user.get('is_bot', False) is not False):
            raise InitDataDenied("invalid initialization data")
        return {'actor': user['id'], 'bot_id': int(bot_token.split(':', 1)[0]), 'auth_date': stamp}
    except InitDataExpired:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise InitDataDenied("invalid initialization data") from None
