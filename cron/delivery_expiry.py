"""Opt-in expiry contract for local cron attachments.

Producers use ``*.expiry.html`` with a first-line JSON receipt. The required
suffix makes a missing receipt fail closed. HTML receipts also survive ordinary
renames. This is integrity/expiry metadata for trusted local producers, not a
signature or permission grant; the existing media path policy still applies.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PREFIX = b'<!-- hermes-delivery-expiry '
SUFFIX = b' -->\n'


class DeliveryExpired(ValueError):
    """An expiring attachment cannot safely be dispatched."""


def _utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('receipt timestamp must be UTC')
    return parsed


def attachment_deadline(media_files):
    """Validate each receipt and return the earliest UTC deadline, or None.

    Called inside the dispatch coroutine, not only in the scheduling thread:
    a busy gateway loop or a standalone fallback can start hours later.
    """
    deadline = None
    for raw_path, _voice in media_files or []:
        path = Path(raw_path)
        required = '.expiry.' in path.name
        if not required and path.suffix.lower() not in {'.html', '.htm'}:
            continue
        try:
            with path.open('rb') as stream:
                header = stream.readline(4097)
                if not header.startswith(PREFIX):
                    if required:
                        raise ValueError('required expiry receipt missing')
                    continue
                if len(header) > 4096 or not header.endswith(SUFFIX):
                    raise ValueError('malformed expiry receipt')
                data = json.loads(header[len(PREFIX):-len(SUFFIX)])
                if data['version'] != 1:
                    raise ValueError('unknown expiry receipt version')
                start, end = _utc(data['not_before']), _utc(data['expires_at'])
                now = datetime.now(timezone.utc)
                if not start <= now < end:
                    raise ValueError('expiry receipt is expired or not yet valid')
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != data['sha256']:
                    raise ValueError('expiry receipt content hash mismatch')
                deadline = min(deadline, end) if deadline else end
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise DeliveryExpired(f'attachment expiry validation failed for {path.name}: {exc}') from exc
    return deadline


async def dispatch_before_expiry(media_files, send):
    """Check at dispatch and bound the entire awaited send, including retry waits.

    Cancellation prevents a cooperative async transport from retrying after expiry.
    It cannot recall a request already accepted by the remote service. Transport
    implementations must not detach sends or swallow cancellation.
    """
    deadline = attachment_deadline(media_files)
    if deadline is None:
        return await send()
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise DeliveryExpired('attachment expired before dispatch')
    timeout = asyncio.timeout(remaining)
    try:
        async with timeout:
            return await send()
    except TimeoutError as exc:
        if not timeout.expired():
            raise
        raise DeliveryExpired('attachment expired during delivery; retry cancelled') from exc
