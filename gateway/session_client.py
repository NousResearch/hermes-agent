"""The client a local session's creator declared itself as (``session.create`` ``client``).

A third-party client names itself (``{"name": "my-client", "version": "1.2.0", "kind": "desktop"}``) so
a viewer or a plugin can tell which client created the session. ``kind`` says what sort of client it is;
one outside ``CLIENT_KINDS`` is not an error (a newer client may name a sort this owner does not know):
the label is kept and read without it. It is a self-declared label, never an
authentication: no policy, permission, toolset, approval, prompt or telemetry path reads it.

It rides the frozen creation request (``LocalSessionPolicy.request_json``) as sent, so it needs no new
column, receipt key or policy field: an older owner restores the session unchanged, and it is part of
the exact-retry identity (a retry naming another client is a different request).
"""
from __future__ import annotations

import json
import re

_NAME = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}')
_VERSION = re.compile(r'[!-~]{1,64}')  # printable ASCII, no whitespace
CLIENT_KINDS = frozenset({'desktop', 'webapp', 'tui', 'mobile'})


def client_problem(value) -> str | None:
    """Why *value* is not a client label (``None`` when it is one)."""
    if not isinstance(value, dict) or 'name' not in value or not set(value) <= {'name', 'version', 'kind'}:
        return 'must be an object with a name and an optional version and kind'
    if not isinstance(value['name'], str) or not _NAME.fullmatch(value['name']):
        return 'name must be 1-64 characters of a-z, 0-9, ".", "_" or "-", starting with a letter or digit'
    if 'version' in value and (not isinstance(value['version'], str) or not _VERSION.fullmatch(value['version'])):
        return 'version must be 1-64 printable ASCII characters without spaces'
    if 'kind' in value and (not isinstance(value['kind'], str) or not 1 <= len(value['kind']) <= 32):
        return 'kind must be a string of 1-32 characters'
    return None


def client_kind(value) -> str | None:
    """The kind *value* (a valid label) names when this owner knows it, else ``None``."""
    kind = value.get('kind')
    return kind if kind in CLIENT_KINDS else None


def normalized_client(value) -> dict:
    """A copy of the valid label *value* without a kind this owner does not know."""
    return {key: item for key, item in value.items() if key != 'kind' or item in CLIENT_KINDS}


def policy_client(policy) -> dict | None:
    """A copy of the client label *policy*'s creation request carries, or ``None`` (none was sent,
    no local policy, or a stored value that is not a valid label)."""
    if policy is None:
        return None
    try:
        value = json.loads(policy.request_json).get('client')
    except (AttributeError, TypeError, ValueError):
        return None
    return normalized_client(value) if client_problem(value) is None else None
