"""Exact-value redaction of values applied from external secret sources (Bitwarden, 1Password,
command sources, ...).

The shape-based passes in ``agent.redact`` cannot recognise an arbitrary vault value (a
``DATABASE_URL``, an opaque field with no vendor prefix), so ``hermes_cli.env_loader`` publishes
each home's applied values here whenever it publishes that home's snapshot, and every
``redact_sensitive_text`` caller plus the tool-result builder scrub them by exact substring.

Keyed by ``hermes_home_key`` so a multiplex gateway never scrubs profile B's output with profile
A's values (which would also confirm to B that the bytes exist). A publish REPLACES the home's set:
the size tracks the configured sources instead of growing with every re-pull, and nothing is
evicted while it is still applied. Memory only: never persisted or logged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REDACTED_SECRET_SOURCE_VALUE = "«redacted:secret-source»"

# A bulk source also supplies ordinary config (``AWS_REGION=eu-central-1``, ``PROJECT_DIR=/srv/app``),
# and a registered value becomes an unanchored substring scrub over every later tool result and file
# read of the profile — a masked path breaks navigation. So a value registers only when its variable
# is secret-named (the redactor's own key predicate), it has a credential shape the redactor already
# masks, or it is multi-line (keys, certificates, service-account JSON). Values under 8 characters
# never register: a short secret-named value echoed bare is left to the ``KEY=VALUE`` pattern pass.
_MIN_VALUE_CHARS = 8
# The name test matches settings ABOUT credentials too (``AUTH_MODE=password``, ``AUTH_PROVIDER=localhost``,
# ``SECRET_BACKEND=bitwarden``, ``API_KEY_HEADER=Authorization``); registering those would scrub the
# word or host from every later result. Under a secret name, a single dictionary-style word or a
# lowercase hostname is configuration, not the credential (a value with a credential shape still
# registers through the shape arm). ``*PASSWORD``/``*PASSWD`` names keep any value: a passphrase is
# often exactly a lowercase word.
_PLAIN_WORD_OR_HOST_RE = re.compile(r"[A-Za-z][a-z]*|[a-z0-9-]+(?:\.[a-z0-9-]+)+")

_VALUES_BY_HOME: dict[str, tuple[str, ...]] = {}


def _worth_registering(name: str, value: str) -> bool:
    from agent.redact import (_has_password_keyword, _is_glued_credential_env, _key_has_secret_keyword,
                              redact_credential_shapes)

    stripped = value.strip()
    if len(stripped) < _MIN_VALUE_CHARS:
        return False
    if "\n" in stripped or redact_credential_shapes(stripped, force=True) != stripped:
        return True
    if _has_password_keyword(name) or _is_glued_credential_env(name):  # PGPASSWORD, MYSQL_PWD, SSHPASS
        return True
    return _key_has_secret_keyword(name) and not _PLAIN_WORD_OR_HOST_RE.fullmatch(stripped)


def publish_secret_source_values(hermes_home: str | Path, values: dict[str, str]) -> None:
    """Replace ``hermes_home``'s registered values with the ones its sources just applied."""
    from hermes_constants import hermes_home_key

    exact: dict[str, None] = {}
    for name, value in values.items():
        if isinstance(value, str) and _worth_registering(name, value):
            exact[value] = None
            exact[value.strip()] = None  # what a shell prints after word splitting / echo
            # How a tool that serialises its result as JSON (MCP, read_file of a .json) carries it.
            exact[json.dumps(value)[1:-1]] = None
            exact[json.dumps(value, ensure_ascii=False)[1:-1]] = None
    key = hermes_home_key(hermes_home)
    if not exact:  # revoked or never supplied: unregister, and keep the empty-registry fast path
        _VALUES_BY_HOME.pop(key, None)
        return
    # Longest first, so a value that is a substring of another never shadows its superstring.
    _VALUES_BY_HOME[key] = tuple(sorted(exact, key=len, reverse=True))


def redact_secret_source_values(text: str) -> str:
    """Exact-substring scrub of the current profile's applied secret-source values."""
    if not _VALUES_BY_HOME or not isinstance(text, str) or not text:
        return text
    from hermes_constants import hermes_home_key

    for value in _VALUES_BY_HOME.get(hermes_home_key(), ()):
        if value in text:
            text = text.replace(value, REDACTED_SECRET_SOURCE_VALUE)
    return text
