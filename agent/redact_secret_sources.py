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
import threading
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

# Line-oriented output (``env | sort``, ``grep -v key``, ``sed -n 2,4p`` of a key file) splits a PEM
# body from its armor and from the rest of the value, so only the exact line is left to recognise.
# Only a PEM-armored value contributes lines, and only lines of bare base64 material: a line of a
# service-account JSON (``"type": "service_account",``) or a short trailer would become an unanchored
# scrub of ordinary text. Body lines are 64 (PEM) or 70 (OpenSSH) chars; the last one can be ~20.
_PEM_ARMOR = "-----BEGIN "
_SECRET_LINE_RE = re.compile(r"[A-Za-z0-9+/=]{16,}")

# home key -> its registered exact values, longest first; or, until a scrub first reads it, the applied
# ``{name: value}`` map. Classifying imports agent.redact, which latches security.redact_secrets at import,
# and startup publishes from load_hermes_dotenv BEFORE hermes_cli/main.py bridges that setting.
_VALUES_BY_HOME: dict[str, tuple[str, ...] | dict[str, str]] = {}
_LOCK = threading.Lock()


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


def _multiline_fragments(value: str) -> list[str]:
    """The joined form (what a text input or ``tr -d '\\n'`` leaves) and, for a PEM value, each body line."""
    if "\n" not in value and "\r" not in value:
        return []
    joined = value.replace("\r", "").replace("\n", "")
    if _PEM_ARMOR not in value:
        return [joined]
    return [joined, *(ln for ln in (raw.strip() for raw in value.splitlines()) if _SECRET_LINE_RE.fullmatch(ln))]


def _exact_forms(values: dict[str, str]) -> tuple[str, ...]:
    exact: dict[str, None] = {}
    for name, value in values.items():
        if _worth_registering(name, value):
            exact[value] = None
            exact[value.strip()] = None  # what a shell prints after word splitting / echo
            # How a tool that serialises its result as JSON (MCP, read_file of a .json) carries it.
            exact[json.dumps(value)[1:-1]] = None
            exact[json.dumps(value, ensure_ascii=False)[1:-1]] = None
            exact.update(dict.fromkeys(_multiline_fragments(value)))
    # Longest first, so a value that is a substring of another never shadows its superstring.
    return tuple(sorted(exact, key=len, reverse=True))


def publish_secret_source_values(hermes_home: str | Path, values: dict[str, str]) -> None:
    """Replace ``hermes_home``'s registered values with the ones its sources just applied (classified by
    the first scrub that reads them, see ``_VALUES_BY_HOME``)."""
    from hermes_constants import hermes_home_key

    applied = {name: value for name, value in values.items() if isinstance(value, str)}
    key = hermes_home_key(hermes_home)
    with _LOCK:
        if applied:
            _VALUES_BY_HOME[key] = applied
        else:  # revoked or never supplied: unregister, and keep the empty-registry fast path
            _VALUES_BY_HOME.pop(key, None)


def _registered(key: str) -> tuple[str, ...]:
    entry = _VALUES_BY_HOME.get(key, ())
    while isinstance(entry, dict):
        exact = _exact_forms(entry)  # outside the lock: the first call imports agent.redact
        with _LOCK:
            if _VALUES_BY_HOME.get(key) is entry:  # else a newer publish landed meanwhile: classify that one
                if exact:
                    _VALUES_BY_HOME[key] = exact
                else:
                    _VALUES_BY_HOME.pop(key, None)
                return exact
            entry = _VALUES_BY_HOME.get(key, ())
    return entry


def redact_secret_source_values(text: str) -> str:
    """Exact-substring scrub of the current profile's applied secret-source values."""
    if not _VALUES_BY_HOME or not isinstance(text, str) or not text:
        return text
    from hermes_constants import hermes_home_key

    for value in _registered(hermes_home_key()):
        if value in text:
            text = text.replace(value, REDACTED_SECRET_SOURCE_VALUE)
    return text
