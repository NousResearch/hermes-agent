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

from pathlib import Path

REDACTED_SECRET_SOURCE_VALUE = "«redacted:secret-source»"

# Mundane values a bulk source also supplies (``DEBUG=true``, ``LOG_LEVEL=info``) must not become
# unanchored substring scrubs over every later tool result, so a single-line value registers only
# when it is non-trivial and either looks like a credential or sits under a secret-named variable.
_MIN_VALUE_CHARS = 8

_VALUES_BY_HOME: dict[str, tuple[str, ...]] = {}


def _worth_registering(name: str, value: str) -> bool:
    from agent.redact import _key_has_secret_keyword, _looks_like_opaque_credential

    stripped = value.strip()
    if len(stripped) < _MIN_VALUE_CHARS:
        return False
    return "\n" in stripped or _looks_like_opaque_credential(stripped) or _key_has_secret_keyword(name)


def publish_secret_source_values(hermes_home: str | Path, values: dict[str, str]) -> None:
    """Replace ``hermes_home``'s registered values with the ones its sources just applied."""
    from hermes_constants import hermes_home_key

    exact: dict[str, None] = {}
    for name, value in values.items():
        if isinstance(value, str) and _worth_registering(name, value):
            exact[value] = None
            exact[value.strip()] = None  # what a shell prints after word splitting / echo
    # Longest first, so a value that is a substring of another never shadows its superstring.
    _VALUES_BY_HOME[hermes_home_key(hermes_home)] = tuple(sorted(exact, key=len, reverse=True))


def redact_secret_source_values(text: str) -> str:
    """Exact-substring scrub of the current profile's applied secret-source values."""
    if not _VALUES_BY_HOME or not isinstance(text, str) or not text:
        return text
    from hermes_constants import hermes_home_key

    for value in _VALUES_BY_HOME.get(hermes_home_key(), ()):
        if value in text:
            text = text.replace(value, REDACTED_SECRET_SOURCE_VALUE)
    return text
