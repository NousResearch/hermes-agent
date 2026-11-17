"""Exact-value redaction of the active profile's own ``.env`` secrets.

The shape passes in ``agent.redact`` only recognise credential *shapes* (vendor prefixes,
``KEY=value`` assignments, auth headers). An opaque key with no known prefix, printed without its
variable name (a provider error quoting it, an MCP server echoing its env, a JSON dump of a
request), passes all of them. The profile's ``.env`` says exactly which bytes are secret, so its
values are masked wherever they appear, in the raw and JSON-escaped forms
``agent.redact_secret_sources`` already derives for secret-source values.

Same registration rule as secret-source values (``_worth_registering``: secret-named, credential
shaped or multi-line) plus a 12-character floor, so ``LOG_LEVEL=debug`` or a short PIN never
becomes an unanchored substring scrub of ordinary text. Only a whole value is masked: a
deliberate slice of a secret passes through.

Keyed by ``hermes_home_key()``: one gateway serves many profiles, and profile A's values must
never mask (or confirm the existence of) bytes in profile B's output. Redaction runs on every log
record, so a home's ``.env`` is re-stat'ed at most every ``_RECHECK_SECONDS`` and re-parsed only
when its mtime/size changed. A frozen managed worker (safe mode / ``--ignore-user-config``) never
opens the profile's files. Memory only: never persisted or logged.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from agent.redact_secret_sources import _exact_forms, _registered as _secret_source_forms

MARKER = "«redacted-secret»"
MIN_VALUE_CHARS = 12
_RECHECK_SECONDS = 2.0

# home key -> (checked_at, (mtime_ns, size) or None, exact forms longest first)
_CACHE: dict[str, tuple[float, tuple[int, int] | None, tuple[str, ...]]] = {}
_LOCK = threading.Lock()


def _env_signature(env_path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(env_path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _load_forms(env_path: Path) -> tuple[str, ...]:
    from agent.secret_scope import load_env_file

    values = {k: v for k, v in load_env_file(env_path).items() if len(v.strip()) >= MIN_VALUE_CHARS}
    return _exact_forms(values)


def profile_env_secret_forms(hermes_home: str | Path | None = None) -> tuple[str, ...]:
    """Every registered exact form of ``hermes_home``'s (default: the active profile's) ``.env``
    secrets, longest first."""
    from agent.safe_worker_policy import worker_config_snapshot
    from hermes_constants import get_hermes_home, hermes_home_key

    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    key = hermes_home_key(home)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached is not None and now - cached[0] < _RECHECK_SECONDS:
        return cached[2]
    if worker_config_snapshot() is not None:  # checked per recheck, not per scrub: it parses JSON
        sig, forms = None, ()
    else:
        env_path = home / ".env"
        sig = _env_signature(env_path)
        if cached is not None and cached[1] == sig:
            forms = cached[2]
        else:
            forms = _load_forms(env_path) if sig is not None else ()
    with _LOCK:
        _CACHE[key] = (now, sig, forms)
    return forms


def clear_profile_env_secret_cache() -> None:
    """Forget every home's registered values (tests; the next scrub re-reads ``.env``)."""
    with _LOCK:
        _CACHE.clear()


def redact_profile_env_values(text: str) -> str:
    """Exact-substring scrub of the active profile's ``.env`` secret values."""
    if not isinstance(text, str) or not text:
        return text
    for value in profile_env_secret_forms():
        if value in text:
            text = text.replace(value, MARKER)
    return text


def registered_secret_forms() -> tuple[str, ...]:
    """The active profile's ``.env`` secret forms plus its applied secret-source values: every exact
    value Hermes knows is secret for this profile (the file-delivery refusal scans for these)."""
    from hermes_constants import hermes_home_key

    return (*profile_env_secret_forms(), *_secret_source_forms(hermes_home_key()))
