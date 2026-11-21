"""Exact-value redaction of the active profile's own ``.env`` secrets, and the one pass that masks
every exact value Hermes knows is secret.

The shape passes in ``agent.redact`` only recognise credential *shapes* (vendor prefixes,
``KEY=value`` assignments, auth headers). An opaque key with no known prefix, printed without its
variable name (a provider error quoting it, an MCP server echoing its env, a JSON dump of a
request), passes all of them. The profile's ``.env`` says exactly which bytes are secret, so its
values are masked wherever they appear, in the raw and JSON-escaped forms
``agent.redact_secret_sources`` already derives for secret-source values.

Values are what the runtime sees: ``hermes_cli.env_loader.resolve_home_dotenv`` (the loader's own
parser, so quoted multi-line values and ``${VAR:-default}`` expand as they do in ``os.environ``;
another profile's templates resolve against process-global vars only, as its secret scope does).
Same registration rule as secret-source values (``_worth_registering``: secret-named, credential
shaped or multi-line) plus a 12-character floor, so ``LOG_LEVEL=debug`` or a short PIN never
becomes an unanchored substring scrub of ordinary text. Only a whole value is masked.

Keyed by ``hermes_home_key()``: one gateway serves many profiles, and profile A's values must
never mask (or confirm the existence of) bytes in profile B's output. Redaction runs on every log
record, so a home's ``.env`` is re-stat'ed at most every ``_RECHECK_SECONDS`` and re-read only when
its mtime/size changed. A managed worker never opens the profile's files: its owner hands the
credentials over in the worker's environment, so its own home masks those. Memory only.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from agent.redact_secret_sources import REDACTED_SECRET_SOURCE_VALUE, _exact_forms
from agent.redact_secret_sources import _registered as _secret_source_forms
from agent.redact_vault_values import REDACTED_VAULT_VALUE, registered_vault_values

MARKER = "«redacted-secret»"
MIN_VALUE_CHARS = 12
_RECHECK_SECONDS = 2.0

# home key -> (checked_at, file signature, exact forms longest first)
_CACHE: dict[str, tuple[float, tuple, tuple[str, ...]]] = {}
_LOCK = threading.Lock()


def _signature(paths: list[Path]) -> tuple:
    sig = []
    for path in paths:
        try:
            st = os.stat(path)
        except OSError:
            sig.append(None)
        else:
            sig.append((st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _dotenv_layers(home: Path) -> list[Path]:
    from hermes_cli.managed_scope import get_managed_dir

    managed = get_managed_dir()
    return [home / ".env", *([managed / ".env"] if managed is not None else [])]


def _resolved_values(home: Path) -> dict[str, str]:
    from agent.secret_scope import _is_global_env, _is_process_home
    from hermes_cli.env_loader import resolve_home_dotenv

    lookup = None if _is_process_home(home) else {k: v for k, v in os.environ.items() if _is_global_env(k)}
    return resolve_home_dotenv(home, lookup_env=lookup)


def _worker_values(home: Path) -> dict[str, str]:
    """A managed worker's credentials arrive in its environment; only its own home takes them."""
    from hermes_constants import get_process_hermes_home, hermes_home_key

    from agent.secret_scope import _is_global_env

    if hermes_home_key(home) != hermes_home_key(get_process_hermes_home()):
        return {}
    return {k: v for k, v in os.environ.items() if not _is_global_env(k)}


def _load(home: Path, worker: bool) -> tuple[tuple, tuple[str, ...]]:
    if worker:
        values, sig = _worker_values(home), ("worker",)
    else:
        layers = _dotenv_layers(home)
        sig = _signature(layers)
        values = _resolved_values(home) if any(sig) else {}
    return sig, _exact_forms({k: v for k, v in values.items() if len(v.strip()) >= MIN_VALUE_CHARS})


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
    worker = worker_config_snapshot() is not None  # checked per recheck, not per scrub: it parses JSON
    if cached is not None and not worker and cached[1] == _signature(_dotenv_layers(home)):
        sig, forms = cached[1], cached[2]
    else:
        sig, forms = _load(home, worker)
    with _LOCK:
        _CACHE[key] = (now, sig, forms)
    return forms


def clear_profile_env_secret_cache() -> None:
    """Forget every home's registered values (tests; the next scrub re-reads ``.env``)."""
    with _LOCK:
        _CACHE.clear()


def env_masking_enabled() -> bool:
    """``.env`` values follow the ``security.redact_secrets`` preference (vault and secret-source
    values do not: they are always masked)."""
    from agent.redact import _redact_enabled

    return _redact_enabled()


def redact_exact_secret_values(text: str, *, env: bool) -> str:
    """Mask every exact value registered for the active profile in ONE longest-first pass: vault
    fills and secret-source values always, ``.env`` values when ``env``. One pass, so a shorter
    value from one registry never splits a longer one from another and leaves its rest bare."""
    if not isinstance(text, str) or not text:
        return text
    from hermes_constants import hermes_home_key

    marked = [(v, REDACTED_VAULT_VALUE) for v in registered_vault_values()]
    marked += [(v, REDACTED_SECRET_SOURCE_VALUE) for v in _secret_source_forms(hermes_home_key())]
    if env:
        marked += [(v, MARKER) for v in profile_env_secret_forms()]
    marked.sort(key=lambda pair: len(pair[0]), reverse=True)
    for value, marker in marked:
        if value in text:
            text = text.replace(value, marker)
    return text


def registered_secret_forms() -> tuple[str, ...]:
    """The active profile's ``.env`` secret forms plus its applied secret-source values: every exact
    value Hermes knows is secret for this profile (the file-delivery refusal scans for these)."""
    from hermes_constants import hermes_home_key

    return (*profile_env_secret_forms(), *_secret_source_forms(hermes_home_key()))
