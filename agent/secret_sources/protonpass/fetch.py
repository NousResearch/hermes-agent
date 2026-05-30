"""Fetching + parsing Proton Pass secrets via ``pass-cli``.

Two fetch modes (both optional, MODE B wins on collision):

* MODE B (refs, PREFERRED): ``env: {ENV_VAR: "pass://SHARE/ITEM/FIELD"}`` — one
  ``pass-cli item view`` call per ref, reading the bare value from stdout.
* MODE A (vault list): ``vault: "<name>"`` — one
  ``pass-cli item list <vault> --show-secrets --output json`` call.

Caching, session establishment, and binary discovery are delegated to the
sibling modules; this module owns the fetch plumbing, the JSON parsing, the
argument-injection validators, and the env-name derivation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .cache import (
    _CACHE,
    _CachedFetch,
    _read_disk_cache,
    _write_disk_cache,
    build_cache_key,
)
from .config import (
    _DEFAULT_SERVICE_TOKEN_ENV,
    is_valid_env_name,
    strip_bootstrap_refs,
)
from .install import find_pass_cli
from .session import _child_env, _clean_stream, _establish_session, _redact_token, _run_pass_cli

logger = logging.getLogger(__name__)

# Max share/item ID length we accept in a pass:// ref before treating it as
# malformed.  Proton IDs are short base64url strings; this is generous.
_MAX_ID_LEN = 256

# A plausible Proton share/item ID: ASCII base64url body with OPTIONAL trailing
# ``=`` padding (real probe IDs end in ``==``, e.g. ``XhBBMrgq...EO90TRBZFA==``).
# Anchored with ``re.fullmatch`` so a ``/``, whitespace, leading ``-``, internal
# ``=``, or a TRAILING NEWLINE can't slip a flag/path into argv.  (A trailing
# ``$`` would have matched before a final ``\n``, so the previous ``^...$``
# anchoring wrongly accepted ``id\n``.)  Length is bounded separately by
# ``_MAX_ID_LEN``.
_ID_RE = re.compile(r"[A-Za-z0-9_-]+={0,2}")


@dataclass
class _FetchResult:
    """Outcome of one fetch leg (MODE A vault list or MODE B refs).

    ``cache_blockers`` counts retry-worthy incompletenesses — a transiently
    failed or empty ref, a glitched vault list.  A non-zero count means the
    result is partial/transient and must NOT be cached, so the failed piece is
    retried before the TTL rather than frozen for it.  Permanent validation
    skips (malformed ref, flag-like name) only add warnings and are NOT counted.
    """

    secrets: Dict[str, str]
    warnings: List[str]
    cache_blockers: int = 0


def _cli_error_detail(proc, token: str, *, limit: int = 200) -> str:
    """Build a redacted, diagnostic-safe error string from a failed pass-cli call.

    SECURITY: ``stdout`` is NEVER read here.  ``pass-cli item view --field``
    writes the bare SECRET VALUE to stdout, and ``item list --show-secrets``
    writes secret JSON to stdout — so a command that emitted the secret and THEN
    exited non-zero would leak it straight into a warning if stdout were
    surfaced.  Only ``stderr`` (token-scrubbed, ANSI-stripped, truncated) is
    diagnostic-safe.  When stderr is empty we fall back to a generic marker,
    NEVER the captured stdout.  Callers prepend ``exited <code>`` themselves.
    """
    err = _redact_token(_clean_stream(proc.stderr or ""), token).strip()
    return err[:limit] if err else "(no stderr output)"


def _drop_bootstrap(
    secrets: Dict[str, str], bootstrap_names: Iterable[str]
) -> Dict[str, str]:
    """Return ``secrets`` without any bootstrap key (a copy when one is present).

    Defensive guard on the cache-READ paths: a legacy disk entry written before
    the fail-closed fix could still carry a bootstrap-named value (e.g. the
    canonical default token env var).  We never want to hand any of them back to
    a caller, so drop every name in ``bootstrap_names`` without mutating the
    cached entry.  A no-op (none present) returns the same object.
    """
    drop = {n for n in bootstrap_names if n and n in secrets}
    if not drop:
        return secrets
    return {k: v for k, v in secrets.items() if k not in drop}


def fetch_protonpass_secrets(
    *,
    service_token: str,
    vault: str = "",
    env_refs: Optional[Dict[str, str]] = None,
    binary: Optional[Path] = None,
    cache_ttl_seconds: float = 300,
    use_cache: bool = True,
    auto_install: bool = True,
    home_path: Optional[Path] = None,
    bootstrap_env: str = "",
) -> Tuple[Dict[str, str], List[str]]:
    """Pull secrets from Proton Pass via ``pass-cli``.

    Returns ``(secrets_dict, warnings_list)``.

    Two modes, both optional and combinable:

    * ``env_refs`` (MODE B, preferred): a mapping of env-var name →
      ``pass://SHARE/ITEM/FIELD`` URI.  Each is resolved with one
      ``pass-cli item view`` call.
    * ``vault`` (MODE A): a vault name listed with
      ``pass-cli item list <vault> --show-secrets``; every item's fields
      become ``ITEM_FIELD`` upper-snake env vars.

    MODE B entries take precedence over MODE A-derived names on collision.

    Caching is a two-layer LRU: an in-process dict (for hot-reload paths inside
    one process) and a disk-persisted JSON file under
    ``<hermes_home>/cache/protonpass_cache.json`` (for back-to-back CLI
    invocations).  Both share the same TTL and store only values + a token
    fingerprint, never the token.  Pass ``home_path`` so disk cache lookups
    find the right directory in tests / non-standard installs; otherwise we
    fall back to ``$HERMES_HOME`` / ``~/.hermes``.

    Raises :class:`RuntimeError` for fatal conditions (missing binary, auth
    failure).  A MODE A list call that returns unparseable / wrong-shape output
    does NOT raise — it degrades to a warning and skips MODE A so a combined
    config's independent MODE B refs still resolve.  Callers in the env_loader
    path catch fatal errors and emit a single warning; callers in the
    user-facing setup wizard let them propagate.  No secret value or the token
    is ever included in a raised message.

    The "never cache or apply the bootstrap token" guarantee is FAIL-CLOSED on
    the canonical default token env var (``config._DEFAULT_SERVICE_TOKEN_ENV``):
    that name is ALWAYS protected even when the caller omits ``bootstrap_env``,
    and a truthy ``bootstrap_env`` (a custom token env var) is protected too.
    For every protected name: a MODE B ref whose KEY equals it is stripped
    BEFORE fetching/keying, and a MODE A vault item whose DERIVED name equals it
    is dropped from ``secrets`` BEFORE the cache write/return — so the token is
    never written to the plaintext disk cache nor returned to a caller.  Legacy
    disk entries are re-filtered on read for the same reason.  The apply-time
    planner is the independent second half of that invariant.
    """
    if not service_token:
        raise RuntimeError("Proton Pass service token is empty")
    env_refs = env_refs or {}
    if not vault and not env_refs:
        raise RuntimeError(
            "Proton Pass source has neither a vault (MODE A) nor env refs (MODE B)"
        )

    warnings: List[str] = []

    # FAIL CLOSED: the canonical default token env var is ALWAYS protected, even
    # when the caller omits ``bootstrap_env`` (a MODE A item that happens to
    # derive it would otherwise leak into the cache/result).  A truthy custom
    # ``bootstrap_env`` is protected on top of it.
    bootstrap_names: Set[str] = {_DEFAULT_SERVICE_TOKEN_ENV}
    if bootstrap_env:
        bootstrap_names.add(bootstrap_env)

    # Strip — before fetching OR keying — every MODE B ref whose KEY is one of
    # the protected names.  Centralized in ``strip_bootstrap_refs`` (config.py);
    # a no-op on already-filtered refs (callers may pre-filter) and on names no
    # ref targeted, so there is no spurious / duplicate warning.  Doing this
    # BEFORE the cache key is built keeps a bootstrap-named ref out of the key.
    env_refs, _bootstrap_skipped, bootstrap_warnings = strip_bootstrap_refs(
        env_refs, bootstrap_names
    )
    warnings.extend(bootstrap_warnings)

    cache_key = build_cache_key(service_token, vault, env_refs, home_path)
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return _drop_bootstrap(cached.secrets, bootstrap_names), list(warnings)
        # L2: disk cache. Cheap read vs re-establishing a session + fetching.
        disk_cached = _read_disk_cache(cache_key, cache_ttl_seconds, home_path)
        if disk_cached is not None:
            dropped = _drop_bootstrap(disk_cached.secrets, bootstrap_names)
            # Promote a bootstrap-dropped copy so the in-process cache never
            # holds a protected key, even from a legacy disk entry.
            _CACHE[cache_key] = _CachedFetch(dropped, disk_cached.fetched_at)
            return dropped, list(warnings)

    # Honor auto_install: when False we never download — only an existing
    # managed copy or a SHA-256-verified PATH binary is used.
    pass_cli = binary or find_pass_cli(install_if_missing=auto_install)
    if pass_cli is None:
        raise RuntimeError(
            "pass-cli binary not available. Run `hermes secrets protonpass "
            "install` to download the verified pinned version, or leave "
            "`auto_install: true` in the secrets.protonpass config so Hermes "
            "downloads it automatically (set `auto_install: false` to opt out "
            "of automatic downloads)."
        )

    warnings.extend(_establish_session(service_token, pass_cli))

    secrets: Dict[str, str] = {}
    # Each fetch leg reports its own retry-worthy incompleteness via
    # _FetchResult.cache_blockers; we sum them and cache only a clean (zero)
    # result.  MODE A first so MODE B can override on collision.
    cache_blockers = 0

    if vault:
        a = _fetch_vault(pass_cli, service_token, vault)
        secrets.update(a.secrets)
        warnings.extend(a.warnings)
        cache_blockers += a.cache_blockers

    if env_refs:
        b = _fetch_refs(pass_cli, service_token, env_refs)
        secrets.update(b.secrets)  # MODE B precedence
        warnings.extend(b.warnings)
        cache_blockers += b.cache_blockers

    # MODE A leak fix: a vault item whose DERIVED env name equals ANY protected
    # bootstrap name must never be written to the plaintext disk cache nor
    # returned (the planner refuses to apply it, but it would otherwise leak into
    # the cache file / result.secrets).  Drop every protected name here, BEFORE
    # the cache write.  Sorted for a deterministic warning order.
    for name in sorted(bootstrap_names):
        removed = secrets.pop(name, None)
        if removed is not None:
            warnings.append(
                f"Skipping fetched value for {name!r}: it matches the "
                "bootstrap service-token env var and is never cached or applied."
            )

    # An empty combined result that ALSO produced warnings is a recoverable
    # hiccup, not an intentional empty success — don't freeze it for the TTL.
    if not secrets and warnings:
        cache_blockers += 1

    # Cache only a clean, complete result.  cache_blockers > 0 means some leg saw
    # a retry-worthy incompleteness (a transient/empty ref, a glitched vault, an
    # all-empty+warning result).  ttl<=0 disables BOTH layers ("always refetch").
    caching_enabled = use_cache and cache_ttl_seconds > 0 and cache_blockers == 0

    if caching_enabled:
        entry = _CachedFetch(secrets=secrets, fetched_at=time.time())
        _CACHE[cache_key] = entry
        _write_disk_cache(cache_key, entry, home_path)
    return secrets, warnings


def _fetch_refs(
    binary: Path,
    token: str,
    env_refs: Dict[str, str],
) -> _FetchResult:
    """MODE B: resolve each ``ENV_VAR -> pass://...`` ref to a single value.

    Confirmed (pass-cli 2.1.1): the config ref is
    ``pass://SHARE_ID/ITEM_ID/FIELD``.  IDs are base64url (no ``/``), so we
    split on ``/`` into exactly ``[share_id, item_id, field]``.  The ref MUST
    carry the ``pass://`` scheme and resolve to exactly three non-empty
    components; a non-``pass://`` URI, a missing FIELD, or an over-long ref
    (``.../F/extra``) is skipped with a warning naming the expected shape.  The
    IDs are validated as base64url and a flag-like FIELD is rejected
    (argument-injection defence).  The value is fetched with::

        pass-cli item view --field <FIELD> -- "pass://SHARE_ID/ITEM_ID"

    and read as the RAW value from stdout (``--field`` returns the bare value,
    NOT JSON, so we do NOT json-parse this path), trailing newline stripped.
    The ``--`` terminates option parsing so the positional URI can't be read as
    a flag.
    """
    secrets: Dict[str, str] = {}
    warnings: List[str] = []
    # Recoverable per-ref failures (timeout, non-zero exit, empty value) — NOT
    # permanent validation skips.  Drives _FetchResult.cache_blockers.
    transient_failures = 0
    env = _child_env(token)
    for env_name, uri in env_refs.items():
        if not _is_valid_env_name(env_name):
            warnings.append(f"Skipping ref {env_name!r}: not a valid env-var name")
            continue
        if not isinstance(uri, str) or not uri:
            warnings.append(f"Skipping ref {env_name!r}: empty pass:// reference")
            continue
        parsed = _split_ref(uri)
        if parsed is None:
            # N3: name only the env var, never the full pass:// ref — the ref
            # encodes the user's secret-location structure (share/item ids).
            warnings.append(
                f"Skipping ref {env_name!r}: pass:// reference is malformed "
                "(expected pass://SHARE_ID/ITEM_ID/FIELD with exactly three "
                "non-empty components)"
            )
            continue
        share_id, item_id, field_name = parsed
        # Argument-injection defence: validate the IDs as base64url and reject
        # a field name that would be read as a flag.  Both go into argv, so a
        # value like "--show-secrets" or "-x" must never slip through.
        if not _is_valid_share_or_item_id(share_id) or not _is_valid_share_or_item_id(
            item_id
        ):
            warnings.append(
                f"Skipping ref {env_name!r}: SHARE_ID/ITEM_ID are not valid "
                "base64url identifiers"
            )
            continue
        if _is_flag_like(field_name):
            warnings.append(
                f"Skipping ref {env_name!r}: FIELD name looks like a flag "
                "(starts with '-')"
            )
            continue
        item_uri = f"pass://{share_id}/{item_id}"
        # ``--`` terminates option parsing so the positional pass:// URI and the
        # field value can't be misinterpreted as flags.  ``--field`` returns the
        # bare value (NOT JSON), so we do not pass ``--output json`` here.
        cmd = [
            str(binary), "item", "view", "--field", field_name, "--", item_uri,
        ]
        proc = _run_pass_cli(cmd, env)
        if proc is None:
            warnings.append(
                f"Skipping ref {env_name!r}: pass-cli timed out or failed to "
                "invoke"
            )
            transient_failures += 1
            continue

        if proc.returncode != 0:
            # SECURITY: never surface stdout here — ``item view --field`` writes
            # the bare secret to stdout, so a non-zero exit AFTER stdout was
            # written would leak it.  Only stderr is diagnostic-safe.  The URI
            # also contains ids the user may consider sensitive, so we keep the
            # env-var name in the warning but never the resolved value.
            warnings.append(
                f"Skipping ref {env_name!r}: pass-cli exited {proc.returncode}: "
                f"{_cli_error_detail(proc, token)}"
            )
            transient_failures += 1
            continue

        # --field emits the bare value on stdout (NOT JSON).  Strip EXACTLY ONE
        # trailing line terminator the CLI appends — never ``.rstrip("\r\n")``
        # (which would eat EVERY trailing CR/LF and corrupt a secret that itself
        # ends in a newline: pass-cli appends one terminator, so a value ending
        # in "\n" arrives as "...\n\n" and would lose BOTH) and never ``.strip()``
        # (which would corrupt significant leading/trailing whitespace, e.g. a
        # key padded with spaces or a multi-line PEM).  Empty stdout → nothing to
        # inject.
        raw_stdout = proc.stdout or ""
        if raw_stdout.endswith("\r\n"):
            value = raw_stdout[:-2]
        elif raw_stdout.endswith("\n") or raw_stdout.endswith("\r"):
            value = raw_stdout[:-1]
        else:
            value = raw_stdout
        if not value:
            warnings.append(
                f"Skipping ref {env_name!r}: pass-cli returned an empty value"
            )
            # Treat empty-on-success conservatively as possibly transient so a
            # partial omitting this ref isn't frozen in the cache for the TTL.
            transient_failures += 1
            continue
        secrets[env_name] = value
    return _FetchResult(secrets, warnings, transient_failures)


def _split_ref(uri: str) -> Optional[Tuple[str, str, str]]:
    """Split ``pass://SHARE_ID/ITEM_ID/FIELD`` into (share_id, item_id, field).

    Strict: the ref MUST carry the ``pass://`` scheme and resolve to EXACTLY
    three NON-EMPTY components (SHARE_ID, ITEM_ID, FIELD).  IDs are base64url
    (no ``/``), so the body splits cleanly on ``/``.  We split WITHOUT filtering
    empty parts and then require ``len(parts) == 3 and all(parts)`` — so an
    empty interior or leading component (``pass://S//F``, ``pass:///I/F``) is
    REJECTED rather than "magically repaired" by collapsing the empty segment.
    Returns ``(share_id, item_id, field)`` on a well-formed ref, or ``None``
    when the URI lacks the ``pass://`` prefix or does not have exactly three
    non-empty components — so ``pass://S/I`` (no FIELD), ``pass://S//F`` (empty
    interior), and ``pass://S/I/F/extra`` are ALL rejected rather than silently
    truncated.  The caller emits a warning naming the expected shape and
    validates the IDs before using them.
    """
    if not uri.startswith("pass://"):
        return None
    body = uri[len("pass://"):]
    parts = body.split("/")
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def _fetch_vault(
    binary: Path,
    token: str,
    vault: str,
) -> _FetchResult:
    """MODE A: list a vault's items and map their fields to env vars.

    Confirmed (pass-cli 2.1.1): ``pass-cli item list --show-secrets --output
    json -- <vault>`` returns ``{"items": [...]}`` with full field values under
    a full/PAT session.  Under a scoped AGENT session ``--show-secrets`` is
    rejected → we catch the non-zero exit, warn, and skip (NEVER crash); MODE A
    is documented as PAT-session-only.

    Argument-injection defence: a vault name beginning with ``-`` would be read
    as a flag, so we reject it; ``--`` terminates option parsing before the
    positional vault name so it can never be misread as one.
    """
    if _is_flag_like(vault):
        # Permanent validation skip (a flag-like name can never become valid),
        # so warn but do NOT block caching of a MODE B result fetched alongside.
        return _FetchResult({}, [
            f"Skipping vault {vault!r} (MODE A): vault name looks like a flag "
            "(starts with '-')"
        ])
    cmd = [
        str(binary), "item", "list", "--show-secrets", "--output", "json",
        "--", vault,
    ]
    env = _child_env(token)
    proc = _run_pass_cli(cmd, env)
    if proc is None:
        return _FetchResult({}, [
            f"Skipping vault {vault!r}: pass-cli timed out or failed to invoke"
        ], cache_blockers=1)

    if proc.returncode != 0:
        # SECURITY: never surface stdout here — ``item list --show-secrets``
        # writes secret JSON to stdout, so a non-zero exit AFTER stdout was
        # written would leak it.  Only stderr is diagnostic-safe.
        #
        # Under a scoped agent session `--show-secrets` is rejected.  Degrade
        # to a warning + skip so MODE A never crashes startup; tell the user to
        # use MODE B refs (which work under agent sessions) or a full PAT
        # session.
        return _FetchResult({}, [
            f"Skipping vault {vault!r} (MODE A): pass-cli exited "
            f"{proc.returncode}: {_cli_error_detail(proc, token)}. If using an "
            "agent session, use MODE B `env:` refs or a PAT session instead."
        ], cache_blockers=1)

    secrets, warnings = _parse_item_list_json(proc.stdout, vault)
    # MODE A blocks caching only on a TOTAL glitch (a warning AND zero keys) — a
    # transient list failure / rejected --show-secrets shouldn't freeze a
    # MODE-A-less result for the TTL.  A partial (some keys + a warning) is still
    # cacheable.
    return _FetchResult(
        secrets, warnings, cache_blockers=1 if (warnings and not secrets) else 0
    )


def _parse_item_list_json(raw: str, vault: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse ``pass-cli item list ... --output json`` into env vars.

    Confirmed (pass-cli 2.1.1): ``{"items": [{... , "content": {...}}]}`` where
    each item's ``content`` is
    ``{"title","note","item_uuid","content":{"<Type>":{field:value}},
    "extra_fields":[{"name","content":{"<Kind>":value}}]}``.  Typed fields live
    under ``content.content.<Type>`` (e.g. Login → email/username/password/
    urls/totp_uri/passkeys); custom fields under ``content.extra_fields[]``
    (one-key ``content`` dict whose value is the secret).  Env names are
    ``UPPER_SNAKE(title)_UPPER_SNAKE(field)`` (e.g. PROBE_LOGIN_PASSWORD,
    PROBE_LOGIN_API_KEY).  Only non-empty SCALAR STRING values are emitted —
    lists (urls/passkeys) and empty strings are skipped.  Names run through
    :func:`_is_valid_env_name`; invalid/colliding names are skipped + warned.
    """
    raw = (raw or "").strip()
    warnings: List[str] = []
    if not raw:
        return {}, [f"pass-cli returned no output for vault {vault!r}"]
    # V6: a MODE A JSON-decode / shape error must NOT raise out of fetch.  When
    # both `vault:` (MODE A) and `env:` (MODE B) are configured, an exit-0 chunk
    # of garbage JSON from the list call would otherwise abort the whole fetch
    # and lose the (independent) MODE B refs.  Mirror the non-zero-exit degrade
    # path: warn + skip MODE A, return ({}, [warning]) so MODE B still resolves.
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, [
            f"Skipping vault {vault!r} (MODE A): pass-cli returned non-JSON "
            f"output: {exc}"
        ]

    if isinstance(payload, dict):
        # Confirmed wrapper: {"items": [...]}.
        payload = payload.get("items", payload)
    if not isinstance(payload, list):
        return {}, [
            f"Skipping vault {vault!r} (MODE A): pass-cli returned an "
            f"unexpected shape ({type(payload).__name__})"
        ]

    secrets: Dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, dict):
            continue
        title = content.get("title") or ""
        if not isinstance(title, str) or not title:
            continue
        for field_name, value in _iter_item_fields(item):
            # Only non-empty scalar STRINGS; skip lists (urls/passkeys), dicts,
            # and empty strings.
            if not isinstance(value, str) or not value:
                continue
            env_name = _env_name_from(title, field_name)
            if not _is_valid_env_name(env_name):
                warnings.append(
                    f"Skipping {title}/{field_name}: derived name "
                    f"{env_name!r} is not a valid env-var name"
                )
                continue
            if env_name in secrets:
                warnings.append(
                    f"Skipping {title}/{field_name}: env name {env_name!r} "
                    "collides with an earlier item"
                )
                continue
            secrets[env_name] = str(value)
    return secrets, warnings


def _iter_item_fields(item: dict):
    """Yield (field_name, value) pairs from one item dict.

    Confirmed (pass-cli 2.1.1) per-item shape: ``item["content"]`` =
    ``{"title","note","item_uuid","content": {"<Type>": {field: value, ...}},
    "extra_fields": [{"name": <str>, "content": {"<Kind>": <value>}}]}``.

    Typed fields live under ``content.content.<Type>`` (e.g. Login →
    email/username/password/urls/totp_uri/passkeys).  Custom fields live under
    ``content.extra_fields[]`` where each is a one-key ``content`` dict whose
    value is the secret.  Values are yielded as-is; the caller filters to
    non-empty scalar strings.
    """
    content = item.get("content")
    if not isinstance(content, dict):
        return

    # Typed body: content.content.<Type> → {field: value}.
    typed = content.get("content")
    if isinstance(typed, dict):
        for type_body in typed.values():
            if isinstance(type_body, dict):
                for name, value in type_body.items():
                    if isinstance(name, str):
                        yield name, value

    # Custom fields: extra_fields[] = {"name", "content": {"<Kind>": value}}.
    extra = content.get("extra_fields")
    if isinstance(extra, list):
        for entry in extra:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            kind_dict = entry.get("content")
            if not isinstance(name, str) or not isinstance(kind_dict, dict):
                continue
            # One-key dict: the single value is the secret.
            for value in kind_dict.values():
                yield name, value


def _env_name_from(title: str, field_name: str) -> str:
    """Derive an ``ITEM_FIELD`` upper-snake env-var name."""
    return f"{_upper_snake(title)}_{_upper_snake(field_name)}"


def _upper_snake(text: str) -> str:
    """Upper-snake a free-form label: non-alnum → ``_``, collapse repeats."""
    out: List[str] = []
    prev_us = False
    for ch in text:
        if ch.isalnum():
            out.append(ch.upper())
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
            prev_us = True
    return "".join(out).strip("_")


def _is_valid_share_or_item_id(value: str) -> bool:
    """True if ``value`` is a plausible Proton share/item ID.

    IDs are ASCII base64url tokens — ``[A-Za-z0-9_-]`` — WITH optional trailing
    ``=`` padding.  Real Proton IDs end in ``==`` (e.g.
    ``XhBBMrgq...EO90TRBZFA==``), so the validator MUST accept that padding or
    every real ``pass://SHARE/ITEM/FIELD`` ref is silently skipped.  We reject
    anything else (whitespace, ``/``, embedded ``=``, empty, over-length) and a
    leading ``-`` (which would be read as a flag) so a crafted ``pass://`` ref
    can't smuggle a flag or path into argv.
    """
    if not value or len(value) > _MAX_ID_LEN:
        return False
    # Leading ``-`` would be misread as a CLI flag — reject up front.
    if value[0] == "-":
        return False
    # ``re.fullmatch`` anchors at END OF STRING (not before a final ``\n`` the
    # way a trailing ``$`` would), so an id like ``"id\n"`` is REJECTED.
    return bool(_ID_RE.fullmatch(value))


def _is_flag_like(value: str) -> bool:
    """True if a positional value would be misread as a CLI flag (starts ``-``)."""
    return value.startswith("-")


# The canonical ASCII env-name validator lives in config.py so the rule has a
# single home (config imports nothing from the package, so this fetch->config
# edge introduces no import cycle).  Re-exported under the historical private
# name so internal call sites and tests that reference ``fetch._is_valid_env_name``
# keep working.
_is_valid_env_name = is_valid_env_name
