"""Helpers for X-Forwarded-Prefix support.

Mission-control style deploys reverse-proxy the dashboard at a path
prefix (e.g. ``mission-control.tilos.com/hermes/*`` -> dashboard on
:9119), injecting ``X-Forwarded-Prefix: /hermes`` so the backend can
reconstruct prefixed URLs (Location: headers, OAuth redirect_uri,
cookie Path attributes, SPA asset URLs).

This module is also the home of the ``HERMES_DASHBOARD_PUBLIC_URL`` /
``dashboard.public_url`` resolution — when the operator declares a
complete public URL (scheme + host + optional path prefix), we use
that directly for the OAuth ``redirect_uri`` and skip the
X-Forwarded-Prefix reconstruction. Relief valve for deploys where the
proxy header chain isn't reliable.

The single source of truth for both helpers lives here so the gate
middleware, the OAuth routes, the cookie helpers, and the SPA mount
all agree on validation rules.
"""
from __future__ import annotations

import logging
import os
import urllib.parse
from typing import Optional

_log = logging.getLogger(__name__)

# Home Assistant Supervisor ingress prefixes are already 63 chars before
# deployments add their own sub-path. Keep a bounded header budget, but leave
# room for mainstream reverse-proxy path mounts.
_MAX_PREFIX_LENGTH = 256

# Characters that, if present in a public_url or prefix value, indicate
# either a typo or a header-injection attempt. Reject the whole value
# rather than try to sanitise — the operator can fix their config.
_REJECT_CHARS = frozenset(('"', "'", "<", ">", " ", "\n", "\r", "\t"))

# Remember which (source, value) pairs we've already warned about.
# ``resolve_public_url`` runs on every authenticated request, so an
# un-deduplicated warning would flood the logs once per request for a
# misconfigured deploy. Keyed on the raw value too, so changing the
# config and reloading surfaces a fresh warning.
_warned_malformed_public_urls: set = set()
_warned_malformed_prefixes: set = set()


def _warn_if_malformed(source: str, raw: str) -> None:
    """Warn (once per distinct value) when a non-empty public-url value
    was rejected by :func:`_normalise_public_url`.

    A non-empty value that normalises to ``""`` is almost always a
    missing scheme (``hermes.example.com`` instead of
    ``https://hermes.example.com``) — the single most common cause of
    "I set HERMES_DASHBOARD_PUBLIC_URL but the OAuth callback is still
    http://". Without this warning the value is silently discarded and
    the dashboard falls back to reconstructing the redirect URI from
    request headers, which behind a reverse proxy can yield the wrong
    scheme. Surfacing it turns a silent footgun into a self-diagnosing
    one.
    """
    cleaned = raw.strip() if raw else ""
    if not cleaned:
        return  # empty/unset is a legitimate "no override" — not malformed
    key = (source, cleaned)
    if key in _warned_malformed_public_urls:
        return
    _warned_malformed_public_urls.add(key)
    _log.warning(
        "%s is set to %r but was ignored because it is not a valid "
        "absolute URL — it must include an http:// or https:// scheme "
        "(e.g. https://%s). Falling back to reconstructing the OAuth "
        "redirect URI from request headers, which may produce the wrong "
        "scheme behind a reverse proxy.",
        source,
        cleaned,
        cleaned.split("://")[-1] or "hermes.example.com",
    )


def _warn_if_malformed_prefix(raw: Optional[str], reason: str) -> None:
    """Warn once when a non-empty X-Forwarded-Prefix value is rejected."""
    cleaned = raw.strip() if raw else ""
    if not cleaned:
        return
    key = (cleaned, reason)
    if key in _warned_malformed_prefixes:
        return
    _warned_malformed_prefixes.add(key)
    _log.warning(
        "X-Forwarded-Prefix header %r was ignored because %s. "
        "Dashboard URLs will be generated without a reverse-proxy path prefix.",
        cleaned,
        reason,
    )


def normalise_prefix(raw: Optional[str]) -> str:
    """Normalise an X-Forwarded-Prefix header value.

    Returns a string like ``"/hermes"`` (no trailing slash) or ``""``
    when no prefix is set / the header is malformed. We deliberately
    reject anything containing ``..`` or non-printable bytes so a
    hostile proxy can't inject HTML or path-traversal sequences via the
    prefix.
    """
    if not raw:
        return ""
    p = raw.strip()
    if not p:
        return ""
    if not p.startswith("/"):
        p = "/" + p
    p = p.rstrip("/")
    if (
        "//" in p
        or ".." in p
        or any(c in p for c in _REJECT_CHARS)
    ):
        _warn_if_malformed_prefix(
            raw,
            "it contains a disallowed character or path sequence",
        )
        return ""
    if len(p) > _MAX_PREFIX_LENGTH:
        _warn_if_malformed_prefix(
            raw,
            f"it is longer than {_MAX_PREFIX_LENGTH} characters",
        )
        return ""
    return p


def prefix_from_request(request) -> str:
    """Convenience wrapper that reads the header off a Starlette/FastAPI
    Request and normalises it. Returns ``""`` when no prefix.
    """
    return normalise_prefix(request.headers.get("x-forwarded-prefix"))


# ---------------------------------------------------------------------------
# HERMES_DASHBOARD_PUBLIC_URL / dashboard.public_url
# ---------------------------------------------------------------------------


def _normalise_public_url(raw: Optional[str]) -> str:
    """Normalise a ``dashboard.public_url`` value.

    Returns the cleaned URL (scheme://netloc[/path], trailing slash
    removed) on success, or ``""`` when the value is empty, malformed,
    or contains characters that suggest header injection. The caller
    must treat ``""`` as "fall back to request reconstruction" — never
    as "the user explicitly chose no public URL", because the two are
    indistinguishable from an empty env var.
    """
    if not raw:
        return ""
    url = raw.strip()
    if not url:
        return ""
    # Reject control / quote / whitespace characters before trying to
    # parse — urlparse is permissive enough to accept some hostile
    # values (e.g. embedded newlines) and we want a hard "no" rather
    # than a soft "maybe".
    if any(c in url for c in _REJECT_CHARS):
        return ""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"}:
        return ""
    if not parsed.netloc:
        return ""
    # Strip a single trailing slash so callers can append paths without
    # producing ``//`` double-slashes.
    return url.rstrip("/")


def _load_dashboard_section() -> dict:
    """Return the ``dashboard`` block from ``config.yaml`` if it exists
    and is a dict; otherwise an empty dict.

    Robust to (a) load_config() raising (malformed YAML, IO error,
    config.yaml absent), and (b) ``dashboard`` being absent or non-dict.
    Both shapes fall through to ``{}`` so the caller can rely on
    ``.get(...)`` access.
    """
    try:
        from hermes_cli.config import load_config
    except Exception:
        return {}
    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        _log.debug(
            "dashboard-auth.prefix: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return {}
    section = cfg.get("dashboard") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def resolve_public_url() -> str:
    """Resolve the operator-declared dashboard public URL.

    Precedence (mirrors ``dashboard.oauth.client_id``):

      1. ``HERMES_DASHBOARD_PUBLIC_URL`` env var (when non-empty after
         strip — empty values are treated as unset so a provisioned-but-
         not-populated Fly secret can't shadow a valid config.yaml entry).
      2. ``dashboard.public_url`` in ``config.yaml``.
      3. Empty string — signals "no override, reconstruct from request"
         to the caller.

    Each candidate value is run through :func:`_normalise_public_url`.
    A malformed env var falls through to the config.yaml entry; a
    malformed config entry falls through to ``""``. This means a typo
    in one surface doesn't prevent the other from working.
    """
    env_raw = os.environ.get("HERMES_DASHBOARD_PUBLIC_URL", "")
    env_clean = _normalise_public_url(env_raw)
    if env_clean:
        return env_clean
    _warn_if_malformed("HERMES_DASHBOARD_PUBLIC_URL env var", env_raw)
    cfg_raw = str(_load_dashboard_section().get("public_url", ""))
    cfg_clean = _normalise_public_url(cfg_raw)
    if not cfg_clean:
        _warn_if_malformed("dashboard.public_url in config.yaml", cfg_raw)
    return cfg_clean
