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

# Characters that, if present in a public_url or prefix value, indicate
# either a typo or a header-injection attempt. Reject the whole value
# rather than try to sanitise — the operator can fix their config.
_REJECT_CHARS = frozenset(('"', "'", "<", ">", " ", "\n", "\r", "\t"))


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
        return ""
    if len(p) > 64:
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
    cfg_raw = _load_dashboard_section().get("public_url", "")
    return _normalise_public_url(str(cfg_raw))
