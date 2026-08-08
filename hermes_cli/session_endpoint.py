"""Helpers for ``hermes sessions set-endpoint`` (issue #77831).

Sessions pin their provider endpoint in the durable session row
(``sessions.model_config`` JSON plus the ``billing_provider`` /
``billing_base_url`` / ``billing_mode`` columns). The runtime resolves the
endpoint from that per-session snapshot on every resume — not from live
``config.yaml`` — so a moved model server keeps stranding open sessions even
across restarts, and there is no supported way to re-point one (only manual
``state.db`` editing).

These helpers back the ``hermes sessions set-endpoint <id> <url>`` command:

* :func:`normalize_endpoint_url` validates the new endpoint URL.
* :func:`resolve_endpoint_provider_id` resolves the provider id to persist so
  the rewritten row is immediately routable — never a ``custom:<name>`` slug
  with no matching config entry, which the runtime refuses with a hard
  ``Unknown provider 'custom:<...>'`` error (the secondary footgun reported
  in the issue).
* :func:`billing_provider_id` maps the persisted provider id to the
  ``sessions.billing_provider`` bucket the runtime itself writes for custom
  endpoints (the bare class ``"custom"``).

Config-backed lookups are imported lazily inside the functions (the
hermes_cli convention) so tests can patch the module of origin.
"""

from typing import Optional, Tuple
from urllib.parse import urlsplit


def normalize_endpoint_url(raw: str) -> str:
    """Validate an endpoint base URL and return it in canonical form.

    Accepts ``http``/``https`` URLs with a host. Returns the URL with a
    single trailing slash stripped (``http://host:8355/v1/`` →
    ``http://host:8355/v1``), matching the convention the runtime stores in
    ``model_config.base_url`` / ``billing_base_url``.

    Raises ``ValueError`` with a user-facing message otherwise.
    """
    value = (raw or "").strip()
    if not value:
        raise ValueError("endpoint URL is empty")
    try:
        parts = urlsplit(value)
    except ValueError as exc:
        raise ValueError(f"invalid endpoint URL {value!r}: {exc}") from exc
    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"invalid endpoint URL {value!r}: scheme must be http or https"
        )
    if not parts.hostname:
        raise ValueError(f"invalid endpoint URL {value!r}: missing host")
    # ``parts.port`` raises for out-of-range ports (e.g. :99999) — surface it.
    try:
        parts.port
    except ValueError as exc:
        raise ValueError(f"invalid endpoint URL {value!r}: {exc}") from exc
    return value.rstrip("/")


def _configured_custom_entry(provider_id: str):
    """Return the configured ``providers:`` / ``custom_providers:`` entry that
    carries ``provider_id`` (a ``custom:<name>`` slug), or None."""
    from hermes_cli.runtime_provider import _get_named_custom_provider

    return _get_named_custom_provider(provider_id)


def _canonical_builtin(provider_id: str) -> Optional[str]:
    """Canonical id for a built-in provider / alias, else None."""
    from hermes_cli.auth import AuthError, resolve_provider

    try:
        canonical = resolve_provider(provider_id)
    except AuthError:
        return None
    if canonical and canonical != "auto":
        return canonical
    return None


def _validate_requested_provider(provider_id: str) -> str:
    """Validate an explicit ``--provider`` value; returns the id to persist."""
    norm = (provider_id or "").strip()
    if not norm:
        raise ValueError("--provider value is empty")
    lower = norm.lower()
    if lower == "custom":
        return "custom"
    if lower == "auto":
        raise ValueError(
            f"Unknown provider '{norm}': 'auto' is not a routable provider id. "
            "Use a built-in provider, bare 'custom', or a configured custom:<name>."
        )
    if lower.startswith("custom:"):
        if _configured_custom_entry(norm) is not None:
            return norm
        raise ValueError(
            f"Unknown provider '{norm}': no providers:/custom_providers: entry "
            "matches this custom:<name>. Use the entry's slug, a built-in "
            "provider, or bare 'custom'."
        )
    canonical = _canonical_builtin(norm)
    if canonical:
        return canonical
    raise ValueError(
        f"Unknown provider '{norm}'. Check 'hermes model' for available "
        "providers, or use bare 'custom' for a generic endpoint."
    )


def resolve_endpoint_provider_id(
    base_url: str,
    existing_provider: Optional[str] = None,
    requested_provider: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve the provider id to persist for a session re-pointed to ``base_url``.

    Returns ``(provider_id, source)`` where source is one of:

    * ``"requested"`` — caller-supplied ``--provider`` (validated routable).
    * ``"config-entry"`` — the ``custom:<name>`` slug of the configured
      entry whose base_url matches the new endpoint (the issue's own
      workaround shape: keep the live ``providers.<name>`` form).
    * ``"existing"`` — the session's stored provider id, kept when it is
      still routable (built-in, bare ``custom``, or a configured entry).
    * ``"custom-fallback"`` — bare ``custom`` (the endpoint URL drives
      routing), used when nothing else names a routable identity.

    Raises ``ValueError`` when ``requested_provider`` is not routable.
    """
    if requested_provider:
        return _validate_requested_provider(requested_provider), "requested"

    from hermes_cli.runtime_provider import find_custom_provider_identity

    try:
        entry_slug = find_custom_provider_identity(base_url)
    except Exception:
        entry_slug = None
    if entry_slug:
        return entry_slug, "config-entry"

    existing = (existing_provider or "").strip()
    if existing:
        lower = existing.lower()
        if lower == "custom":
            return "custom", "existing"
        if lower.startswith("custom:"):
            if _configured_custom_entry(existing) is not None:
                return existing, "existing"
        elif _canonical_builtin(existing):
            return existing, "existing"

    return "custom", "custom-fallback"


def billing_provider_id(provider_id: str) -> str:
    """Map the persisted provider id to the ``sessions.billing_provider`` value.

    Custom endpoints (bare ``custom`` or any ``custom:<name>`` slug) are
    billed under the bare class ``"custom"`` — the same value the runtime's
    own billing-route persistence (``update_session_billing_route``) writes
    for them. Built-in providers keep their canonical id.
    """
    lower = (provider_id or "").strip().lower()
    if lower == "custom" or lower.startswith("custom:"):
        return "custom"
    return provider_id
