"""Unified provider catalog — one source of truth for the provider universe.

The provider list shown by ``hermes model`` (CLI/TUI) and the desktop Settings
→ Providers tabs (Accounts + API keys) **must be the same set**.  Historically
they were not: the CLI picker read :data:`hermes_cli.models.CANONICAL_PROVIDERS`
(which auto-extends from ``plugins/model-providers/<name>/``), while the desktop
tabs read separate hand-maintained lists (``_OAUTH_PROVIDER_CATALOG``,
``OPTIONAL_ENV_VARS`` + ``PROVIDER_GROUPS``) that nobody kept in sync.  Every
provider added after those lists were written silently went missing from the
GUI — e.g. GitHub Copilot showing up only under "tools", or ``openai-api`` being
configurable from the CLI but not the desktop app.

This module fixes that at the root: it derives ONE descriptor per provider from
the same universe ``hermes model`` renders (``CANONICAL_PROVIDERS``), joining:

* ``auth_type`` / ``api_key_env_vars`` / ``base_url_env_var`` from
  :data:`hermes_cli.auth.PROVIDER_REGISTRY` (credential truth), and
* ``display_name`` / ``description`` / ``signup_url`` from the provider's
  :class:`providers.base.ProviderProfile` when one exists, falling back to the
  ``CANONICAL_PROVIDERS`` entry's ``label`` / ``tui_desc`` and the
  ``OPTIONAL_ENV_VARS`` signup URL otherwise (many profiles leave these blank,
  and four canonical providers have no profile at all — lmstudio, openai-api,
  tencent-tokenhub, xai-oauth — so the fallbacks are load-bearing).

Each descriptor is tagged with the ``tab`` it belongs on (``keys`` vs
``accounts``) based purely on how the provider authenticates.  The desktop
``/api/env`` and ``/api/providers/oauth`` endpoints derive their MEMBERSHIP from
this catalog; the old hand lists are demoted to presentation/override overlays
(bespoke OAuth flow + status resolvers, richer copy, icons, ordering) and no
longer decide which providers exist.

Parity contract (locked by tests): the union of the two tabs equals the
``CANONICAL_PROVIDERS`` universe, i.e. exactly what ``hermes model`` shows.
"""

from __future__ import annotations

from dataclasses import dataclass

# Auth types that authenticate via an account / sign-in flow rather than a
# pasted API key.  These route to the desktop "Accounts" tab; everything else
# (api_key, and aws_sdk which is configured via AWS_REGION/AWS_PROFILE) routes
# to the "API keys" tab.  Mirrors the auth_type strings used in
# hermes_cli.auth.PROVIDER_REGISTRY and providers.base.ProviderProfile.
_ACCOUNTS_AUTH_TYPES: frozenset[str] = frozenset(
    {
        "oauth_device_code",
        "oauth_external",
        "oauth_minimax",
        "external_process",  # copilot-acp / devin-acp / grok-acp: spawn local ACP CLIs
        "copilot",           # GitHub Copilot token / gh auth
    }
)


@dataclass(frozen=True)
class ProviderDescriptor:
    """One provider, as seen by every surface (CLI picker + both GUI tabs)."""

    slug: str                      # canonical id, e.g. "openai-codex"
    label: str                     # human display name
    description: str               # one-line description
    auth_type: str                 # api_key | oauth_* | external_process | copilot | aws_sdk
    tab: str                       # "keys" | "accounts"
    api_key_env_vars: tuple[str, ...]  # credential env vars (may be empty)
    base_url_env_var: str          # base-URL override env var (may be "")
    signup_url: str                # signup / console URL (may be "")
    order: int                     # CANONICAL_PROVIDERS index — mirrors `hermes model`


def tab_for_auth_type(auth_type: str) -> str:
    """Return the desktop tab ("keys"|"accounts") a provider's auth maps to."""
    return "accounts" if auth_type in _ACCOUNTS_AUTH_TYPES else "keys"


def _split_env_vars(env_vars: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """Split a profile's ``env_vars`` into (api_key_vars, base_url_var)."""
    keys = tuple(v for v in env_vars if not (v.endswith("_BASE_URL") or v.endswith("_URL")))
    base = next((v for v in env_vars if v.endswith("_BASE_URL") or v.endswith("_URL")), "")
    return keys, base


def provider_catalog() -> list[ProviderDescriptor]:
    """Return one descriptor per provider in the ``hermes model`` universe.

    Membership is :data:`CANONICAL_PROVIDERS` (the list the CLI/TUI picker
    renders, which auto-extends from provider plugins).  Auth + env come from
    ``PROVIDER_REGISTRY``; display metadata from ``ProviderProfile`` with
    canonical/env fallbacks so providers without a profile (or with blank
    profile metadata) still resolve sensibly.
    """
    from hermes_cli.models import CANONICAL_PROVIDERS

    # PROVIDER_REGISTRY / list_providers are imported lazily and defensively:
    # this module is on the import path of the web server and the CLI, and we
    # never want a provider-plugin import error to blank the whole catalog.
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception:
        PROVIDER_REGISTRY = {}

    try:
        from providers import list_providers

        profiles = {p.name: p for p in list_providers()}
    except Exception:
        profiles = {}

    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS
    except Exception:
        OPTIONAL_ENV_VARS = {}

    # Hermes overlays carry auth_type for providers that have no registry/profile
    # entry of their own — notably the ``moa`` virtual provider (auth_type
    # "virtual"), which has no real credential and no network endpoint.
    try:
        from hermes_cli.providers import HERMES_OVERLAYS
    except Exception:
        HERMES_OVERLAYS = {}

    out: list[ProviderDescriptor] = []
    for order, entry in enumerate(CANONICAL_PROVIDERS):
        slug = entry.slug
        cfg = PROVIDER_REGISTRY.get(slug)
        prof = profiles.get(slug)
        overlay = HERMES_OVERLAYS.get(slug)

        # auth_type: registry is authoritative; fall back to profile, then the
        # Hermes overlay (e.g. moa → "virtual"), then api_key.
        auth_type = (
            (getattr(cfg, "auth_type", "") if cfg else "")
            or (getattr(prof, "auth_type", "") if prof else "")
            or (getattr(overlay, "auth_type", "") if overlay else "")
            or "api_key"
        )

        # Credential env vars: registry first (it already normalizes these),
        # else derive from the profile's env_vars tuple.
        if cfg and getattr(cfg, "api_key_env_vars", ()):
            api_key_vars = tuple(cfg.api_key_env_vars)
            base_url_var = getattr(cfg, "base_url_env_var", "") or ""
        elif prof and getattr(prof, "env_vars", ()):
            api_key_vars, base_url_var = _split_env_vars(tuple(prof.env_vars))
        else:
            api_key_vars, base_url_var = (), ""

        label = (
            (getattr(prof, "display_name", "") if prof else "")
            or entry.label
            or slug
        )
        description = (
            (getattr(prof, "description", "") if prof else "")
            or entry.tui_desc
            or label
        )
        signup_url = (getattr(prof, "signup_url", "") if prof else "") or ""
        if not signup_url and api_key_vars:
            info = OPTIONAL_ENV_VARS.get(api_key_vars[0]) or {}
            signup_url = info.get("url") or ""

        out.append(
            ProviderDescriptor(
                slug=slug,
                label=label,
                description=description,
                auth_type=auth_type,
                tab=tab_for_auth_type(auth_type),
                api_key_env_vars=api_key_vars,
                base_url_env_var=base_url_var,
                signup_url=signup_url,
                order=order,
            )
        )
    return out


def provider_catalog_by_slug() -> dict[str, ProviderDescriptor]:
    """Convenience: the catalog keyed by slug."""
    return {d.slug: d for d in provider_catalog()}
