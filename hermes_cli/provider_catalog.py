"""Unified provider catalog — one source of truth for the provider universe.

The provider list shown by ``hermes model`` (CLI/TUI) and the desktop Settings → Providers tabs
(Accounts + API keys) **must be the same set**; providers added after those lists were written
silently went missing from the GUI. ``auth_type`` / ``api_key_env_vars`` / ``base_url_env_var``
come from :data:`hermes_cli.auth.PROVIDER_REGISTRY` (credential truth); ``display_name`` /
``description`` / ``signup_url`` from the provider's :class:`providers.base.ProviderProfile`, falling
back to the ``CANONICAL_PROVIDERS`` entry's ``label`` / ``tui_desc`` and the ``OPTIONAL_ENV_VARS``
signup URL (many profiles leave these blank, and lmstudio, openai-api, tencent-tokenhub, xai-oauth
have no profile at all — the fallbacks are load-bearing).
"""

from __future__ import annotations

from dataclasses import dataclass

# Auth types that authenticate via an account / sign-in flow rather than a pasted API key; these
# route to the desktop "Accounts" tab, everything else (api_key, and aws_sdk configured via
# AWS_REGION/AWS_PROFILE) to "API keys". Mirrors the auth_type strings in PROVIDER_REGISTRY and
# ProviderProfile: external_process = copilot-acp (spawns `copilot --acp --stdio`), copilot = GitHub
# Copilot token / gh auth.
_ACCOUNTS_AUTH_TYPES: frozenset[str] = frozenset(
    {"oauth_device_code", "oauth_external", "oauth_minimax", "external_process", "copilot"}
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


def _is_url_var(name: str) -> bool:
    return name.endswith("_BASE_URL") or name.endswith("_URL")


def _split_env_vars(env_vars: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """Split a profile's ``env_vars`` into (api_key_vars, base_url_var)."""
    return tuple(v for v in env_vars if not _is_url_var(v)), next((v for v in env_vars if _is_url_var(v)), "")


def _safe_import(module: str, attr: str, default):
    """Import ``attr`` from ``module``; ``default`` on ANY failure — this module is on the import
    path of the web server and the CLI, and a provider-plugin import error must never blank the
    whole catalog."""
    try:
        return getattr(__import__(module, fromlist=[attr]), attr)
    except Exception:
        return default


def provider_catalog() -> list[ProviderDescriptor]:
    """One descriptor per provider in the ``hermes model`` universe (:data:`CANONICAL_PROVIDERS`,
    auto-extended by provider plugins). Auth/env from ``PROVIDER_REGISTRY``; display metadata from
    ``ProviderProfile`` with canonical/env fallbacks so profile-less providers still resolve."""
    from hermes_cli.models import CANONICAL_PROVIDERS
    PROVIDER_REGISTRY = _safe_import("hermes_cli.auth", "PROVIDER_REGISTRY", {})
    OPTIONAL_ENV_VARS = _safe_import("hermes_cli.config", "OPTIONAL_ENV_VARS", {})
    # Overlays carry auth_type for providers with no registry/profile entry — notably the ``moa``
    # virtual provider (auth_type "virtual"), which has no credential and no network endpoint.
    HERMES_OVERLAYS = _safe_import("hermes_cli.providers", "HERMES_OVERLAYS", {})
    try:
        from providers import list_providers
        profiles = {p.name: p for p in list_providers()}
    except Exception:
        profiles = {}
    out: list[ProviderDescriptor] = []
    for order, entry in enumerate(CANONICAL_PROVIDERS):
        slug = entry.slug
        cfg = PROVIDER_REGISTRY.get(slug)
        prof = profiles.get(slug)
        overlay = HERMES_OVERLAYS.get(slug)
        # auth_type: registry is authoritative; then profile, then overlay (moa → "virtual"), then api_key.
        auth_type = ((cfg.auth_type if cfg else "") or (prof.auth_type if prof else "")
                     or (overlay.auth_type if overlay else "") or "api_key")
        # Credential env vars: registry first (already normalized), else derived from the profile.
        if cfg and cfg.api_key_env_vars:
            api_key_vars, base_url_var = tuple(cfg.api_key_env_vars), cfg.base_url_env_var or ""
        elif prof and prof.env_vars:
            api_key_vars, base_url_var = _split_env_vars(tuple(prof.env_vars))
        else:
            api_key_vars, base_url_var = (), ""
        label = (prof.display_name if prof else "") or entry.label or slug
        signup_url = (prof.signup_url if prof else "") or ""
        if not signup_url and api_key_vars:
            signup_url = (OPTIONAL_ENV_VARS.get(api_key_vars[0]) or {}).get("url") or ""
        out.append(
            ProviderDescriptor(
                slug=slug, label=label, description=(prof.description if prof else "") or entry.tui_desc or label,
                auth_type=auth_type, tab=tab_for_auth_type(auth_type), api_key_env_vars=api_key_vars,
                base_url_env_var=base_url_var, signup_url=signup_url, order=order,
            )
        )
    return out


def provider_catalog_by_slug() -> dict[str, ProviderDescriptor]:
    """Convenience: the catalog keyed by slug."""
    return {d.slug: d for d in provider_catalog()}


# ---------------------------------------------------------------------------
# Login-flow metadata
# ---------------------------------------------------------------------------
#
# These rows used to live only inside ``hermes_cli/web_server_oauth.py`` as
# ``_OAUTH_PROVIDER_CATALOG`` — welded to the dashboard process and unreachable from any
# surface that does not run it. The module docstring above already names that tuple as one
# of the hand-maintained lists this module exists to unify, so the DATA moves here and the
# dashboard keeps only the binding of its per-provider ``status_fn`` callables.
#
# ``flow`` describes the login SHAPE so any client can pick the right UI:
#   ``pkce``        — open a URL, paste the callback code back
#   ``device_code`` — show a user code + verification URI, poll for the token
#   ``external``    — delegated to a third-party CLI; Hermes only reads it
#
# Two rows are deliberately NOT catalog providers but must still be offered as sign-ins:
# the Anthropic account card and the synthetic ``claude-code`` subscription row.
OAUTH_FLOW_OVERRIDES: tuple[dict, ...] = ({'id': 'nous',
  'name': 'Nous Portal',
  'flow': 'device_code',
  'cli_command': 'hermes auth add nous',
  'docs_url': 'https://portal.nousresearch.com'},
 {'id': 'openai-codex',
  'name': 'ChatGPT or Codex Subscription',
  'flow': 'device_code',
  'cli_command': 'hermes auth add openai-codex',
  'docs_url': 'https://platform.openai.com/docs'},
 {'id': 'qwen-oauth',
  'name': 'Qwen (via Qwen CLI)',
  'flow': 'external',
  'cli_command': 'hermes auth add qwen-oauth',
  'docs_url': 'https://github.com/QwenLM/qwen-code'},
 {'id': 'minimax-oauth',
  'name': 'MiniMax (OAuth)',
  'flow': 'device_code',
  'cli_command': 'hermes auth add minimax-oauth',
  'docs_url': 'https://www.minimax.io'},
 {'id': 'xai-oauth',
  'name': 'xAI Grok OAuth (SuperGrok / Premium+)',
  'flow': 'device_code',
  'cli_command': 'hermes auth add xai-oauth',
  'docs_url': 'https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth'},
 {'id': 'copilot-acp',
  'name': 'GitHub Copilot (ACP)',
  'flow': 'external',
  'cli_command': 'copilot login',
  'docs_url': 'https://docs.github.com/en/copilot'},
 {'id': 'anthropic',
  'name': 'Anthropic Account',
  'flow': 'external',
  'cli_command': 'hermes auth add anthropic',
  'docs_url': 'https://docs.claude.com/en/api/getting-started'},
 {'id': 'claude-code',
  'name': 'Anthropic OAuth: Required Extra Usage Credits to Use Subscription',
  'flow': 'external',
  'cli_command': 'claude setup-token',
  'docs_url': 'https://docs.claude.com/en/docs/claude-code'})


def disconnect_command_for(slug: str, flow: str, platform: str | None = None) -> str | None:
    """The documented command that clears an EXTERNAL provider's credentials.

    External providers store credentials outside Hermes, so Hermes never
    deletes them behind a silent API call — it hands the operator the exact
    command instead. Returns None for providers we cannot safely clear (the UI
    shows a manual hint) and for every non-external flow.

    Claude Code has no scriptable logout (only the interactive ``/logout``), so
    the command removes the same two sources ``read_claude_code_credentials()``
    consults, host-native per ``platform`` (default: this process).
    """
    if flow != "external":
        return None
    if slug == "claude-code":
        import sys as _sys

        host = platform or _sys.platform
        if host == "win32":
            literal = '"$HOME/.claude/.credentials.json"'
            return (
                f"if (Test-Path -LiteralPath {literal}) {{ "
                f"Remove-Item -LiteralPath {literal} -Force -ErrorAction Stop }}"
            )
        rm_file = "rm -f ~/.claude/.credentials.json"
        if host == "darwin":
            return (
                'security delete-generic-password -s "Claude Code-credentials" '
                f"2>/dev/null; {rm_file}"
            )
        return rm_file
    return None
