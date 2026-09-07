"""Credential-pool auth subcommands."""

from __future__ import annotations
from hermes_cli.cli_output import line_input

import math
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable
import uuid

from agent.credential_pool import (
    AUTH_TYPE_API_KEY, AUTH_TYPE_OAUTH, CUSTOM_POOL_PREFIX, SOURCE_MANUAL,
    SOURCE_MANUAL_DEVICE_CODE, STATUS_EXHAUSTED, STRATEGY_FILL_FIRST, STRATEGY_ROUND_ROBIN,
    STRATEGY_RANDOM, STRATEGY_LEAST_USED, PooledCredential, _exhausted_until,
    _normalize_custom_pool_name, get_pool_strategy, label_from_token, list_custom_pool_providers,
    load_pool)
import hermes_cli.auth as auth_mod
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli.secret_prompt import masked_secret_prompt


# Providers that support OAuth login in addition to API keys.
_OAUTH_CAPABLE_PROVIDERS = {"anthropic", "nous", "openai-codex", "xai-oauth", "qwen-oauth", "minimax-oauth"}


def _get_custom_provider_entries() -> list[dict]:
    """Return configured provider entries with legacy and canonical pool IDs."""
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config
        config = load_config()
    except Exception:
        return []
    result: list[dict] = []
    for entry in get_compatible_custom_providers(config):
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, str) and name.strip():
            result.append({
                **entry, "name": name.strip(),
                "pool_key": f"{CUSTOM_POOL_PREFIX}{_normalize_custom_pool_name(name)}",
                "provider_key": str(entry.get("provider_key", "") or "").strip()})
    return result


def _configured_provider_entry(provider: str) -> dict | None:
    """Resolve a canonical ``providers.<key>`` entry."""
    normalized = (provider or "").strip().lower()
    if not normalized or normalized.startswith(CUSTOM_POOL_PREFIX):
        return None
    return next((e for e in _get_custom_provider_entries() if e["provider_key"].lower() == normalized), None)


def _resolve_custom_provider_input(raw: str) -> str | None:
    """Resolve legacy names and keyed providers to their credential-pool ID."""
    normalized = (raw or "").strip().lower().replace(" ", "-")
    if not normalized:
        return None
    if normalized.startswith(CUSTOM_POOL_PREFIX):
        return normalized
    for entry in _get_custom_provider_entries():
        # ``providers:`` entries already have a durable runtime slug; keep credentials under it
        # instead of leaking the legacy ``custom:`` identity into auth.json and discovery.
        provider_key = entry["provider_key"].lower()
        if provider_key and provider_key == normalized:
            return provider_key
        if _normalize_custom_pool_name(entry["name"]) == normalized:
            return provider_key or entry["pool_key"]
    return None


_PROVIDER_ALIASES = {
    "or": "openrouter", "open-router": "openrouter", "grok-oauth": "xai-oauth",
    "xai-oauth": "xai-oauth", "x-ai-oauth": "xai-oauth", "xai-grok-oauth": "xai-oauth"}


def _normalize_provider(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    return _PROVIDER_ALIASES.get(normalized) or _resolve_custom_provider_input(normalized) or normalized


def _migrate_legacy_custom_pool_key(provider: str, legacy_key: str) -> None:
    """Move a keyed provider's old ``custom:`` pool into its runtime slug."""
    with auth_mod._auth_store_lock():
        auth_store = auth_mod._load_auth_store()
        credential_pool = auth_store.get("credential_pool")
        if not isinstance(credential_pool, dict):
            return
        legacy_entries = credential_pool.get(legacy_key)
        if not isinstance(legacy_entries, list) or not legacy_entries:
            return
        current_entries = credential_pool.get(provider)
        merged = list(current_entries) if isinstance(current_entries, list) else []
        known_ids = {e.get("id") for e in merged if isinstance(e, dict) and e.get("id")}
        for entry in legacy_entries:
            entry_id = entry.get("id") if isinstance(entry, dict) else None
            if not entry_id or entry_id not in known_ids:
                merged.append(entry)
                if entry_id:
                    known_ids.add(entry_id)
        credential_pool[provider] = merged
        del credential_pool[legacy_key]
        auth_mod._save_auth_store(auth_store)
    try:
        from hermes_cli.models import clear_provider_models_cache
        clear_provider_models_cache(legacy_key)
    except Exception:
        pass


def _provider_base_url(provider: str) -> str:
    if provider == "openrouter":
        return OPENROUTER_BASE_URL
    if provider.startswith(CUSTOM_POOL_PREFIX):
        from agent.credential_pool import _get_custom_provider_config
        return str((_get_custom_provider_config(provider) or {}).get("base_url") or "").strip()
    configured = _configured_provider_entry(provider)
    if configured is not None:
        return str(configured.get("base_url") or "").strip()
    pconfig = PROVIDER_REGISTRY.get(provider)
    return pconfig.inference_base_url if pconfig else ""


def _is_known_provider(provider: str, configured_provider: dict | None) -> bool:
    return (provider in PROVIDER_REGISTRY or provider == "openrouter"
            or provider.startswith(CUSTOM_POOL_PREFIX) or configured_provider is not None)


def _display_source(source: str) -> str:
    return source.split(":", 1)[1] if source.startswith("manual:") else source


# (label, show_retry_window, http codes, reason substrings, message substrings) — first match wins.
_EXHAUSTED_CLASSES = (
    ("rate-limited", True, {429},
     ("rate_limit", "usage_limit", "quota", "exhausted"),
     ("rate limit", "usage limit", "quota", "too many requests")),
    ("auth failed", False, {401, 403},
     ("invalid_token", "invalid_grant", "unauthorized", "forbidden", "auth"),
     ("unauthorized", "forbidden", "expired", "revoked", "invalid token", "authentication")))


def _classify_exhausted_status(entry) -> tuple[str, bool]:
    code = getattr(entry, "last_error_code", None)
    reason = str(getattr(entry, "last_error_reason", "") or "").strip().lower()
    message = str(getattr(entry, "last_error_message", "") or "").strip().lower()
    for label, retry_window, codes, reason_tokens, message_tokens in _EXHAUSTED_CLASSES:
        if (code in codes or any(t in reason for t in reason_tokens)
                or any(t in message for t in message_tokens)):
            return label, retry_window
    return "exhausted", True


def _format_exhausted_status(entry) -> str:
    if entry.last_status != STATUS_EXHAUSTED:
        return ""
    label, show_retry_window = _classify_exhausted_status(entry)
    reason = getattr(entry, "last_error_reason", None)
    reason_text = f" {reason}" if isinstance(reason, str) and reason.strip() else ""
    code = f" ({entry.last_error_code})" if entry.last_error_code else ""
    head = f" {label}{reason_text}{code}"
    if not show_retry_window:
        return f"{head} (re-auth may be required)"
    exhausted_until = _exhausted_until(entry)
    if exhausted_until is None:
        return head
    remaining = max(0, int(math.ceil(exhausted_until - time.time())))
    if remaining <= 0:
        return f"{head} (ready to retry)"
    minutes, seconds = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = [(days, "d"), (hours, "h"), (minutes, "m"), (seconds, "s")]
    first = next(i for i, (value, _) in enumerate(parts) if value or i == 3)
    wait = " ".join(f"{value}{unit}" for value, unit in parts[first:first + 2])
    return f"{head} ({wait} left)"


def _anthropic_oauth_login(args) -> dict:
    from agent import anthropic_credentials as anthropic_mod
    creds = anthropic_mod.run_hermes_oauth_login_pure()
    if not creds:
        raise SystemExit("Anthropic OAuth login did not return credentials.")
    return creds


def _qwen_oauth_login(args) -> dict:
    from hermes_cli.auth_qwen import _mark_qwen_oauth_active

    creds = auth_mod.resolve_qwen_runtime_credentials(refresh_if_expiring=False)
    _mark_qwen_oauth_active(creds)
    return creds


@dataclass(frozen=True)
class _OAuthAddSpec:
    """Per-provider parameters for the generic ``hermes auth add <provider> --type oauth`` path."""

    login: Callable[[Any], dict]
    token: Callable[[dict], str]
    source: str
    fields: Callable[[dict, str], dict]
    activate_first: bool = False


_OAUTH_ADD_SPECS: dict[str, _OAuthAddSpec] = {
    "anthropic": _OAuthAddSpec(
        login=_anthropic_oauth_login,
        token=lambda creds: creds["access_token"],
        source=f"{SOURCE_MANUAL}:hermes_pkce",
        fields=lambda creds, provider: {
            "refresh_token": creds.get("refresh_token"),
            "expires_at_ms": creds.get("expires_at_ms"),
            "base_url": _provider_base_url(provider)}),
    "openai-codex": _OAuthAddSpec(
        login=lambda args: auth_mod._codex_device_code_login(),
        token=lambda creds: creds["tokens"]["access_token"],
        source=SOURCE_MANUAL_DEVICE_CODE,
        fields=lambda creds, provider: {
            "refresh_token": creds["tokens"].get("refresh_token"),
            "base_url": creds.get("base_url"),
            "last_refresh": creds.get("last_refresh")},
        activate_first=True),
    "xai-oauth": _OAuthAddSpec(
        login=lambda args: auth_mod._xai_oauth_device_code_login(
            timeout_seconds=getattr(args, "timeout", None) or 20.0,
            open_browser=not getattr(args, "no_browser", False)),
        token=lambda creds: creds["tokens"]["access_token"],
        source=SOURCE_MANUAL_DEVICE_CODE,
        fields=lambda creds, provider: {
            "refresh_token": creds["tokens"].get("refresh_token"),
            "base_url": creds.get("base_url") or auth_mod.DEFAULT_XAI_OAUTH_BASE_URL,
            "last_refresh": creds.get("last_refresh")},
        activate_first=True),
    "qwen-oauth": _OAuthAddSpec(
        login=_qwen_oauth_login,
        token=lambda creds: creds["api_key"],
        source=f"{SOURCE_MANUAL}:qwen_cli",
        fields=lambda creds, provider: {"base_url": creds.get("base_url")}),
    "minimax-oauth": _OAuthAddSpec(
        login=lambda args: auth_mod._minimax_oauth_login(
            open_browser=not getattr(args, "no_browser", False),
            timeout_seconds=getattr(args, "timeout", None) or 15.0),
        token=lambda creds: creds["access_token"],
        source=f"{SOURCE_MANUAL}:minimax_oauth",
        fields=lambda creds, provider: {
            "refresh_token": creds.get("refresh_token"), "base_url": creds.get("inference_base_url")}),
}


def _ask(prompt: str, reader: Callable[[str], str] | None = None) -> str | None:
    """Stripped answer from *reader* (default ``input``); None when the user hits EOF / Ctrl-C."""
    try:
        return (reader or input)(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None


def _add_nous_oauth_credential(args, provider: str) -> None:
    """``hermes auth add nous --type oauth``: shared-credential import, else device-code login."""
    custom_label = (getattr(args, "label", None) or "").strip() or None
    timeout = getattr(args, "timeout", None) or 15.0

    def _persist(creds: dict, what: str) -> None:
        # `--label` is embedded into providers.nous so label_from_token doesn't overwrite it on every
        # subsequent load_pool("nous").
        entry = auth_mod.persist_nous_credentials(creds, label=custom_label)
        shown_label = entry.label if entry is not None else label_from_token(
            creds.get("access_token", ""), f"{provider}-oauth-1")
        print(f'{what} {provider} OAuth {"device-code " if what == "Saved" else ""}credentials: "{shown_label}"')

    # Codex-style auto-import: a shared Nous credential at <hermes-root>/shared/nous_auth.json
    # (written by any previous login) makes `hermes --profile <name> auth add nous --type oauth`
    # a one-tap operation for multi-profile users.
    if auth_mod._read_shared_nous_state():
        try:
            found = f"Found existing Nous OAuth credentials at {auth_mod._nous_shared_store_path()}"
        except RuntimeError:
            found = "Found existing shared Nous OAuth credentials"
        print()
        print(found)
        do_import = _ask("Import these credentials? [Y/n]: ")
        if do_import is None or do_import.lower() in {"", "y", "yes"}:
            print("Rehydrating Nous session from shared credentials...")
            rehydrated = auth_mod._try_import_shared_nous_state(timeout_seconds=timeout)
            if rehydrated is not None:
                _persist(rehydrated, "Imported")
                return
            # Expired refresh_token, portal down, etc. — fall through to device-code.
            print("Could not refresh shared credentials — falling back to device-code login.")

    creds = auth_mod._nous_device_code_login(
        portal_base_url=getattr(args, "portal_url", None),
        inference_base_url=getattr(args, "inference_url", None),
        client_id=getattr(args, "client_id", None), scope=getattr(args, "scope", None),
        open_browser=not getattr(args, "no_browser", False), timeout_seconds=timeout,
        insecure=bool(getattr(args, "insecure", False)), ca_bundle=getattr(args, "ca_bundle", None))
    _persist(creds, "Saved")


def _unsuppress_provider_sources(provider: str) -> None:
    """Clear ALL suppressions for this provider — re-adding a credential is a strong signal the
    user wants auth re-enabled. Covers env:* (shell-exported vars), gh_cli (copilot), claude_code,
    qwen-cli, device_code (codex), etc. — one consistent re-engagement pattern."""
    try:
        suppressed = auth_mod._load_auth_store().get("suppressed_sources", {})
        for src in list(suppressed.get(provider, []) or []):
            auth_mod.unsuppress_credential_source(provider, src)
    except Exception:
        pass


def _add_api_key_credential(args, provider: str, pool) -> None:
    token = ((getattr(args, "api_key", None) or "").strip()
             or masked_secret_prompt("Paste your API key: ").strip())
    if not token:
        raise SystemExit("No API key provided.")
    default_label = f"api-key-{len(pool.entries()) + 1}"
    label = (getattr(args, "label", None) or "").strip()
    if not label and sys.stdin.isatty():
        label = line_input(f"Label (optional, default: {default_label}): ").strip()
    label = label or default_label
    entry = PooledCredential(
        provider=provider, id=uuid.uuid4().hex[:6], label=label, auth_type=AUTH_TYPE_API_KEY,
        priority=0, source=SOURCE_MANUAL, access_token=token, base_url=_provider_base_url(provider))
    pool.add_entry(entry)
    print(f'Added {provider} credential #{len(pool.entries())}: "{label}"')


def auth_add_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    configured_provider = _configured_provider_entry(provider)
    if not _is_known_provider(provider, configured_provider):
        raise SystemExit(f"Unknown provider: {provider}")
    if configured_provider is not None:
        _migrate_legacy_custom_pool_key(provider, configured_provider["pool_key"])

    is_custom = provider.startswith(CUSTOM_POOL_PREFIX)
    requested_type = str(getattr(args, "auth_type", "") or "").strip().lower()
    if requested_type == "api-key":
        requested_type = AUTH_TYPE_API_KEY
    elif not requested_type:
        oauth_default = provider in _OAUTH_CAPABLE_PROVIDERS and not is_custom
        requested_type = AUTH_TYPE_OAUTH if oauth_default else AUTH_TYPE_API_KEY

    pool = load_pool(provider)
    if not is_custom:
        _unsuppress_provider_sources(provider)

    if requested_type == AUTH_TYPE_API_KEY:
        _add_api_key_credential(args, provider, pool)
        return
    if provider == "nous":
        _add_nous_oauth_credential(args, provider)
        return

    spec = _OAUTH_ADD_SPECS.get(provider)
    if spec is None:
        raise SystemExit(f"`hermes auth add {provider}` is not implemented for auth type {requested_type} yet.")

    creds = spec.login(args)
    token = spec.token(creds)
    label = (getattr(args, "label", None) or "").strip() or label_from_token(
        token, f"{provider}-oauth-{len(pool.entries()) + 1}")
    # Every account gets a distinct, self-contained pool entry instead of routing through a
    # singleton save path (which collapsed every added account into the latest login).
    # ``manual:*`` entries refresh from their own token pair, so they need no singleton shadow.
    entry = PooledCredential(
        provider=provider, id=uuid.uuid4().hex[:6], label=label, auth_type=AUTH_TYPE_OAUTH, priority=0,
        source=spec.source, access_token=token, **spec.fields(creds, provider))
    first_credential = not pool.entries()
    pool.add_entry(entry)
    # The first Codex/xAI credential becomes the active provider (as the old singleton save path
    # did implicitly); subsequent adds leave the active provider as-is.
    if spec.activate_first and first_credential:
        auth_mod.mark_provider_active_if_unset(provider)
    print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')


def auth_list_command(args) -> None:
    provider_filter = _normalize_provider(getattr(args, "provider", "") or "")
    if provider_filter:
        providers = [provider_filter]
    else:
        credential_pool = auth_mod._load_auth_store().get("credential_pool")
        providers = sorted({
            *PROVIDER_REGISTRY.keys(), "openrouter", *list_custom_pool_providers(),
            *(e["provider_key"] for e in _get_custom_provider_entries() if e["provider_key"]),
            *(credential_pool.keys() if isinstance(credential_pool, dict) else ())})
    for provider in providers:
        pool = load_pool(provider)
        entries = pool.entries()
        if not entries:
            continue
        current = pool.peek()
        print(f"{provider} ({len(entries)} credentials):")
        for idx, entry in enumerate(entries, start=1):
            marker = "← " if current is not None and entry.id == current.id else "  "
            status = _format_exhausted_status(entry)
            source = _display_source(entry.source)
            print(f"  #{idx}  {entry.label:<20} {entry.auth_type:<7} {source}{status} {marker}".rstrip())
        print()
    _print_oauth_heal_notices()


def _print_oauth_heal_notices() -> None:
    """Tell the user when load_pool() just consolidated a forked OAuth grant."""
    for note in auth_mod.consume_oauth_heal_notices():
        print(f"note: {note}")


def auth_remove_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    target = getattr(args, "target", None)
    target = getattr(args, "index", None) if target is None else target
    pool = load_pool(provider)
    index, matched, error = pool.resolve_target(target)
    if matched is None or index is None:
        raise SystemExit(f"{error} Provider: {provider}.")
    removed = pool.remove_index(index)
    if removed is None:
        raise SystemExit(f'No credential matching "{target}" for provider {provider}.')
    print(f"Removed {provider} credential #{index} ({removed.label})")

    # Every credential source Hermes reads from (env vars, external OAuth files, auth.json blocks,
    # custom config) has a RemovalStep in agent.credential_sources; it does the source-specific
    # cleanup while suppression + user-facing output are centralised here.
    from agent.credential_sources import find_removal_step
    step = find_removal_step(provider, removed.source)
    if step is None:  # unregistered source, e.g. "manual": nothing external to clean up
        return
    result = step.remove_fn(provider, removed)
    for line in result.cleaned:
        print(line)
    if result.suppress:
        auth_mod.suppress_credential_source(provider, removed.source)
    for line in result.hints:
        print(line)


def auth_reset_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    pool = load_pool(provider)
    count = pool.reset_statuses()
    print(f"Reset status on {count} {provider} credentials")


def auth_status_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", "") or "")
    if not provider:
        raise SystemExit("Provider is required. Example: `hermes auth status spotify`.")
    if provider in auth_mod.SINGLE_USE_REFRESH_POOL_PROVIDERS:
        load_pool(provider)  # runs the forked-grant heal first so the report reflects the consolidated grant
    status = auth_mod.get_auth_status(provider)
    _print_oauth_heal_notices()
    if not status.get("logged_in"):
        reason = status.get("error")
        print(f"{provider}: logged out" + (f" ({reason})" if reason else ""))
        return
    print(f"{provider}: logged in")
    for key in ("auth_type", "client_id", "redirect_uri", "scope", "expires_at", "api_base_url"):
        value = status.get(key)
        if value:
            print(f"  {key}: {value}")


def auth_logout_command(args) -> None:
    auth_mod.logout_command(SimpleNamespace(provider=getattr(args, "provider", None)))


def auth_spotify_command(args) -> None:
    action = str(getattr(args, "spotify_action", "") or "login").strip().lower()
    if action in {"", "login"}:
        auth_mod.login_spotify_command(args)
        return
    handler = {"status": auth_status_command, "logout": auth_logout_command}.get(action)
    if handler is None:
        raise SystemExit(f"Unknown Spotify auth action: {action}")
    handler(SimpleNamespace(provider="spotify"))


def _print_bedrock_status() -> None:
    """Show AWS Bedrock credential status (not in the pool — uses boto3 chain)."""
    try:
        from agent.bedrock_adapter import has_aws_credentials, resolve_aws_auth_env_var, resolve_bedrock_region
        if not has_aws_credentials():
            return
        region = resolve_bedrock_region()
        print("bedrock (AWS SDK credential chain):")
        print(f"  Auth: {resolve_aws_auth_env_var() or 'unknown'}")
        print(f"  Region: {region}")
        try:
            import boto3
            arn = boto3.client("sts", region_name=region).get_caller_identity().get("Arn", "unknown")
            print(f"  Identity: {arn}")
        except Exception:
            print("  Identity: (could not resolve — boto3 STS call failed)")
        print()
    except ImportError:
        pass  # boto3 or bedrock_adapter not available


def _print_azure_entra_status() -> None:
    """Show Azure Foundry Entra ID status when model.provider is azure-foundry with entra_id auth."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
        if not isinstance(model_cfg, dict) or (
            str(model_cfg.get("provider") or "").strip().lower() != "azure-foundry"
            or str(model_cfg.get("auth_mode") or "").strip().lower() != "entra_id"):
            return
        from agent.azure_identity_adapter import (
            EntraIdentityConfig, SCOPE_AI_AZURE_DEFAULT, describe_active_credential, has_azure_identity_installed,
        )
        base_url = str(model_cfg.get("base_url") or "").strip()
        entra = model_cfg.get("entra") or {}
        scope = (str(entra.get("scope") or "").strip() if isinstance(entra, dict) else "") or SCOPE_AI_AZURE_DEFAULT
        print("azure-foundry (Microsoft Entra ID):")
        print(f"  Endpoint: {base_url or '(not configured)'}")
        print(f"  Scope: {scope}")
        if not has_azure_identity_installed():
            print("  Status: ⚠ azure-identity not installed (pip install azure-identity)")
        else:
            info = describe_active_credential(config=EntraIdentityConfig(scope=scope), timeout_seconds=10.0)
            env_sources = info.get("env_sources") or []
            if info.get("ok"):
                print(f"  Status: ✓ token acquired ({', '.join(env_sources) if env_sources else 'default chain'})")
            else:
                print(f"  Status: ⚠ {info.get('error') or 'credential chain exhausted'}")
                if info.get("hint"):
                    print(f"  Hint: {info['hint']}")
        print()
    except Exception:
        pass


def _interactive_auth() -> None:
    """Interactive credential pool management when `hermes auth` is called bare."""
    print("Credential Pool Status")
    print("=" * 50)
    auth_list_command(SimpleNamespace(provider=None))
    _print_bedrock_status()
    _print_azure_entra_status()
    print()

    choices = [
        "Add a credential", "Remove a credential", "Reset cooldowns for a provider",
        "Set rotation strategy for a provider", "Exit"]
    print("What would you like to do?")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    raw = _ask("\nChoice: ")
    handler = {"1": _interactive_add, "2": _interactive_remove, "3": _interactive_reset,
               "4": _interactive_strategy}.get(raw)
    if handler is not None:
        handler()


def _pick_provider(prompt: str = "Provider") -> str:
    """Prompt for a provider name with auto-complete hints."""
    known = sorted(set(list(PROVIDER_REGISTRY.keys()) + ["openrouter"]))
    custom_display = [entry["name"] for entry in _get_custom_provider_entries()]
    print(f"\nKnown providers: {', '.join(known)}")
    if custom_display:
        print(f"Custom endpoints: {', '.join(custom_display)}")
    raw = _ask(f"{prompt}: ", line_input)
    if raw is None:
        raise SystemExit()
    return _normalize_provider(raw)


def _interactive_add() -> None:
    provider = _pick_provider("Provider to add credential for")
    configured_provider = _configured_provider_entry(provider)
    if not _is_known_provider(provider, configured_provider):
        raise SystemExit(f"Unknown provider: {provider}")

    auth_type = "api_key"
    if provider in _OAUTH_CAPABLE_PROVIDERS:
        print(f"\n{provider} supports both API keys and OAuth login.")
        print("  1. API key (paste a key from the provider dashboard)")
        print("  2. OAuth login (authenticate via browser)")
        type_choice = _ask("Type [1/2]: ")
        if type_choice is None:
            return
        if type_choice == "2":
            auth_type = "oauth"
    label = _ask("Label / account name (optional): ", line_input)
    if label is None:
        return
    auth_add_command(SimpleNamespace(
        provider=provider, auth_type=auth_type, label=label or None, api_key=None,
        portal_url=None, inference_url=None, client_id=None, scope=None,
        no_browser=False, timeout=None, insecure=False, ca_bundle=None))


def _interactive_remove() -> None:
    provider = _pick_provider("Provider to remove credential from")
    pool = load_pool(provider)
    if not pool.has_credentials():
        print(f"No credentials for {provider}.")
        return
    for i, e in enumerate(pool.entries(), 1):
        print(f"  #{i}  {e.label:25s} {e.auth_type:10s} {e.source}{_format_exhausted_status(e)} [id:{e.id}]")
    raw = _ask("Remove #, id, or label (blank to cancel): ", line_input)
    if raw:
        auth_remove_command(SimpleNamespace(provider=provider, target=raw))


def _interactive_reset() -> None:
    auth_reset_command(SimpleNamespace(provider=_pick_provider("Provider to reset cooldowns for")))


_STRATEGY_DESCRIPTIONS = {
    STRATEGY_FILL_FIRST: "Use first key until exhausted, then next",
    STRATEGY_ROUND_ROBIN: "Cycle through keys evenly",
    STRATEGY_LEAST_USED: "Always pick the least-used key",
    STRATEGY_RANDOM: "Random selection"}


def _interactive_strategy() -> None:
    provider = _pick_provider("Provider to set strategy for")
    current = get_pool_strategy(provider)
    strategies = list(_STRATEGY_DESCRIPTIONS)

    print(f"\nCurrent strategy for {provider}: {current}")
    print()
    for i, s in enumerate(strategies, 1):
        print(f"  {i}. {s:15s} — {_STRATEGY_DESCRIPTIONS[s]}{' ←' if s == current else ''}")
    raw = _ask("\nStrategy [1-4]: ")
    if not raw:
        return
    try:
        strategy = strategies[int(raw) - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    pool_strategies = cfg.get("credential_pool_strategies")
    if not isinstance(pool_strategies, dict):
        pool_strategies = {}
    pool_strategies[provider] = strategy
    cfg["credential_pool_strategies"] = pool_strategies
    save_config(cfg)
    print(f"Set {provider} strategy to: {strategy}")


_AUTH_ACTIONS = {
    "add": auth_add_command, "list": auth_list_command, "remove": auth_remove_command,
    "reset": auth_reset_command, "status": auth_status_command, "logout": auth_logout_command,
    "spotify": auth_spotify_command}


def auth_command(args) -> None:
    handler = _AUTH_ACTIONS.get(getattr(args, "auth_action", ""))
    if handler is not None:
        handler(args)
    else:
        _interactive_auth()  # no subcommand
