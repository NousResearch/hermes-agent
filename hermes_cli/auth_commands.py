"""Credential-pool auth subcommands."""

from __future__ import annotations

import math
import sys
import time
from types import SimpleNamespace
import uuid

from agent.account_usage import fetch_codex_account_usage_for_token, render_account_usage_lines
from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OAUTH,
    CUSTOM_POOL_PREFIX,
    SOURCE_MANUAL,
    SOURCE_MANUAL_DEVICE_CODE,
    STATUS_DEAD,
    STATUS_EXHAUSTED,
    STRATEGY_FILL_FIRST,
    STRATEGY_ROUND_ROBIN,
    STRATEGY_RANDOM,
    STRATEGY_LEAST_USED,
    PooledCredential,
    _exhausted_until,
    _normalize_custom_pool_name,
    get_pool_strategy,
    label_from_token,
    list_custom_pool_providers,
    load_pool,
)
import hermes_cli.auth as auth_mod
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli.secret_prompt import masked_secret_prompt


# Providers that support OAuth login in addition to API keys.
_OAUTH_CAPABLE_PROVIDERS = {"anthropic", "nous", "openai-codex", "xai-oauth", "qwen-oauth", "minimax-oauth"}


def _get_custom_provider_names() -> list:
    """Return list of (display_name, pool_key, provider_key) tuples."""
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config

        config = load_config()
    except Exception:
        return []
    result = []
    for entry in get_compatible_custom_providers(config):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        pool_key = f"{CUSTOM_POOL_PREFIX}{_normalize_custom_pool_name(name)}"
        provider_key = str(entry.get("provider_key", "") or "").strip()
        result.append((name.strip(), pool_key, provider_key))
    return result


def _resolve_custom_provider_input(raw: str) -> str | None:
    """If raw input matches a custom_providers entry name (case-insensitive), return its pool key."""
    normalized = (raw or "").strip().lower().replace(" ", "-")
    if not normalized:
        return None
    # Direct match on 'custom:name' format
    if normalized.startswith(CUSTOM_POOL_PREFIX):
        return normalized
    for display_name, pool_key, provider_key in _get_custom_provider_names():
        if _normalize_custom_pool_name(display_name) == normalized:
            return pool_key
        if provider_key and provider_key.strip().lower() == normalized:
            return pool_key
    return None


def _normalize_provider(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    if normalized in {"or", "open-router"}:
        return "openrouter"
    if normalized in {"grok-oauth", "xai-oauth", "x-ai-oauth", "xai-grok-oauth"}:
        return "xai-oauth"
    # Check if it matches a custom provider name
    custom_key = _resolve_custom_provider_input(normalized)
    if custom_key:
        return custom_key
    return normalized


def _provider_base_url(provider: str) -> str:
    if provider == "openrouter":
        return OPENROUTER_BASE_URL
    if provider.startswith(CUSTOM_POOL_PREFIX):
        from agent.credential_pool import _get_custom_provider_config

        cp_config = _get_custom_provider_config(provider)
        if cp_config:
            return str(cp_config.get("base_url") or "").strip()
        return ""
    pconfig = PROVIDER_REGISTRY.get(provider)
    return pconfig.inference_base_url if pconfig else ""


def _oauth_default_label(provider: str, count: int) -> str:
    return f"{provider}-oauth-{count}"


def _api_key_default_label(count: int) -> str:
    return f"api-key-{count}"


def _display_source(source: str) -> str:
    return source.split(":", 1)[1] if source.startswith("manual:") else source


def _classify_exhausted_status(entry) -> tuple[str, bool]:
    code = getattr(entry, "last_error_code", None)
    reason = str(getattr(entry, "last_error_reason", "") or "").strip().lower()
    message = str(getattr(entry, "last_error_message", "") or "").strip().lower()

    if code == 429 or any(token in reason for token in ("rate_limit", "usage_limit", "quota", "exhausted")) or any(
        token in message for token in ("rate limit", "usage limit", "quota", "too many requests")
    ):
        return "rate-limited", True

    if code in {401, 403} or any(token in reason for token in ("invalid_token", "invalid_grant", "unauthorized", "forbidden", "auth")) or any(
        token in message for token in ("unauthorized", "forbidden", "expired", "revoked", "invalid token", "authentication")
    ):
        return "auth failed", False

    return "exhausted", True



def _format_exhausted_status(entry) -> str:
    if entry.last_status != STATUS_EXHAUSTED:
        return ""
    label, show_retry_window = _classify_exhausted_status(entry)
    reason = getattr(entry, "last_error_reason", None)
    reason_text = f" {reason}" if isinstance(reason, str) and reason.strip() else ""
    code = f" ({entry.last_error_code})" if entry.last_error_code else ""
    if not show_retry_window:
        return f" {label}{reason_text}{code} (re-auth may be required)"
    exhausted_until = _exhausted_until(entry)
    if exhausted_until is None:
        return f" {label}{reason_text}{code}"
    remaining = max(0, int(math.ceil(exhausted_until - time.time())))
    if remaining <= 0:
        return f" {label}{reason_text}{code} (ready to retry)"
    minutes, seconds = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        wait = f"{days}d {hours}h"
    elif hours:
        wait = f"{hours}h {minutes}m"
    elif minutes:
        wait = f"{minutes}m {seconds}s"
    else:
        wait = f"{seconds}s"
    return f" {label}{reason_text}{code} ({wait} left)"


def auth_add_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    if provider not in PROVIDER_REGISTRY and provider != "openrouter" and not provider.startswith(CUSTOM_POOL_PREFIX):
        raise SystemExit(f"Unknown provider: {provider}")

    requested_type = str(getattr(args, "auth_type", "") or "").strip().lower()
    if requested_type in {AUTH_TYPE_API_KEY, "api-key"}:
        requested_type = AUTH_TYPE_API_KEY
    if not requested_type:
        if provider.startswith(CUSTOM_POOL_PREFIX):
            requested_type = AUTH_TYPE_API_KEY
        else:
            requested_type = AUTH_TYPE_OAUTH if provider in _OAUTH_CAPABLE_PROVIDERS else AUTH_TYPE_API_KEY

    pool = load_pool(provider)

    # Clear ALL suppressions for this provider — re-adding a credential is
    # a strong signal the user wants auth re-enabled.  This covers env:*
    # (shell-exported vars), gh_cli (copilot), claude_code, qwen-cli,
    # device_code (codex), etc.  One consistent re-engagement pattern.
    # Matches the Codex device_code re-link pattern that predates this.
    if not provider.startswith(CUSTOM_POOL_PREFIX):
        try:
            from hermes_cli.auth import (
                _load_auth_store,
                unsuppress_credential_source,
            )
            suppressed = _load_auth_store().get("suppressed_sources", {})
            for src in list(suppressed.get(provider, []) or []):
                unsuppress_credential_source(provider, src)
        except Exception:
            pass

    if requested_type == AUTH_TYPE_API_KEY:
        token = (getattr(args, "api_key", None) or "").strip()
        if not token:
            token = masked_secret_prompt("Paste your API key: ").strip()
        if not token:
            raise SystemExit("No API key provided.")
        default_label = _api_key_default_label(len(pool.entries()) + 1)
        label = (getattr(args, "label", None) or "").strip()
        if not label:
            if sys.stdin.isatty():
                label = input(f"Label (optional, default: {default_label}): ").strip() or default_label
            else:
                label = default_label
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token=token,
            base_url=_provider_base_url(provider),
        )
        pool.add_entry(entry)
        print(f'Added {provider} credential #{len(pool.entries())}: "{label}"')
        return

    if provider == "anthropic":
        from agent import anthropic_adapter as anthropic_mod

        creds = anthropic_mod.run_hermes_oauth_login_pure()
        if not creds:
            raise SystemExit("Anthropic OAuth login did not return credentials.")
        label = (getattr(args, "label", None) or "").strip() or label_from_token(
            creds["access_token"],
            _oauth_default_label(provider, len(pool.entries()) + 1),
        )
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:hermes_pkce",
            access_token=creds["access_token"],
            refresh_token=creds.get("refresh_token"),
            expires_at_ms=creds.get("expires_at_ms"),
            base_url=_provider_base_url(provider),
        )
        pool.add_entry(entry)
        print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')
        return

    if provider == "nous":
        # Codex-style auto-import: if a shared Nous credential lives at
        # <hermes-root>/shared/nous_auth.json (written by any previous
        # successful login), offer to import it instead of running the
        # full device-code flow. This makes `hermes --profile <name>
        # auth add nous --type oauth` a one-tap operation for users who
        # run multiple profiles.
        shared = auth_mod._read_shared_nous_state()
        if shared:
            try:
                path = auth_mod._nous_shared_store_path()
            except RuntimeError:
                path = None
            print()
            if path:
                print(f"Found existing Nous OAuth credentials at {path}")
            else:
                print("Found existing shared Nous OAuth credentials")
            try:
                do_import = input("Import these credentials? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                do_import = "y"
            if do_import in {"", "y", "yes"}:
                print("Rehydrating Nous session from shared credentials...")
                rehydrated = auth_mod._try_import_shared_nous_state(
                    timeout_seconds=getattr(args, "timeout", None) or 15.0,
                )
                if rehydrated is not None:
                    custom_label = (getattr(args, "label", None) or "").strip() or None
                    entry = auth_mod.persist_nous_credentials(rehydrated, label=custom_label)
                    shown_label = entry.label if entry is not None else label_from_token(
                        rehydrated.get("access_token", ""), _oauth_default_label(provider, 1),
                    )
                    print(f'Imported {provider} OAuth credentials: "{shown_label}"')
                    return
                # Rehydrate failed (expired refresh_token, portal down, etc.)
                # — fall through to device-code flow.
                print("Could not refresh shared credentials — falling back to device-code login.")

        creds = auth_mod._nous_device_code_login(
            portal_base_url=getattr(args, "portal_url", None),
            inference_base_url=getattr(args, "inference_url", None),
            client_id=getattr(args, "client_id", None),
            scope=getattr(args, "scope", None),
            open_browser=not getattr(args, "no_browser", False),
            timeout_seconds=getattr(args, "timeout", None) or 15.0,
            insecure=bool(getattr(args, "insecure", False)),
            ca_bundle=getattr(args, "ca_bundle", None),
        )
        # Honor `--label <name>` so nous matches other providers' UX.  The
        # helper embeds this into providers.nous so that label_from_token
        # doesn't overwrite it on every subsequent load_pool("nous").
        custom_label = (getattr(args, "label", None) or "").strip() or None
        entry = auth_mod.persist_nous_credentials(creds, label=custom_label)
        shown_label = entry.label if entry is not None else label_from_token(
            creds.get("access_token", ""), _oauth_default_label(provider, 1),
        )
        print(f'Saved {provider} OAuth device-code credentials: "{shown_label}"')
        return

    if provider == "openai-codex":
        creds = auth_mod._codex_device_code_login()
        access_token = creds["tokens"]["access_token"]
        account_id = (
            auth_mod.extract_codex_account_id(access_token)
            or str(creds["tokens"].get("account_id") or "").strip()
        )
        label = (getattr(args, "label", None) or "").strip() or label_from_token(
            access_token,
            _oauth_default_label(provider, len(pool.entries()) + 1),
        )
        # Add a distinct, self-contained pool entry per account (matching the
        # xai-oauth / qwen-oauth patterns) instead of
        # routing through the singleton ``_save_codex_tokens`` save path.
        # The singleton round-trip collapsed every added account into the
        # latest login: a second ``hermes auth add openai-codex`` overwrote
        # the first account's singleton-mirrored ``device_code`` entry rather
        # than creating an independent one (#39236). ``manual:device_code``
        # entries refresh from their own token pair, so they need no singleton
        # shadow.
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=SOURCE_MANUAL_DEVICE_CODE,
            access_token=access_token,
            refresh_token=creds["tokens"].get("refresh_token"),
            base_url=creds.get("base_url"),
            last_refresh=creds.get("last_refresh"),
            extra={"account_id": account_id} if account_id else {},
        )
        first_credential = not pool.entries()
        pool.add_entry(entry)
        # Adding the first Codex credential should make it the active provider
        # (the old singleton save path did this implicitly via
        # _save_provider_state). Subsequent adds leave the active provider as-is.
        if first_credential:
            auth_mod.mark_provider_active_if_unset(provider)
        print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')
        return

    if provider == "xai-oauth":
        creds = auth_mod._xai_oauth_device_code_login(
            timeout_seconds=getattr(args, "timeout", None) or 20.0,
            open_browser=not getattr(args, "no_browser", False),
        )
        auth_mod._save_xai_oauth_tokens(
            creds["tokens"],
            discovery=creds.get("discovery"),
            redirect_uri=creds.get("redirect_uri", ""),
            last_refresh=creds.get("last_refresh"),
            auth_mode="oauth_device_code",
        )
        pool = load_pool(provider)
        entry = next((e for e in pool.entries() if getattr(e, "source", "") == "device_code"), None)
        shown_label = entry.label if entry is not None else label_from_token(
            creds["tokens"]["access_token"], _oauth_default_label(provider, 1)
        )
        print(f'Saved {provider} OAuth credentials: "{shown_label}"')
        return

    if provider == "qwen-oauth":
        creds = auth_mod.resolve_qwen_runtime_credentials(refresh_if_expiring=False)
        auth_mod._mark_qwen_oauth_active(creds)
        label = (getattr(args, "label", None) or "").strip() or label_from_token(
            creds["api_key"],
            _oauth_default_label(provider, len(pool.entries()) + 1),
        )
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:qwen_cli",
            access_token=creds["api_key"],
            base_url=creds.get("base_url"),
        )
        pool.add_entry(entry)
        print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')
        return

    if provider == "minimax-oauth":
        creds = auth_mod._minimax_oauth_login(
            open_browser=not getattr(args, "no_browser", False),
            timeout_seconds=getattr(args, "timeout", None) or 15.0,
        )
        label = (getattr(args, "label", None) or "").strip() or label_from_token(
            creds["access_token"],
            _oauth_default_label(provider, len(pool.entries()) + 1),
        )
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:minimax_oauth",
            access_token=creds["access_token"],
            refresh_token=creds.get("refresh_token"),
            base_url=creds.get("inference_base_url"),
        )
        pool.add_entry(entry)
        print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')
        return

    raise SystemExit(f"`hermes auth add {provider}` is not implemented for auth type {requested_type} yet.")


def auth_list_command(args) -> None:
    provider_filter = _normalize_provider(getattr(args, "provider", "") or "")
    if provider_filter:
        providers = [provider_filter]
    else:
        providers = sorted({
            *PROVIDER_REGISTRY.keys(),
            "openrouter",
            *list_custom_pool_providers(),
        })
    for provider in providers:
        pool = load_pool(provider)
        entries = pool.entries()
        if not entries:
            continue
        current = pool.peek()
        print(f"{provider} ({len(entries)} credentials):")
        for idx, entry in enumerate(entries, start=1):
            marker = "  "
            if current is not None and entry.id == current.id:
                marker = "← "
            status = _format_exhausted_status(entry)
            source = _display_source(entry.source)
            print(f"  #{idx}  {entry.label:<20} {entry.auth_type:<7} {source}{status} {marker}".rstrip())
        print()


def auth_remove_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    target = getattr(args, "target", None)
    if target is None:
        target = getattr(args, "index", None)
    pool = load_pool(provider)
    index, matched, error = pool.resolve_target(target)
    if matched is None or index is None:
        raise SystemExit(f"{error} Provider: {provider}.")
    removed = pool.remove_index(index)
    if removed is None:
        raise SystemExit(f'No credential matching "{target}" for provider {provider}.')
    print(f"Removed {provider} credential #{index} ({removed.label})")

    # Unified removal dispatch.  Every credential source Hermes reads from
    # (env vars, external OAuth files, auth.json blocks, custom config)
    # has a RemovalStep registered in agent.credential_sources.  The step
    # handles its source-specific cleanup and we centralise suppression +
    # user-facing output here so every source behaves identically from
    # the user's perspective.
    from agent.credential_sources import find_removal_step
    from hermes_cli.auth import suppress_credential_source

    step = find_removal_step(provider, removed.source)
    if step is None:
        # Unregistered source — e.g. "manual", which has nothing external
        # to clean up.  The pool entry is already gone; we're done.
        return

    result = step.remove_fn(provider, removed)
    for line in result.cleaned:
        print(line)
    if result.suppress:
        suppress_credential_source(provider, removed.source)
    for line in result.hints:
        print(line)


def auth_reset_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    pool = load_pool(provider)
    count = pool.reset_statuses()
    print(f"Reset status on {count} {provider} credentials")


def _entry_account_id(entry) -> str | None:
    value = getattr(entry, "account_id", None)
    if value in {None, ""}:
        extra = getattr(entry, "extra", None)
        if isinstance(extra, dict):
            value = extra.get("account_id")
    normalized = str(value or "").strip()
    return normalized or None


def _format_usage_error(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    reason = str(getattr(response, "reason_phrase", "") or "").strip()
    if status_code:
        return f"HTTP {status_code}{f' {reason}' if reason else ''}"
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _maybe_refresh_usage_entry(pool, entry, *, refresh: bool):
    if not refresh:
        return entry, None
    entry_status = getattr(entry, "last_status", None)
    if entry_status == STATUS_DEAD:
        return entry, None
    if entry_status == STATUS_EXHAUSTED:
        exhausted_until = _exhausted_until(entry)
        if exhausted_until is not None and time.time() < exhausted_until:
            return entry, None
    needs_refresh = getattr(pool, "_entry_needs_refresh", None)
    refresh_entry = getattr(pool, "_refresh_entry", None)
    if not callable(needs_refresh) or not callable(refresh_entry):
        return entry, None
    try:
        if not needs_refresh(entry):
            return entry, None
        refreshed = refresh_entry(entry, force=False)
    except Exception as exc:
        return entry, f"token refresh failed: {_format_usage_error(exc)}"
    if refreshed is None:
        return entry, "token refresh failed"
    return refreshed, None


def _print_usage_snapshot_lines(snapshot) -> None:
    rendered = render_account_usage_lines(snapshot)
    body = rendered[2:] if len(rendered) >= 2 else rendered
    if snapshot and getattr(snapshot, "plan", None):
        body = [f"Plan: {snapshot.plan}", *body]
    if not body:
        body = ["No usage windows returned by provider."]
    for line in body:
        print(f"    {line}")


def auth_usage_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", "") or "")
    if provider != "openai-codex":
        raise SystemExit("`hermes auth usage` currently supports pooled usage for openai-codex only.")
    pool = load_pool(provider)
    current_fn = getattr(pool, "current", None)
    current = current_fn() if callable(current_fn) else None
    entries = pool.entries()
    if not entries:
        print("No pooled credentials found for openai-codex.")
        return
    timeout = getattr(args, "timeout", 15.0)
    refresh = not bool(getattr(args, "no_refresh", False))
    print(f"openai-codex account usage ({len(entries)} credentials):")
    for idx, entry in enumerate(entries, start=1):
        marker = " ← current" if current is not None and entry.id == current.id else ""
        source = _display_source(entry.source)
        print(f"  #{idx} {entry.label} [{entry.id}] {entry.auth_type} {source}{marker}")
        if entry.auth_type != AUTH_TYPE_OAUTH:
            print("    Unavailable: Codex account usage requires OAuth credentials.")
            continue
        entry, refresh_error = _maybe_refresh_usage_entry(pool, entry, refresh=refresh)
        if refresh_error:
            print(f"    Unavailable: {refresh_error}")
            continue
        try:
            snapshot = fetch_codex_account_usage_for_token(
                entry.access_token,
                base_url=entry.base_url,
                account_id=_entry_account_id(entry),
                timeout_seconds=timeout,
            )
        except Exception as exc:
            print(f"    Unavailable: {_format_usage_error(exc)}")
            continue
        _print_usage_snapshot_lines(snapshot)


def auth_status_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", "") or "")
    if not provider:
        raise SystemExit("Provider is required. Example: `hermes auth status spotify`.")
    status = auth_mod.get_auth_status(provider)
    if not status.get("logged_in"):
        reason = status.get("error")
        if reason:
            print(f"{provider}: logged out ({reason})")
        else:
            print(f"{provider}: logged out")
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
    if action == "status":
        auth_status_command(SimpleNamespace(provider="spotify"))
        return
    if action == "logout":
        auth_logout_command(SimpleNamespace(provider="spotify"))
        return
    raise SystemExit(f"Unknown Spotify auth action: {action}")


def _interactive_auth() -> None:
    """Interactive credential pool management when `hermes auth` is called bare."""
    # Show current pool status first
    print("Credential Pool Status")
    print("=" * 50)

    auth_list_command(SimpleNamespace(provider=None))

    # Show AWS Bedrock credential status (not in the pool — uses boto3 chain)
    try:
        from agent.bedrock_adapter import has_aws_credentials, resolve_aws_auth_env_var, resolve_bedrock_region
        if has_aws_credentials():
            auth_source = resolve_aws_auth_env_var() or "unknown"
            region = resolve_bedrock_region()
            print("bedrock (AWS SDK credential chain):")
            print(f"  Auth: {auth_source}")
            print(f"  Region: {region}")
            try:
                import boto3
                sts = boto3.client("sts", region_name=region)
                identity = sts.get_caller_identity()
                arn = identity.get("Arn", "unknown")
                print(f"  Identity: {arn}")
            except Exception:
                print("  Identity: (could not resolve — boto3 STS call failed)")
            print()
    except ImportError:
        pass  # boto3 or bedrock_adapter not available

    # Show Azure Foundry Entra ID status
    try:
        from hermes_cli.config import load_config
        _cfg = load_config()
        _model_cfg = _cfg.get("model") if isinstance(_cfg, dict) else None
        if isinstance(_model_cfg, dict):
            _cfg_provider = str(_model_cfg.get("provider") or "").strip().lower()
            _cfg_auth_mode = str(_model_cfg.get("auth_mode") or "").strip().lower()
            if _cfg_provider == "azure-foundry" and _cfg_auth_mode == "entra_id":
                from agent.azure_identity_adapter import (
                    EntraIdentityConfig,
                    SCOPE_AI_AZURE_DEFAULT,
                    describe_active_credential,
                    has_azure_identity_installed,
                )
                _base_url = str(_model_cfg.get("base_url") or "").strip()
                _entra = _model_cfg.get("entra") or {}
                if not isinstance(_entra, dict):
                    _entra = {}
                _scope = (
                    str(_entra.get("scope") or "").strip()
                    or SCOPE_AI_AZURE_DEFAULT
                )
                print("azure-foundry (Microsoft Entra ID):")
                print(f"  Endpoint: {_base_url or '(not configured)'}")
                print(f"  Scope: {_scope}")
                if not has_azure_identity_installed():
                    print("  Status: ⚠ azure-identity not installed "
                          "(pip install azure-identity)")
                else:
                    _entra_cfg = EntraIdentityConfig(
                        scope=_scope,
                    )
                    _info = describe_active_credential(config=_entra_cfg, timeout_seconds=10.0)
                    _env_sources = _info.get("env_sources") or []
                    if _info.get("ok"):
                        _tag = ", ".join(_env_sources) if _env_sources else "default chain"
                        print(f"  Status: ✓ token acquired ({_tag})")
                    else:
                        _err = _info.get("error") or "credential chain exhausted"
                        print(f"  Status: ⚠ {_err}")
                        _hint = _info.get("hint")
                        if _hint:
                            print(f"  Hint: {_hint}")
                print()
    except Exception:
        pass
    print()

    # Main menu
    choices = [
        "Add a credential",
        "Remove a credential",
        "Reset cooldowns for a provider",
        "Set rotation strategy for a provider",
        "Exit",
    ]
    print("What would you like to do?")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")

    try:
        raw = input("\nChoice: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not raw or raw == str(len(choices)):
        return

    if raw == "1":
        _interactive_add()
    elif raw == "2":
        _interactive_remove()
    elif raw == "3":
        _interactive_reset()
    elif raw == "4":
        _interactive_strategy()


def _pick_provider(prompt: str = "Provider") -> str:
    """Prompt for a provider name with auto-complete hints."""
    known = sorted(set(list(PROVIDER_REGISTRY.keys()) + ["openrouter"]))
    custom_names = _get_custom_provider_names()
    if custom_names:
        custom_display = [name for name, _key, _provider_key in custom_names]
        print(f"\nKnown providers: {', '.join(known)}")
        print(f"Custom endpoints: {', '.join(custom_display)}")
    else:
        print(f"\nKnown providers: {', '.join(known)}")
    try:
        raw = input(f"{prompt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit()
    return _normalize_provider(raw)


def _interactive_add() -> None:
    provider = _pick_provider("Provider to add credential for")
    if provider not in PROVIDER_REGISTRY and provider != "openrouter" and not provider.startswith(CUSTOM_POOL_PREFIX):
        raise SystemExit(f"Unknown provider: {provider}")

    # For OAuth-capable providers, ask which type
    if provider in _OAUTH_CAPABLE_PROVIDERS:
        print(f"\n{provider} supports both API keys and OAuth login.")
        print("  1. API key (paste a key from the provider dashboard)")
        print("  2. OAuth login (authenticate via browser)")
        try:
            type_choice = input("Type [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if type_choice == "2":
            auth_type = "oauth"
        else:
            auth_type = "api_key"
    else:
        auth_type = "api_key"

    label = None
    try:
        typed_label = input("Label / account name (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if typed_label:
        label = typed_label

    auth_add_command(SimpleNamespace(
        provider=provider, auth_type=auth_type, label=label, api_key=None,
        portal_url=None, inference_url=None, client_id=None, scope=None,
        no_browser=False, timeout=None, insecure=False, ca_bundle=None,
    ))


def _interactive_remove() -> None:
    provider = _pick_provider("Provider to remove credential from")
    pool = load_pool(provider)
    if not pool.has_credentials():
        print(f"No credentials for {provider}.")
        return

    # Show entries with indices
    for i, e in enumerate(pool.entries(), 1):
        exhausted = _format_exhausted_status(e)
        print(f"  #{i}  {e.label:25s} {e.auth_type:10s} {e.source}{exhausted} [id:{e.id}]")

    try:
        raw = input("Remove #, id, or label (blank to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return

    auth_remove_command(SimpleNamespace(provider=provider, target=raw))


def _interactive_reset() -> None:
    provider = _pick_provider("Provider to reset cooldowns for")

    auth_reset_command(SimpleNamespace(provider=provider))


def _interactive_strategy() -> None:
    provider = _pick_provider("Provider to set strategy for")
    current = get_pool_strategy(provider)
    strategies = [STRATEGY_FILL_FIRST, STRATEGY_ROUND_ROBIN, STRATEGY_LEAST_USED, STRATEGY_RANDOM]

    print(f"\nCurrent strategy for {provider}: {current}")
    print()
    descriptions = {
        STRATEGY_FILL_FIRST: "Use first key until exhausted, then next",
        STRATEGY_ROUND_ROBIN: "Cycle through keys evenly",
        STRATEGY_LEAST_USED: "Always pick the least-used key",
        STRATEGY_RANDOM: "Random selection",
    }
    for i, s in enumerate(strategies, 1):
        marker = " ←" if s == current else ""
        print(f"  {i}. {s:15s} — {descriptions.get(s, '')}{marker}")

    try:
        raw = input("\nStrategy [1-4]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return

    try:
        idx = int(raw) - 1
        strategy = strategies[idx]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    pool_strategies = cfg.get("credential_pool_strategies") or {}
    if not isinstance(pool_strategies, dict):
        pool_strategies = {}
    pool_strategies[provider] = strategy
    cfg["credential_pool_strategies"] = pool_strategies
    save_config(cfg)
    print(f"Set {provider} strategy to: {strategy}")


def auth_command(args) -> None:
    action = getattr(args, "auth_action", "")
    if action == "add":
        auth_add_command(args)
        return
    if action == "list":
        auth_list_command(args)
        return
    if action == "remove":
        auth_remove_command(args)
        return
    if action == "reset":
        auth_reset_command(args)
        return
    if action == "usage":
        auth_usage_command(args)
        return
    if action == "status":
        auth_status_command(args)
        return
    if action == "logout":
        auth_logout_command(args)
        return
    if action == "spotify":
        auth_spotify_command(args)
        return
    # No subcommand — launch interactive mode
    _interactive_auth()
