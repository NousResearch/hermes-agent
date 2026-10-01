"""Credential-pool source ingestion."""

from __future__ import annotations
from typing import Optional  # noqa: F401 (pool collaborators consume these bindings)

from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from auth.credential_pool import PooledCredential
from auth.pool_environment import PoolEnvironment


class _Seeder:
    """Accumulates ``_upsert_entry`` results for one ``load_pool`` seeding pass."""

    def __init__(
        self,
        provider: str,
        entries: List[PooledCredential],
        *,
        environment: PoolEnvironment,
    ):
        from auth.credential_pool import _is_source_suppressed_fn

        self.environment = environment
        self.provider = provider
        self.entries = entries
        self.changed = False
        self.active_sources: Set[str] = set()
        self.is_suppressed = _is_source_suppressed_fn()

    def upsert(self, source: str, payload: Dict[str, Any]) -> bool:
        """Upsert unless suppressed (``hermes auth remove`` must stay stable across loads)."""
        from auth.credential_pool import _upsert_entry

        if self.is_suppressed(self.provider, source):
            return False
        self.active_sources.add(source)
        ingested = _upsert_entry(
            self.entries, self.provider, source, {"source": source, **payload}
        )
        self.changed |= ingested
        return ingested

    @property
    def result(self) -> Tuple[bool, Set[str]]:

        return self.changed, self.active_sources


def _seed_anthropic_singletons(seed: _Seeder, *, environment) -> None:
    # Only auto-discover external credentials (Claude Code, Hermes PKCE) when
    # the user explicitly configured anthropic; otherwise auxiliary fallback
    # chains would read ~/.claude/.credentials.json without consent (PR #4210).
    environment.require_current_scope()
    from auth.credential_pool import (
        AUTH_TYPE_OAUTH,
        _retain_sources_not_in,
        label_from_token,
    )

    try:
        if not seed.environment.provider_configured("anthropic"):
            return
    except ImportError:
        pass

    # API-key vs OAuth is a user-visible choice at `hermes setup`. The API-key
    # signal is ANTHROPIC_API_KEY set AND no OAuth env vars (the save_* helpers
    # zero the other side). Then we MUST NOT seed autodiscovered OAuth tokens:
    # rotation on a 401/429 would silently flip the session onto OAuth, which
    # forces the Claude Code identity injection, `mcp_` tool-name rewrite and
    # claude-cli User-Agent the user explicitly opted out of. Prefer
    # ~/.hermes/.env over os.environ, as `_seed_from_env` does.
    _env_file = seed.environment.read_env()

    def _env_val(key: str) -> str:
        from auth.credential_pool import _get_secret

        return (_env_file.get(key) or _get_secret(key, "") or "").strip()

    anthropic_oauth_env = _env_val("ANTHROPIC_TOKEN") or _env_val(
        "CLAUDE_CODE_OAUTH_TOKEN"
    )
    if _env_val("ANTHROPIC_API_KEY") and not anthropic_oauth_env:
        # Prune stale autodiscovered OAuth entries from a previous OAuth
        # session so a transient 401 cannot revive them.
        seed.changed |= _retain_sources_not_in(
            seed.entries, {"hermes_pkce", "claude_code"}
        )
        return

    from auth.providers.anthropic import (
        read_claude_code_credentials,
        read_hermes_oauth_credentials,
    )
    from auth.source_policy import adopt_external_logins_enabled

    sources = [("hermes_pkce", read_hermes_oauth_credentials())]
    if adopt_external_logins_enabled(environment=seed.environment):
        sources.append((
            "claude_code",
            read_claude_code_credentials(environment=environment),
        ))
    else:
        # Singleton-seeded rows are otherwise never pruned; the opt-out must also drop the row an
        # earlier (adopting) process persisted, or it keeps rotating a login Hermes no longer reads.
        seed.changed |= _retain_sources_not_in(seed.entries, {"claude_code"})
    for source_name, creds in sources:
        if creds and creds.get("accessToken"):
            seed.upsert(
                source_name,
                {
                    "auth_type": AUTH_TYPE_OAUTH,
                    "access_token": creds.get("accessToken", ""),
                    "refresh_token": creds.get("refreshToken"),
                    "expires_at_ms": creds.get("expiresAt"),
                    "label": label_from_token(
                        creds.get("accessToken", ""), source_name
                    ),
                },
            )


def _seed_nous_singleton(seed: _Seeder, auth_store: Dict[str, Any]) -> None:
    from auth.credential_pool import AUTH_TYPE_OAUTH, _NOUS_EXTRA_STATE_KEYS, _global_auth_file_path, _load_provider_state_with_source, _retain_sources_not_in, _same_path, _store_owns_pool_provider, label_from_token
    state, source_path = _load_provider_state_with_source(auth_store, "nous")
    global_root = _global_auth_file_path()
    if (
        source_path is not None and global_root is not None and _same_path(source_path, global_root)
        and _store_owns_pool_provider(auth_store, "nous")
    ):
        # A profile that owns local nous rows (e.g. an agent_key-only row surviving a
        # fork strip/heal) must not re-seed root's single-use refresh token into its
        # own pool from the global-root fallback: that re-creates the fork.
        return
    has_runtime_material = bool(
        isinstance(state, dict)
        and (str(state.get("access_token") or "").strip() or str(state.get("agent_key") or "").strip())
    )
    if state and not has_runtime_material:
        seed.changed |= _retain_sources_not_in(seed.entries, {"device_code", "manual:device_code"})
    if not (state and has_runtime_material):
        return
    # Prefer a user-supplied label embedded in the singleton state (``hermes
    # auth add nous --label <name>``) over the token-derived fingerprint.
    custom_label = str(state.get("label") or "").strip()
    seed.upsert("device_code", {
        "auth_type": AUTH_TYPE_OAUTH,
        "access_token": state.get("access_token", ""),
        "refresh_token": state.get("refresh_token"),
        "expires_at": state.get("expires_at"),
        "token_type": state.get("token_type"),
        "scope": state.get("scope"),
        "client_id": state.get("client_id"),
        "portal_base_url": state.get("portal_base_url"),
        "inference_base_url": state.get("inference_base_url"),
        "agent_key": state.get("agent_key"),
        "agent_key_expires_at": state.get("agent_key_expires_at"),
        # Refresh timestamps let freshness-sensitive consumers (self-heal
        # hooks, pruning by age) tell just-refreshed credentials from stale
        # ones (#15099).
        **{key: state.get(key) for key in _NOUS_EXTRA_STATE_KEYS},
        "tls": state.get("tls") if isinstance(state.get("tls"), dict) else None,
        "label": custom_label or label_from_token(state.get("access_token", ""), "device_code"),
    })


# Warn once per token per process when Copilot exchange degrades to raw token (#114740).
_COPILOT_RAW_DEGRADATION_WARNED: Set[str] = set()


def _warn_copilot_raw_degradation_once(token: str) -> None:
    """WARN once per token per process when Copilot exchange degrades to raw token (#114740)."""
    from auth.credential_pool import fingerprint_secret_value, logger
    fingerprint = fingerprint_secret_value(token) or "unknown"
    if fingerprint in _COPILOT_RAW_DEGRADATION_WARNED:
        return
    _COPILOT_RAW_DEGRADATION_WARNED.add(fingerprint)
    logger.warning(
        "Copilot token exchange degraded to RAW token (exchange "
        "unavailable); enterprise-only models may 400 with "
        "model_not_available_for_integrator until exchange recovers."
    )


def _reset_copilot_raw_degradation_warned() -> None:
    """Clear the degradation warning cache (for test isolation)."""
    _COPILOT_RAW_DEGRADATION_WARNED.clear()


def _seed_copilot_singleton(seed: _Seeder) -> None:
    # Copilot tokens are resolved dynamically via `gh auth token` or env vars
    # (COPILOT_GITHUB_TOKEN / GH_TOKEN); they don't live in the auth store.
    from auth.credential_pool import AUTH_TYPE_API_KEY, logger
    try:
        hooks = seed.environment.provider_hooks(seed.provider)
        COPILOT_ENV_VARS = hooks.external_env_vars
        resolve_copilot_token = hooks.resolve_external_token
        get_copilot_api_token = hooks.exchange_external_token
        # All-sources gate BEFORE any work: resolve_copilot_token() shells out
        # and the exchange retries 3x with backoff (~35s worst case); a user
        # who suppressed every copilot source must not pay that on every pool
        # load. The source space here matches credential_sources._remove_copilot_gh.
        copilot_sources = ["gh_cli"] + [f"env:{v}" for v in COPILOT_ENV_VARS]
        if all(seed.is_suppressed(seed.provider, s) for s in copilot_sources):
            return
        token, source = resolve_copilot_token()
        if not token:
            return
        # Exact match: a substring test would classify GH_TOKEN/GITHUB_TOKEN
        # as gh_cli and bypass a user's per-env-var suppression.
        source_name = "gh_cli" if source == "gh auth token" else f"env:{source}"
        # Per-source gate BEFORE the (~35s worst case) network exchange.
        if seed.is_suppressed(seed.provider, source_name):
            return
        if not seed.environment.provider_configured(seed.provider):
            # Copilot is only discovered here (ambient gh CLI login), not selected anywhere: no
            # model will be routed to it, so the network exchange — and its degradation warning on
            # every pool load — buys nothing (#114740). Seed the raw token; the load that follows
            # the user selecting copilot re-seeds and exchanges.
            api_token, enterprise_base_url = token, None
        else:
            api_token, enterprise_base_url = get_copilot_api_token(token)
            # get_copilot_api_token falls back to the RAW token when the exchange
            # fails; the Copilot API then routes it to the fallback
            # "copilot-language-server" integrator whose allowlist omits
            # enterprise-only models -> HTTP 400 on every turn. Surface it once.
            if api_token == token and not enterprise_base_url:
                _warn_copilot_raw_degradation_once(token)
        pconfig = seed.environment.provider_config(seed.provider)
        seed.upsert(source_name, {
            "auth_type": AUTH_TYPE_API_KEY,
            "access_token": api_token,
            "base_url": enterprise_base_url or (pconfig.inference_base_url if pconfig else ""),
            "label": source,
        })
    except Exception as exc:
        logger.debug("Copilot token seed failed: %s", exc)


def _seed_qwen_singleton(seed: _Seeder) -> None:
    # Qwen OAuth tokens live in ~/.qwen/oauth_creds.json (written by the Qwen
    # CLI). refresh_if_expiring=False avoids network calls during pool loading.
    from auth.credential_pool import AUTH_TYPE_OAUTH, logger
    try:
        creds = seed.environment.provider_hooks(seed.provider).resolve_credentials(refresh_if_expiring=False)
        token = creds.get("api_key", "")
        if token:
            source_name = creds.get("source", "qwen-cli")
            seed.upsert(source_name, {
                "auth_type": AUTH_TYPE_OAUTH,
                "access_token": token,
                "expires_at_ms": creds.get("expires_at_ms"),
                "base_url": creds.get("base_url", ""),
                "label": creds.get("auth_file", source_name),
            })
    except Exception as exc:
        logger.debug("Qwen OAuth token seed failed: %s", exc)


def _seed_minimax_singleton(seed: _Seeder) -> None:
    # Read the raw auth.json state rather than resolve_minimax_oauth_runtime_credentials,
    # which always refreshes on expiry (surprise network calls during discovery).
    from auth.credential_pool import AUTH_TYPE_OAUTH, datetime, label_from_token, logger
    try:
        from auth.provider_state import get_provider_auth_state
        state = get_provider_auth_state("minimax-oauth")
        if not (state and state.get("access_token")):
            return
        expires_at_ms = None
        try:
            raw = state.get("expires_at", "")
            if raw:
                expires_at_ms = int(datetime.fromisoformat(raw).timestamp() * 1000)
        except Exception:
            expires_at_ms = None
        seed.upsert("oauth", {
            "auth_type": AUTH_TYPE_OAUTH,
            "access_token": state["access_token"],
            "refresh_token": state.get("refresh_token"),
            "expires_at_ms": expires_at_ms,
            "base_url": str(state.get("inference_base_url", "") or "").rstrip("/"),
            "label": state.get("label", "") or label_from_token(state.get("access_token", ""), "oauth"),
        })
    except Exception as exc:
        logger.debug("MiniMax OAuth token seed failed: %s", exc)


def _seed_tokens_singleton(seed: _Seeder, auth_store: Dict[str, Any]) -> None:
    """Codex / xAI: surface the auth.json ``providers.<id>.tokens`` singleton as ``device_code``.

    Hermes owns its own Codex auth state and does NOT auto-import
    ~/.codex/auth.json: refresh tokens are single-use, so sharing them with
    Codex CLI / VS Code causes refresh_token_reused races. Adoption is an
    explicit one-time prompt via `hermes auth openai-codex`.
    """
    from auth.credential_pool import AUTH_TYPE_OAUTH, _load_provider_state, label_from_token
    state = _load_provider_state(auth_store, seed.provider)
    tokens = state.get("tokens") if isinstance(state, dict) else None
    if not (isinstance(tokens, dict) and tokens.get("access_token")):
        return
    if seed.provider == "openai-codex":
        base_url = "https://chatgpt.com/backend-api/codex"
        custom_label = str(state.get("label") or "").strip()
    else:
        base_url = "https://api.x.ai/v1"
        custom_label = ""
    seed.upsert("device_code", {
        "auth_type": AUTH_TYPE_OAUTH,
        "access_token": tokens.get("access_token", ""),
        "refresh_token": tokens.get("refresh_token"),
        "base_url": base_url,
        "last_refresh": state.get("last_refresh"),
        "label": custom_label or label_from_token(tokens.get("access_token", ""), "device_code"),
    })


def _seed_from_singletons(
    provider: str, entries: List[PooledCredential], *, environment: PoolEnvironment
) -> Tuple[bool, Set[str]]:
    environment.require_current_scope()
    from auth.credential_pool import _TOKENS_SINGLETON_PROVIDERS, _load_auth_store

    seed = _Seeder(provider, entries, environment=environment)
    auth_store = _load_auth_store()
    if provider == "anthropic":
        _seed_anthropic_singletons(seed, environment=environment)
    elif provider == "nous":
        _seed_nous_singleton(seed, auth_store)
    elif provider == "copilot":
        _seed_copilot_singleton(seed)
    elif provider == "qwen-oauth":
        _seed_qwen_singleton(seed)
    elif provider == "minimax-oauth":
        _seed_minimax_singleton(seed)
    elif provider in _TOKENS_SINGLETON_PROVIDERS:
        # `hermes auth remove openai-codex` suppresses device_code; without
        # this gate the removal is undone on the next load_pool().
        if provider == "openai-codex" and seed.is_suppressed(provider, "device_code"):
            return seed.result
        _seed_tokens_singleton(seed, auth_store)
    return seed.result


def get_env_prefer_dotenv(key: str, *, environment: PoolEnvironment) -> str:
    """Resolve a credential env var, preferring ~/.hermes/.env over os.environ.

    The user's config file is authoritative; stale env vars from parent
    processes (Codex CLI, test scripts) must not override deliberate .env
    changes. load_env() memoizes on mtime, so per-call reads cost a stat().
    An unresolved ``op://`` reference in .env yields to the already-resolved
    value from the active secret scope (set by apply_onepassword_secrets());
    otherwise every provider auth attempt would receive a URL instead of a key.
    """
    from auth.credential_pool import _get_secret
    env_file = environment.read_env()
    raw = env_file.get(key, "").strip()
    scoped_value = (_get_secret(key, "") or "").strip()
    if raw.startswith("op://") and scoped_value:
        return scoped_value
    return raw or scoped_value


# Providers already warned about env-key -> pool ingestion, once per process
# (#81952 expected-behavior #3).
_ENV_INGESTION_WARNED: Set[str] = set()


def _warn_env_ingestion_once(provider: str, env_var: str) -> None:
    """WARN once per process per provider when an env credential is ingested into a paid pool.

    Auto-ingesting OPENROUTER_API_KEY is what ARMS silent OpenRouter spend —
    every downstream auto-detect keys off the pool having credentials.
    Ingestion stays allowed (an exported key is arguable intent) but must
    never be silent.
    """
    from auth.credential_pool import logger
    if provider in _ENV_INGESTION_WARNED:
        return
    _ENV_INGESTION_WARNED.add(provider)
    logger.warning(
        "Ingested %s from environment into the %s credential pool — this "
        "enables %s spend. Remove the key or run "
        "hermes auth remove %s <n> to suppress.",
        env_var,
        provider,
        "OpenRouter" if provider == "openrouter" else provider,
        provider,
    )


def _env_payload(*, env_var: str, token: str, base_url: str, environment: PoolEnvironment) -> Dict[str, Any]:
    from auth.credential_pool import AUTH_TYPE_API_KEY
    payload: Dict[str, Any] = {
        "auth_type": AUTH_TYPE_API_KEY,
        "access_token": token,
        "base_url": base_url,
        "label": env_var,
    }
    try:
        source_label = environment.secret_source(env_var)
    except Exception:
        source_label = None
    secret_source = str(source_label).strip() if source_label else None
    if secret_source:
        payload["secret_source"] = secret_source
    return payload


# Region-specific endpoints inferred from the key itself.


def _env_key_var_candidates(
    env_vars: List[str],
    entries: List[PooledCredential],
    *,
    environment: PoolEnvironment,
) -> List[str]:
    """*env_vars*, their numbered siblings, and the ``env:VAR`` names already persisted.

    ``VAR_2``, ``VAR_3``, ... are tried for every declared VAR until the first
    one that does not resolve, so a `.env` or secret-manager project can back a
    whole rotation pool with no config: setting ``NVIDIA_API_KEY_2`` is the
    whole opt-in (#76593).

    Env-backed rows are written to auth.json without their secret and
    re-hydrated on every load; a row whose VAR the registry does not
    declare would otherwise stay empty forever and be silently dropped
    from rotation by ``_available_entries``.
    """
    names = list(env_vars)
    for base in env_vars:
        n = 2
        while get_env_prefer_dotenv(f"{base}_{n}", environment=environment):
            names.append(f"{base}_{n}")
            n += 1
    for entry in entries:
        if entry.source.startswith("env:"):
            env_name = entry.source.split(":", 1)[1].strip()
            if env_name and env_name not in names:
                names.append(env_name)
    return names


def _seed_from_env(
    provider: str, entries: List[PooledCredential], *, environment: PoolEnvironment
) -> Tuple[bool, Set[str]]:
    from auth.credential_pool import AUTH_TYPE_API_KEY, OPENROUTER_BASE_URL

    seed = _Seeder(provider, entries, environment=environment)
    # Copilot's singleton branch exchanges the raw ghu_ OAuth token for the
    # api token via `get_copilot_api_token`; the generic loop would re-read
    # COPILOT_GITHUB_TOKEN and overwrite it with the RAW token, causing 400s
    # ("not available for integrator copilot-language-server").
    if provider == "copilot":
        return seed.result

    if provider == "openrouter":
        for env_var in _env_key_var_candidates(
            ["OPENROUTER_API_KEY"], entries, environment=environment
        ):
            token = get_env_prefer_dotenv(env_var, environment=environment)
            if token and seed.upsert(
                f"env:{env_var}",
                _env_payload(
                    env_var=env_var,
                    token=token,
                    base_url=OPENROUTER_BASE_URL,
                    environment=environment,
                ),
            ):
                _warn_env_ingestion_once(provider, env_var)
        return seed.result

    pconfig = environment.provider_config(provider)
    if not pconfig or pconfig.auth_type != AUTH_TYPE_API_KEY:
        return seed.result

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = get_env_prefer_dotenv(
            pconfig.base_url_env_var, environment=environment
        ).rstrip("/")

    env_vars = list(pconfig.api_key_env_vars)
    if provider == "anthropic":
        env_vars = ["ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]
    env_vars = _env_key_var_candidates(env_vars, entries, environment=environment)

    resolve_base_url = environment.key_endpoint
    for env_var in env_vars:
        token = get_env_prefer_dotenv(env_var, environment=environment)
        if not token:
            continue
        base_url = env_url or pconfig.inference_base_url
        if resolve_base_url is not None:
            base_url = resolve_base_url(
                provider, token, pconfig.inference_base_url, env_url
            )
        seed.upsert(
            f"env:{env_var}",
            _env_payload(
                env_var=env_var, token=token, base_url=base_url, environment=environment
            ),
        )
    return seed.result


def _prune_stale_seeded_entries(
    entries: List[PooledCredential],
    active_sources: Set[str],
    *,
    prune_env_sources: bool = True,
) -> bool:
    from auth.credential_pool import _is_manual_source

    def _is_prunable(entry: PooledCredential) -> bool:
        # ``env:*`` entries are persisted references re-hydrated on every load.
        # A process that merely lacks the env var must NOT delete the on-disk
        # entry for every other process (#9331); prune only when explicitly
        # requested (an `hermes auth` command that confirmed the source is gone).
        from auth.credential_pool import is_borrowed_credential_source

        if entry.source.startswith("env:"):
            return prune_env_sources
        # File-backed singletons and Hermes PKCE disappear when their backing file is gone.
        return (
            is_borrowed_credential_source(entry.source, entry.provider)
            or entry.source == "hermes_pkce"
        )

    retained = [
        entry
        for entry in entries
        if _is_manual_source(entry.source)
        or entry.source in active_sources
        or not _is_prunable(entry)
    ]
    if len(retained) == len(entries):
        return False
    entries[:] = retained
    return True


def _seed_custom_pool(
    pool_key: str, entries: List[PooledCredential], *, environment: PoolEnvironment
) -> Tuple[bool, Set[str]]:
    """Seed a custom endpoint pool from custom_providers config and model config."""
    environment.require_current_scope()
    from auth.credential_pool import (
        AUTH_TYPE_API_KEY,
        _get_custom_provider_config,
        _load_config_safe,
        _norm_url,
        custom_provider_pool_key_candidates,
    )

    seed = _Seeder(pool_key, entries, environment=environment)

    cp_config = _get_custom_provider_config(pool_key, environment=environment)
    if cp_config:
        api_key = str(cp_config.get("api_key") or "").strip()
        name = str(cp_config.get("name") or "").strip()
        if api_key:
            seed.upsert(
                f"config:{name}",
                {
                    "auth_type": AUTH_TYPE_API_KEY,
                    "access_token": api_key,
                    "base_url": _norm_url(cp_config.get("base_url")),
                    "label": name or f"config:{name}",
                },
            )

    # Seed from model.api_key when model.provider=='custom' and model.base_url matches
    try:
        config = _load_config_safe(environment=environment)
        model_cfg = config.get("model") if config else None
        if isinstance(model_cfg, dict):
            model_provider = str(model_cfg.get("provider") or "").strip().lower()
            model_base_url = _norm_url(model_cfg.get("base_url"))
            model_api_key = next(
                (
                    v.strip()
                    for k in ("api_key", "api")
                    for v in (model_cfg.get(k),)
                    if isinstance(v, str) and v.strip()
                ),
                "",
            )
            if model_provider == "custom" and model_base_url and model_api_key:
                # The pool may be keyed under the durable ``providers.<key>``
                # slug or legacy ``custom:<name>``; accept any candidate, or
                # seeding is skipped when the pool holds the other identity.
                # Check if this model's base_url matches our custom provider. See #100413.
                matched_keys = {
                    str(key).strip().lower()
                    for key in custom_provider_pool_key_candidates(
                        model_base_url, environment=environment
                    )
                }
                if pool_key in matched_keys:
                    seed.upsert(
                        "model_config",
                        {
                            "auth_type": AUTH_TYPE_API_KEY,
                            "access_token": model_api_key,
                            "base_url": model_base_url,
                            "label": "model_config",
                        },
                    )
    except Exception:
        pass

    return seed.result
