"""
hermes fallback — manage the fallback provider chain.

Fallback providers are tried in order when the primary model fails with
rate-limit, overload, or connection errors. See:
https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers

Subcommands:
  hermes fallback [list]   Show the current fallback chain (default when no subcommand)
  hermes fallback add      Pick provider + model via the same picker as `hermes model`,
                           then append the selection to the chain
  hermes fallback remove   Pick an entry to delete from the chain
  hermes fallback clear    Remove all fallback entries

Storage: ``fallback_providers`` in ``~/.hermes/config.yaml`` (top-level, list of
``{provider, model, base_url?, api_mode?}`` dicts).  The legacy single-dict
``fallback_model`` format is migrated to the new list format on first add.
"""
from __future__ import annotations

import copy
import json
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hermes_cli.fallback_config import get_fallback_chain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_chain(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the normalized fallback chain as a list of dicts.

    Accepts both the new list format (``fallback_providers``) and the legacy
    ``fallback_model`` format. When both are present, the effective chain is
    merged with ``fallback_providers`` entries kept first. The returned list is
    always a fresh copy — callers can mutate without touching the config dict.
    """
    return get_fallback_chain(config)


def _write_chain(config: Dict[str, Any], chain: List[Dict[str, Any]]) -> None:
    """Persist the chain to ``fallback_providers`` and clear legacy key."""
    config["fallback_providers"] = chain
    # Drop the legacy single-dict key on write so there's only one source of truth.
    if "fallback_model" in config:
        config.pop("fallback_model", None)


def _format_entry(entry: Dict[str, Any]) -> str:
    """One-line human-readable rendering of a fallback entry."""
    provider = entry.get("provider", "?")
    model = entry.get("model", "?")
    base = entry.get("base_url")
    suffix = f"  [{base}]" if base else ""
    return f"{model}  (via {provider}){suffix}"


def _extract_fallback_from_model_cfg(model_cfg: Any) -> Optional[Dict[str, Any]]:
    """Pull the ``{provider, model, base_url?, api_mode?}`` dict from a ``config["model"]`` snapshot."""
    if not isinstance(model_cfg, dict):
        return None
    provider = (model_cfg.get("provider") or "").strip()
    # The picker writes the selected model to ``model.default``.
    model = (model_cfg.get("default") or model_cfg.get("model") or "").strip()
    if not provider or not model:
        return None
    entry: Dict[str, Any] = {"provider": provider, "model": model}
    base_url = (model_cfg.get("base_url") or "").strip()
    if base_url:
        entry["base_url"] = base_url
    api_mode = (model_cfg.get("api_mode") or "").strip()
    if api_mode:
        entry["api_mode"] = api_mode
    return entry


def _snapshot_auth_active_provider() -> Any:
    """Return the current ``active_provider`` in auth.json, or a sentinel if unavailable."""
    try:
        from hermes_cli.auth import _load_auth_store
        store = _load_auth_store()
        return store.get("active_provider")
    except Exception:
        return None


def _restore_auth_active_provider(value: Any) -> None:
    """Write back a previously snapshotted ``active_provider`` value."""
    try:
        from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
        with _auth_store_lock():
            store = _load_auth_store()
            store["active_provider"] = value
            _save_auth_store(store)
    except Exception:
        # Best-effort — if auth.json can't be restored, the user's primary
        # provider may have been deactivated by the picker.  They can re-run
        # `hermes model` to fix it.  Don't fail the fallback add.
        pass


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_fallback_list(args) -> None:  # noqa: ARG001
    """Print the current fallback chain."""
    from hermes_cli.config import load_config

    config = load_config()
    chain = _read_chain(config)

    print()
    if not chain:
        print("  No fallback providers configured.")
        print()
        print("  Add one with:  hermes fallback add")
        print()
        return

    primary = _describe_primary(config)
    if primary:
        print(f"  Primary:   {primary}")
        print()
    print(f"  Fallback chain ({len(chain)} {'entry' if len(chain) == 1 else 'entries'}):")
    for i, entry in enumerate(chain, 1):
        print(f"    {i}. {_format_entry(entry)}")
    print()
    print("  Tried in order when the primary fails (rate-limit, 5xx, connection errors).")
    print("  Docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers")
    print()


def _describe_primary(config: Dict[str, Any]) -> Optional[str]:
    """One-line description of the primary model for display purposes."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        provider = (model_cfg.get("provider") or "?").strip() or "?"
        model = (model_cfg.get("default") or model_cfg.get("model") or "?").strip() or "?"
        return f"{model}  (via {provider})"
    if isinstance(model_cfg, str) and model_cfg.strip():
        return model_cfg.strip()
    return None


def cmd_fallback_add(args) -> None:
    """Launch the same picker as `hermes model`, then append the selection to the chain."""
    from hermes_cli.main import _require_tty, select_provider_and_model
    from hermes_cli.config import load_config, save_config

    _require_tty("fallback add")

    # Snapshot BEFORE the picker runs so we can distinguish "user actually
    # picked something" from "user cancelled" by comparing before/after.
    before_cfg = load_config()
    model_before = copy.deepcopy(before_cfg.get("model"))
    active_provider_before = _snapshot_auth_active_provider()

    print()
    print("  Adding a fallback provider.  The picker below is the same one used by")
    print("  `hermes model` — select the provider + model you want as a fallback.")
    print()

    try:
        select_provider_and_model(args=args)
    except SystemExit:
        # Some provider flows exit on auth failure — restore state and re-raise.
        _restore_model_cfg(model_before)
        _restore_auth_active_provider(active_provider_before)
        raise

    # Read the post-picker state to see what the user selected.
    after_cfg = load_config()
    model_after = after_cfg.get("model")

    new_entry = _extract_fallback_from_model_cfg(model_after)
    if not new_entry:
        # Picker didn't complete (user cancelled or flow bailed).  Nothing to do.
        _restore_model_cfg(model_before)
        _restore_auth_active_provider(active_provider_before)
        print()
        print("  No fallback added.")
        return

    # Picker picked the same thing that's already the primary → nothing changed,
    # and there's nothing useful to add as a fallback to itself.
    primary_entry = _extract_fallback_from_model_cfg(model_before)
    if primary_entry and primary_entry["provider"] == new_entry["provider"] \
            and primary_entry["model"] == new_entry["model"]:
        _restore_model_cfg(model_before)
        _restore_auth_active_provider(active_provider_before)
        print()
        print(f"  Selected model matches the current primary ({_format_entry(new_entry)}).")
        print("  A provider cannot be a fallback for itself — no change.")
        return

    # Reload the config with the primary restored, then append the new entry
    # to ``fallback_providers``.  We deliberately re-load (rather than mutating
    # ``after_cfg``) because the picker may have touched other top-level keys
    # (custom_providers, providers credentials) that we want to keep.
    _restore_model_cfg(model_before)
    _restore_auth_active_provider(active_provider_before)

    final_cfg = load_config()
    chain = _read_chain(final_cfg)

    # Reject exact-duplicate fallback entries.
    for existing in chain:
        if existing.get("provider") == new_entry["provider"] \
                and existing.get("model") == new_entry["model"]:
            print()
            print(f"  {_format_entry(new_entry)} is already in the fallback chain — skipped.")
            return

    chain.append(new_entry)
    _write_chain(final_cfg, chain)
    save_config(final_cfg)

    print()
    print(f"  Added fallback: {_format_entry(new_entry)}")
    print(f"  Chain is now {len(chain)} {'entry' if len(chain) == 1 else 'entries'} long.")
    print()
    print("  Run `hermes fallback list` to view, or `hermes fallback remove` to delete.")


def _restore_model_cfg(model_before: Any) -> None:
    """Restore ``config["model"]`` to a previously-captured snapshot."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    if model_before is None:
        cfg.pop("model", None)
    else:
        cfg["model"] = copy.deepcopy(model_before)
    save_config(cfg)


def cmd_fallback_remove(args) -> None:  # noqa: ARG001
    """Pick an entry from the chain and remove it."""
    from hermes_cli.config import load_config, save_config

    config = load_config()
    chain = _read_chain(config)

    if not chain:
        print()
        print("  No fallback providers configured — nothing to remove.")
        print()
        return

    choices = [_format_entry(e) for e in chain]
    choices.append("Cancel")

    try:
        from hermes_cli.setup import _curses_prompt_choice
        idx = _curses_prompt_choice("Select a fallback to remove:", choices, 0)
    except Exception:
        idx = _numbered_pick("Select a fallback to remove:", choices)

    if idx is None or idx < 0 or idx >= len(chain):
        print()
        print("  Cancelled — no change.")
        return

    removed = chain.pop(idx)
    _write_chain(config, chain)
    save_config(config)

    print()
    print(f"  Removed fallback: {_format_entry(removed)}")
    if chain:
        print(f"  Chain is now {len(chain)} {'entry' if len(chain) == 1 else 'entries'} long.")
    else:
        print("  Fallback chain is now empty.")
    print()


def cmd_fallback_clear(args) -> None:  # noqa: ARG001
    """Remove all fallback entries (with confirmation)."""
    from hermes_cli.config import load_config, save_config

    config = load_config()
    chain = _read_chain(config)

    if not chain:
        print()
        print("  No fallback providers configured — nothing to clear.")
        print()
        return

    print()
    print(f"  Current fallback chain ({len(chain)} {'entry' if len(chain) == 1 else 'entries'}):")
    for i, entry in enumerate(chain, 1):
        print(f"    {i}. {_format_entry(entry)}")
    print()
    try:
        resp = input("  Clear all entries? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        print("  Cancelled.")
        return
    if resp not in {"y", "yes"}:
        print("  Cancelled — no change.")
        return

    _write_chain(config, [])
    save_config(config)
    print()
    print("  Fallback chain cleared.")
    print()


def _numbered_pick(question: str, choices: List[str]) -> Optional[int]:
    """Fallback numbered-list picker when curses is unavailable."""
    print(question)
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    print()
    while True:
        try:
            val = input(f"Choice [1-{len(choices)}]: ").strip()
            if not val:
                return None
            idx = int(val) - 1
            if 0 <= idx < len(choices):
                return idx
            print(f"Please enter 1-{len(choices)}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


# ---------------------------------------------------------------------------
# Fallback chain readiness check
# ---------------------------------------------------------------------------

# Mapping from provider name (as used in fallback entries) to the API key
# environment variable(s) it needs.  Covers API-key providers not in
# PROVIDER_REGISTRY and the most common auth_type=api_key registrations.
# For OAuth providers, the presence of a credential pool entry is sufficient.
_PROVIDER_KEY_ENV_VARS: Dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    "openai-api": ("OPENAI_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openai-codex": (),
    "xai-oauth": (),
    "xai": ("XAI_API_KEY",),
    "qwen-oauth": (),
    "minimax-oauth": (),
    "minimax": ("MINIMAX_API_KEY",),
    "minimax-cn": ("MINIMAX_CN_API_KEY",),
    "nous": (),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"),
    "gemini": ("GEMINI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "kimi-coding": ("KIMI_API_KEY",),
    "kimi-coding-cn": ("KIMI_CN_API_KEY",),
    "zai": ("ZAI_API_KEY",),
    "gmi": ("GMI_API_KEY",),
    "fireworks": ("FIREWORKS_API_KEY",),
    "copilot": ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
    "copilot-acp": (),
    "stepfun": ("STEP_API_KEY",),
    "arcee": ("ARCEE_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "opencode-zen": ("OPENCODE_ZEN_API_KEY",),
    "opencode-go": ("OPENCODE_GO_API_KEY",),
    "xiaomi": ("XIAOMI_API_KEY",),
    "tencent-tokenhub": ("TOKENHUB_API_KEY",),
    "kilocode": ("KILOCODE_API_KEY",),
    "bedrock": (),
    "vertex": (),
    "azure-foundry": (),
    "ollama-cloud": ("OLLAMA_CLOUD_API_KEY",),
    "custom": (),
    "lmstudio": ("LM_API_KEY",),
}


def _resolve_provider_name(raw: str) -> str:
    """Normalize a provider name using the same alias table as resolve_provider()."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    normalized = raw.strip().lower()
    # Check aliases (simplified version of auth.resolve_provider's alias map)
    _PROVIDER_ALIASES = {
        "glm": "zai",
        "z-ai": "zai",
        "z.ai": "zai",
        "zhipu": "zai",
        "google": "gemini",
        "google-gemini": "gemini",
        "google-ai-studio": "gemini",
        "x-ai": "xai",
        "x.ai": "xai",
        "grok": "xai",
        "xai-oauth": "xai-oauth",
        "grok-oauth": "xai-oauth",
        "kimi": "kimi-coding",
        "moonshot": "kimi-coding",
        "kimi-cn": "kimi-coding-cn",
        "moonshot-cn": "kimi-coding-cn",
        "step": "stepfun",
        "arcee-ai": "arcee",
        "arceeai": "arcee",
        "gmi-cloud": "gmi",
        "gmicloud": "gmi",
        "minimax-china": "minimax-cn",
        "minimax_cn": "minimax-cn",
        "minimax-portal": "minimax-oauth",
        "minimax-global": "minimax-oauth",
        "minimax_oauth": "minimax-oauth",
        "claude": "anthropic",
        "claude-code": "anthropic",
        "github": "copilot",
        "github-copilot": "copilot",
        "github-models": "copilot",
        "github-copilot-acp": "copilot-acp",
        "copilot-acp-agent": "copilot-acp",
        "opencode": "opencode-zen",
        "zen": "opencode-zen",
        "qwen-portal": "qwen-oauth",
        "qwen-cli": "qwen-oauth",
        "hf": "huggingface",
        "hugging-face": "huggingface",
        "huggingface-hub": "huggingface",
        "mimo": "xiaomi",
        "xiaomi-mimo": "xiaomi",
        "tencent": "tencent-tokenhub",
        "tokenhub": "tencent-tokenhub",
        "tencent-cloud": "tencent-tokenhub",
        "tencentmaas": "tencent-tokenhub",
        "aws": "bedrock",
        "aws-bedrock": "bedrock",
        "amazon-bedrock": "bedrock",
        "amazon": "bedrock",
        "go": "opencode-go",
        "opencode-go-sub": "opencode-go",
        "kilo": "kilocode",
        "kilo-code": "kilocode",
        "kilo-gateway": "kilocode",
        "lmstudio": "lmstudio",
        "lm-studio": "lmstudio",
        "lm_studio": "lmstudio",
        "ollama": "custom",
        "ollama_cloud": "ollama-cloud",
        "vllm": "custom",
        "llamacpp": "custom",
        "llama.cpp": "custom",
        "llama-cpp": "custom",
    }
    resolved = _PROVIDER_ALIASES.get(normalized, normalized)
    # Check if it's a known provider
    if resolved in PROVIDER_REGISTRY:
        return resolved
    return normalized


def _get_credential_env_for_provider(provider: str) -> tuple[str, ...]:
    """Return the env var names to check for a given provider."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.api_key_env_vars:
        return pconfig.api_key_env_vars
    return _PROVIDER_KEY_ENV_VARS.get(provider, ())


def _has_pool_credential_for_provider(provider: str) -> bool:
    """Check whether the credential pool has at least one entry for *provider*."""
    try:
        from hermes_cli.auth import read_credential_pool

        pool = read_credential_pool(provider)
        return bool(pool)
    except Exception:
        return False


def _check_entry_config(
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Static check: is the fallback entry structurally valid and configured?"""
    provider_raw = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip()
    base_url = str(entry.get("base_url") or "").strip()

    result: dict[str, Any] = {
        "provider": provider_raw or "?",
        "model": model or "?",
        "base_url": base_url or "",
        "configured": False,
        "auth_ready": False,
        "config_errors": [],
        "auth_errors": [],
    }

    if not provider_raw:
        result["config_errors"].append("provider name is empty")
        return result
    if not model:
        result["config_errors"].append("model name is empty")
        return result

    # Resolve provider name
    resolved = _resolve_provider_name(provider_raw)
    from hermes_cli.auth import PROVIDER_REGISTRY

    if resolved not in PROVIDER_REGISTRY and resolved not in (
        "custom",
        "openai-codex",
        "xai-oauth",
        "qwen-oauth",
        "minimax-oauth",
        "copilot-acp",
        "bedrock",
        "vertex",
        "azure-foundry",
        "opencode-zen",
        "opencode-go",
        "ollama-cloud",
    ):
        # Could be a named custom provider — check custom_providers in config.
        try:
            from hermes_cli.runtime_provider import has_named_custom_provider

            if not has_named_custom_provider(resolved):
                # Check if the original name (before alias resolution) matches
                from hermes_cli.runtime_provider import has_named_custom_provider as _hncp

                if not _hncp(provider_raw):
                    result["config_errors"].append(
                        f"unknown provider '{provider_raw}' (resolved to '{resolved}')"
                    )
                    return result
        except Exception:
            result["config_errors"].append(f"unknown provider '{provider_raw}'")
            return result

    result["configured"] = True
    return result


def _check_entry_auth(entry: dict[str, Any]) -> dict[str, Any]:
    """Static auth check: are credentials present for this fallback entry?"""
    provider_raw = str(entry.get("provider") or "").strip()
    resolved = _resolve_provider_name(provider_raw)
    env_vars = _get_credential_env_for_provider(resolved)

    result: dict[str, Any] = {
        "auth_ready": False,
        "auth_errors": [],
        "auth_source": "",
    }

    # Check env vars
    found_env = None
    for var in env_vars:
        import os

        val = os.environ.get(var, "")
        from hermes_cli.auth import has_usable_secret

        if has_usable_secret(val):
            found_env = var
            break

    if found_env:
        result["auth_ready"] = True
        result["auth_source"] = f"env:{found_env}"
        return result

    # Check credential pool
    if _has_pool_credential_for_provider(resolved):
        result["auth_ready"] = True
        result["auth_source"] = "credential_pool"
        return result

    # Also check the credential pool for the original provider name
    if resolved != provider_raw.lower().strip() and _has_pool_credential_for_provider(
        provider_raw.lower().strip()
    ):
        result["auth_ready"] = True
        result["auth_source"] = "credential_pool"
        return result

    # Special cases: OAuth providers don't always have env vars
    oauth_like = {
        "openai-codex",
        "xai-oauth",
        "qwen-oauth",
        "minimax-oauth",
        "nous",
        "copilot-acp",
    }
    if resolved in oauth_like:
        # For OAuth providers, the credential pool is the only source
        result["auth_errors"].append(
            "OAuth credential required — run 'hermes auth add <provider>'"
        )
    else:
        # If there are env vars to check, list what's expected
        if env_vars:
            result["auth_errors"].append(
                f"no credential found — expected {', '.join(env_vars)} env var(s) or credential pool entry"
            )
        else:
            result["auth_errors"].append(
                "no auth method detected for this provider"
            )

    return result


def _resolve_entry_endpoint(
    entry: dict[str, Any],
) -> tuple[str, str, str]:
    """Resolve the base URL and API key for a fallback entry.

    Returns ``(base_url, api_key, api_mode)``.
    """
    provider_raw = str(entry.get("provider") or "").strip()
    resolved = _resolve_provider_name(provider_raw)

    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY.get(resolved)

    # Base URL
    base_url = str(entry.get("base_url") or "").strip()
    if not base_url and pconfig:
        base_url = pconfig.inference_base_url
    if not base_url:
        base_url = "https://api.openai.com/v1"  # generic fallback

    # API key - try env vars first
    import os

    api_key = ""
    env_vars = _get_credential_env_for_provider(resolved)
    for var in env_vars:
        val = os.environ.get(var, "")
        from hermes_cli.auth import has_usable_secret

        if has_usable_secret(val):
            api_key = val
            break

    # If no env var, try credential pool
    if not api_key:
        try:
            from hermes_cli.auth import read_credential_pool

            pool = read_credential_pool(resolved)
            if pool and isinstance(pool, list) and pool:
                entry_data = pool[0]
                if isinstance(entry_data, dict):
                    api_key = (
                        str(entry_data.get("access_token") or entry_data.get("api_key") or "")
                    )
        except Exception:
            pass

    # API mode
    from hermes_cli.auth import PROVIDER_REGISTRY as _PR

    pcfg = _PR.get(resolved)
    api_mode = "chat_completions"
    if pcfg and pcfg.auth_type == "oauth_external":
        if resolved == "openai-codex":
            api_mode = "codex_responses"
        elif resolved == "xai-oauth":
            api_mode = "codex_responses"
    if resolved in {"anthropic"}:
        api_mode = "anthropic_messages"

    return base_url, api_key, api_mode


def _check_entry_live(
    entry: dict[str, Any],
    max_tokens: int = 3,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Make a minimal inference request to verify this entry is actually usable.

    Returns the check result dict with ``inference_ready``, ``latency_ms``, etc.
    This is NOT a full AIAgent session — no SOUL, no tools, no memory, no
    conversation history. Just a single minimal API call.
    """
    provider_raw = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip()
    base_url, api_key, api_mode = _resolve_entry_endpoint(entry)

    result: dict[str, Any] = {
        "inference_ready": False,
        "latency_ms": 0,
        "error_class": "",
        "error_detail": "",
        "model_returned": "",
    }

    if not api_key:
        result["error_class"] = "auth"
        result["error_detail"] = "no API key/credential available for live check"
        return result

    start = time.monotonic()
    try:
        # Build a minimal chat completion request using raw HTTP
        # (avoids importing openai or constructing a full client)
        url = base_url.rstrip("/") + "/chat/completions"

        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with: OK"}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "hermes-cli/fallback-check",
            },
            method="POST",
        )

        from hermes_cli.urllib_security import open_credentialed_url

        with open_credentialed_url(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        elapsed_ms = int((time.monotonic() - start) * 1000)
        result["latency_ms"] = elapsed_ms

        # Extract the model that actually responded
        returned_model = ""
        if isinstance(raw, dict):
            returned_model = str(raw.get("model") or raw.get("id") or "")

        result["model_returned"] = returned_model

        # Verify the response
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not choices or not isinstance(choices, list):
            result["error_class"] = "response"
            result["error_detail"] = f"unexpected response shape: {type(raw).__name__}"
            return result

        result["inference_ready"] = True

        # Check that the model attribution matches what was requested
        # Strips prefixes like "openrouter/" or "google/" for comparison
        def _normalize_model_id(mid: str) -> str:
            return mid.strip().lower().lstrip("/").split("/")[-1] if "/" in mid else mid.strip().lower()

        if returned_model and _normalize_model_id(returned_model) != _normalize_model_id(model):
            result["inference_ready"] = False
            result["error_class"] = "model_mismatch"
            result["error_detail"] = (
                f"requested '{model}' but '{returned_model}' answered"
            )

    except urllib.error.HTTPError as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        result["latency_ms"] = elapsed_ms
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        result["error_class"] = f"http_{exc.code}"
        result["error_detail"] = f"HTTP {exc.code}: {body_text or exc.reason}"

    except urllib.error.URLError as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        result["latency_ms"] = elapsed_ms
        result["error_class"] = "connection"
        result["error_detail"] = str(exc.reason)

    except TimeoutError:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        result["latency_ms"] = elapsed_ms
        result["error_class"] = "timeout"
        result["error_detail"] = f"request timed out after {timeout}s"

    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        result["latency_ms"] = elapsed_ms
        result["error_class"] = type(exc).__name__
        result["error_detail"] = str(exc)

    return result


def _format_check_table(results: list[dict[str, Any]]) -> str:
    """Format check results as a human-readable table."""
    lines: list[str] = []
    lines.append("")
    lines.append(
        f"  {'Provider/model':<42} {'Config':<9} {'Auth':<9} {'Inference':<10} {'Latency':<9} Status"
    )
    lines.append(
        f"  {'─' * 42} {'─' * 9} {'─' * 9} {'─' * 10} {'─' * 9} ──────────"
    )

    for r in results:
        label = f"{r['provider']} / {r['model']}"[:42]
        if r.get("base_url"):
            label = f"{label} [{r['base_url']}]"[:42]

        config_status = "ready" if r.get("configured") else "FAIL"
        auth_status = "ready" if r.get("auth_ready") else "FAIL"

        if r.get("inference_ready"):
            inference_status = "passed"
        elif r.get("error_class"):
            inference_status = "FAIL"
        else:
            inference_status = "—"

        latency = ""
        if r.get("latency_ms"):
            latency = f"{r['latency_ms']} ms"
        elif r.get("inference_ready") is not None and not r.get("latency_ms"):
            latency = "—"

        if r.get("inference_ready"):
            status = "READY"
        elif r.get("error_class"):
            status = f"FAIL ({r['error_class']})"
        elif not r.get("auth_ready"):
            status = "FAIL (auth)"
        elif not r.get("configured"):
            status = "FAIL (config)"
        elif r.get("inference_ready") is False:
            status = "—"
        else:
            status = "—"

        lines.append(
            f"  {label:<42} {config_status:<9} {auth_status:<9} {inference_status:<10} {latency:<9} {status}"
        )

    # Summary errors
    errors: list[str] = []
    for r in results:
        label = f"{r['provider']} / {r['model']}"
        for e in r.get("config_errors", []):
            errors.append(f"  {label}: config error — {e}")
        for e in r.get("auth_errors", []):
            errors.append(f"  {label}: auth error — {e}")
        if r.get("error_detail"):
            errors.append(f"  {label}: {r['error_detail']}")

    if errors:
        lines.append("")
        lines.append("  Errors:")
        for e in errors:
            lines.append(f"    {e}")

    lines.append("")
    return "\n".join(lines)


def cmd_fallback_check(args) -> None:  # noqa: ARG001
    """Check the readiness of the fallback chain.

    Static mode (default): validate each entry's configuration and auth state
    without making any API calls.

    Live mode (``--live``): send one minimal inference request per entry to
    verify the provider/model actually responds.

    No AIAgent sessions are created.  No gateway messages are sent.
    """
    # Read options
    do_live = getattr(args, "fallback_check_live", False)
    as_json = getattr(args, "fallback_check_json", False)

    from hermes_cli.config import load_config

    config = load_config()
    chain = get_fallback_chain(config)

    if not chain:
        if as_json:
            print(json.dumps({"fallback_chain": [], "results": [], "checked_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        else:
            print()
            print("  No fallback providers configured.")
            print()
        return

    results: list[dict[str, Any]] = []
    for entry in chain:
        # Static checks
        config_result = _check_entry_config(entry)
        auth_result = _check_entry_auth(entry)

        result: dict[str, Any] = {
            "provider": config_result["provider"],
            "model": config_result["model"],
            "base_url": config_result["base_url"],
            "configured": config_result["configured"],
            "auth_ready": auth_result["auth_ready"],
            "config_errors": config_result["config_errors"],
            "auth_errors": auth_result["auth_errors"],
            "auth_source": auth_result.get("auth_source", ""),
            # Live fields — populated only when --live is used
            "inference_ready": None,
            "latency_ms": 0,
            "model_returned": "",
            "error_class": "",
            "error_detail": "",
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        if do_live:
            live_result = _check_entry_live(entry)
            result.update(
                {
                    "inference_ready": live_result["inference_ready"],
                    "latency_ms": live_result["latency_ms"],
                    "model_returned": live_result.get("model_returned", ""),
                    "error_class": live_result.get("error_class", ""),
                    "error_detail": live_result.get("error_detail", ""),
                }
            )
        else:
            # In static mode, inference is considered ready if config + auth pass
            # (without having actually called the provider)
            if result["configured"] and result["auth_ready"]:
                result["inference_ready"] = True

        results.append(result)

    if as_json:
        output = {
            "fallback_chain": chain,
            "results": results,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(_format_check_table(results))
        if not do_live:
            print("  (static check only — use --live for inference verification)")
            print()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def cmd_fallback(args) -> None:
    """Top-level dispatcher for ``hermes fallback [subcommand]``."""
    sub = getattr(args, "fallback_command", None)
    if sub in {None, "", "list", "ls"}:
        cmd_fallback_list(args)
    elif sub == "add":
        cmd_fallback_add(args)
    elif sub in {"remove", "rm"}:
        cmd_fallback_remove(args)
    elif sub in {"check", "chk"}:
        cmd_fallback_check(args)
    elif sub == "clear":
        cmd_fallback_clear(args)
    else:
        print(f"Unknown fallback subcommand: {sub}")
        print("Use one of: list, add, remove, check, clear")
        raise SystemExit(2)
