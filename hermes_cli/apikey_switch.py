"""API key hotswap support for the ``/apikey`` slash command.

Provides argument parsing, provider-to-env-var resolution, .env persistence,
and the small amount of runtime orchestration shared between the CLI and the
messaging gateway.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
from dataclasses import dataclass
from typing import Optional

from hermes_cli.auth import PROVIDER_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class ApiKeySwitchResult:
    """Result of an API key hotswap attempt."""

    success: bool
    message: str = ""
    provider: str = ""
    model: str = ""
    key_env: str = ""
    saved_to_env: bool = False


def resolve_provider_key_env(provider: str) -> str:
    """Return the preferred env var name for ``provider``'s API key.

    Checks the provider registry, user-defined providers (``providers:``),
    and custom providers (``custom_providers:``) in config.yaml.  Falls back
    to a conventional ``PROVIDER_API_KEY`` name when nothing else matches.
    """
    provider = (provider or "").strip().lower()
    if not provider:
        return "OPENROUTER_API_KEY"

    # OpenRouter is not in PROVIDER_REGISTRY but is the default aggregator.
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"

    pconfig = PROVIDER_REGISTRY.get(provider)
    if pconfig and pconfig.api_key_env_vars:
        return pconfig.api_key_env_vars[0]

    # User-defined providers and custom endpoints may declare key_env.
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception:
        cfg = {}

    user_providers = cfg.get("providers", {}) or {}
    if isinstance(user_providers, dict) and provider in user_providers:
        key_env = str(user_providers[provider].get("key_env", "")).strip()
        if key_env:
            return key_env

    custom_providers = cfg.get("custom_providers", []) or []
    if isinstance(custom_providers, list):
        for entry in custom_providers:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip()
            slug = f"custom:{name}" if name else ""
            if slug == provider or name == provider:
                key_env = str(entry.get("key_env", "")).strip()
                if key_env:
                    return key_env

    # Conventional fallback.
    return f"{provider.upper().replace('-', '_')}_API_KEY"


def parse_apikey_args(raw_args: str) -> tuple[argparse.Namespace, list[str]]:
    """Parse the argument string passed to ``/apikey``.

    Returns a tuple of (parsed_args, errors).  Errors are returned rather than
    raised so callers can present them without printing argparse help to the
    chat.
    """
    parser = argparse.ArgumentParser(prog="/apikey", add_help=False)
    parser.add_argument("--save", "-s", action="store_true", dest="save")
    parser.add_argument("--reload", "-r", action="store_true", dest="reload")
    parser.add_argument("key", nargs="?", default="")

    try:
        tokens = shlex.split(raw_args) if raw_args else []
        args = parser.parse_args(tokens)
    except SystemExit:
        # argparse calls sys.exit on --help; treat as a parse error.
        return (
            argparse.Namespace(save=False, reload=False, key=""),
            ["usage: /apikey [--save] [--reload] [KEY]"],
        )
    except Exception as exc:
        return (
            argparse.Namespace(save=False, reload=False, key=""),
            [str(exc)],
        )

    return args, []


def save_api_key_to_env(provider: str, api_key: str) -> tuple[bool, str]:
    """Persist ``api_key`` to ``~/.hermes/.env`` under the provider's env var.

    Returns ``(True, env_var_name)`` on success, or ``(False, error_message)``.
    """
    if not api_key:
        return False, "API key is empty"

    key_env = resolve_provider_key_env(provider)
    try:
        from hermes_cli.config import save_env_value

        save_env_value(key_env, api_key)
        return True, key_env
    except Exception as exc:
        logger.warning("Failed to save API key for %s: %s", provider, exc)
        return False, str(exc)


def resolve_current_key(provider: str) -> Optional[str]:
    """Return the current API key for ``provider`` from os.environ or .env."""
    key_env = resolve_provider_key_env(provider)
    value = os.environ.get(key_env, "").strip() or None
    if value is not None:
        return value
    try:
        from hermes_cli.config import load_env

        return load_env().get(key_env, "").strip() or None
    except Exception:
        return None


def mask_api_key(key: Optional[str]) -> str:
    """Return a display-safe representation of an API key."""
    if not key:
        return "(not set)"
    try:
        from agent.redact import mask_secret

        return mask_secret(key)
    except Exception:
        # Defensive fallback: show first/last few chars.
        if len(key) <= 12:
            return "***"
        return f"{key[:4]}...{key[-4:]}"


def format_apikey_status(provider: str, model: str, api_key: Optional[str]) -> str:
    """Format the no-argument ``/apikey`` response."""
    key_env = resolve_provider_key_env(provider)
    masked = mask_api_key(api_key)
    lines = [
        f"Provider: {provider or '(unknown)'}",
        f"Model:    {model or '(unknown)'}" if model else "",
        f"Key env:  {key_env}",
        f"Key:      {masked}",
    ]
    return "\n".join(line for line in lines if line is not None)


def apply_api_key_switch(
    agent,
    provider: str,
    model: str,
    api_key: str,
    save_to_env: bool = False,
) -> ApiKeySwitchResult:
    """Apply a new API key to a live agent.

    Args:
        agent: A live :class:`run_agent.AIAgent` instance, or ``None`` if the
            caller only wants to persist the key (e.g. gateway cache eviction).
        provider: The provider slug the key belongs to.
        model: The current model name (for display only).
        api_key: The new API key.
        save_to_env: If True, persist the key to ``~/.hermes/.env``.

    Returns:
        :class:`ApiKeySwitchResult` describing the outcome.
    """
    if not provider:
        return ApiKeySwitchResult(
            success=False,
            message="No provider is currently active.",
        )

    if not api_key:
        return ApiKeySwitchResult(
            success=False,
            message="No API key provided.",
            provider=provider,
            model=model,
        )

    saved = False
    key_env = ""
    if save_to_env:
        saved, key_env_or_err = save_api_key_to_env(provider, api_key)
        if not saved:
            return ApiKeySwitchResult(
                success=False,
                message=f"Failed to persist API key: {key_env_or_err}",
                provider=provider,
                model=model,
            )
        key_env = key_env_or_err
        try:
            from hermes_cli.config import reload_env

            reload_env()
        except Exception:
            pass

    if agent is not None:
        try:
            agent.switch_api_key(api_key, provider=provider)
        except Exception as exc:
            logger.warning("Live API key hotswap failed: %s", exc)
            return ApiKeySwitchResult(
                success=False,
                message=f"Agent rejected the new key: {exc}",
                provider=provider,
                model=model,
                key_env=key_env,
                saved_to_env=saved,
            )

    return ApiKeySwitchResult(
        success=True,
        message="API key hotswapped successfully.",
        provider=provider,
        model=model,
        key_env=key_env,
        saved_to_env=saved,
    )
