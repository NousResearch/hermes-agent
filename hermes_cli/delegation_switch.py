"""Delegation API key hotswap support for the ``/apikey-d`` slash command.

Provides argument parsing and runtime/persistent updates for the credentials
used by :func:`tools.delegate_tool.delegate_task`.  This lets users hotswap
the model/provider/key used by subagents ("droogs") independently of the main
agent.

This module deliberately does **not** import ``cli.py``.  Importing ``cli`` in
processes that do not already host the CLI (e.g. the gateway) is expensive and
can be mistaken for "initialising the agent".  Runtime config is updated only
when the ``cli`` module is already loaded; otherwise changes are persisted so
that the next spawn reads them from disk.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from dataclasses import dataclass
from typing import Optional

from hermes_cli.apikey_switch import mask_api_key, resolve_provider_key_env

logger = logging.getLogger(__name__)


@dataclass
class DelegationSwitchResult:
    """Result of a delegation credential hotswap attempt."""

    success: bool
    message: str = ""
    provider: str = ""
    model: str = ""
    api_key: str = ""
    saved_to_config: bool = False


def parse_api_d_args(raw_args: str) -> tuple[argparse.Namespace, list[str]]:
    """Parse the argument string passed to ``/apikey-d``.

    Returns a tuple of (parsed_args, errors).  Errors are returned rather than
    raised so callers can present them without printing argparse help to the
    chat.
    """
    parser = argparse.ArgumentParser(prog="/apikey-d", add_help=False)
    parser.add_argument("--save", "-s", action="store_true", dest="save")
    parser.add_argument("--reload", "-r", action="store_true", dest="reload")
    parser.add_argument("--provider", "-p", default="", dest="provider")
    parser.add_argument("--model", "-m", default="", dest="model")
    parser.add_argument("key", nargs="?", default="")

    try:
        tokens = shlex.split(raw_args) if raw_args else []
        args = parser.parse_args(tokens)
    except SystemExit:
        return (
            argparse.Namespace(
                save=False, reload=False, provider="", model="", key=""
            ),
            ["usage: /apikey-d [--save] [--provider name] [--model name] [KEY]"],
        )
    except Exception as exc:
        return (
            argparse.Namespace(
                save=False, reload=False, provider="", model="", key=""
            ),
            [str(exc)],
        )

    return args, []


def _runtime_config() -> dict | None:
    """Return the live ``CLI_CONFIG`` dict if the ``cli`` module is loaded."""
    cli_mod = sys.modules.get("cli")
    if cli_mod is not None:
        return getattr(cli_mod, "CLI_CONFIG", None)
    return None


def _persistent_config() -> dict:
    """Return the persistent config dict from ``config.yaml``."""
    try:
        from hermes_cli.config import load_config

        return load_config() or {}
    except Exception:
        return {}


def get_delegation_config() -> dict:
    """Return the current delegation config dict from runtime or disk."""
    runtime = _runtime_config()
    if runtime is not None:
        cfg = runtime.get("delegation")
        if cfg:
            return cfg
    return _persistent_config().get("delegation") or {}


def format_api_d_status(provider: str, model: str, api_key: Optional[str]) -> str:
    """Format the no-argument ``/apikey-d`` response."""
    masked = mask_api_key(api_key)
    lines = [
        f"Delegation provider: {provider or '(inherit from parent)'}" if provider else "Delegation provider: (inherit from parent)",
        f"Delegation model:    {model or '(inherit from parent)'}" if model else "Delegation model:    (inherit from parent)",
        f"Delegation key:      {masked}",
    ]
    return "\n".join(lines)


def apply_api_d_switch(
    provider: str,
    model: str,
    api_key: str,
    save_to_config: bool = False,
) -> DelegationSwitchResult:
    """Apply new delegation credentials without importing ``cli.py``.

    Args:
        provider: Provider slug for subagents (empty = keep current).
        model: Model name for subagents (empty = keep current).
        api_key: API key for subagents (empty = keep current).
        save_to_config: If True, persist the changes to the active config file.
            In gateway mode (no runtime CLI_CONFIG), --save is required to
            persist; without it the change is rejected with guidance.

    Returns:
        :class:`DelegationSwitchResult` describing the outcome.
    """
    runtime = _runtime_config()
    cfg = get_delegation_config()

    # Read current values so we can report them and keep unspecified fields.
    current_provider = str(cfg.get("provider") or "")
    current_model = str(cfg.get("model") or "")
    current_key = str(cfg.get("api_key") or "")
    current_base_url = str(cfg.get("base_url") or "")

    new_provider = provider or current_provider
    new_model = model or current_model
    new_key = api_key or current_key

    if not provider and not model and not api_key:
        return DelegationSwitchResult(
            success=False,
            message="No provider, model, or API key provided.",
            provider=current_provider,
            model=current_model,
            api_key=current_key,
        )

    # Validate coherence: writing only delegation.api_key is ineffective when
    # delegation inherits the parent (no provider or base_url configured),
    # because _resolve_delegation_credentials() returns api_key=None in that
    # case so the child inherits the parent's key.  Warn the user rather than
    # silently doing nothing useful.
    warnings = []
    if api_key and not new_provider and not current_base_url:
        warnings.append(
            "Note: delegation.api_key alone has no effect when no "
            "delegation.provider or delegation.base_url is configured, "
            "because subagents inherit the parent agent's key. "
            "Use --provider to set a separate delegation provider."
        )
    # An existing delegation.base_url takes precedence over a newly-set
    # delegation.provider in _resolve_delegation_credentials().  Warn so the
    # user knows the provider change will be ignored until base_url is cleared.
    if provider and current_base_url and not current_provider:
        warnings.append(
            f"Note: delegation.base_url is set ({current_base_url}). "
            "It takes precedence over delegation.provider, so the new "
            "provider will not take effect until delegation.base_url is cleared."
        )

    # In gateway mode (no runtime CLI_CONFIG loaded), only persist when the
    # user explicitly passes --save.  Without --save, the change would be
    # written to disk silently, making the save flag meaningless on that
    # surface.  Instead, inform the user that --save is required.
    if runtime is None and not save_to_config:
        return DelegationSwitchResult(
            success=False,
            message=(
                "No runtime config available in this process (gateway mode). "
                "Use --save to persist delegation credentials to config.yaml."
            )
            + (" " + " ".join(warnings) if warnings else ""),
            provider=current_provider,
            model=current_model,
            api_key=current_key,
        )

    # Update runtime config when the cli module is already loaded.  This keeps
    # the hotswap immediate for the current process without forcing a heavy
    # import of cli.py in processes that do not already use it.
    if runtime is not None:
        delegation_cfg = runtime.setdefault("delegation", {})
        if provider:
            delegation_cfg["provider"] = provider
        if model:
            delegation_cfg["model"] = model
        if api_key:
            delegation_cfg["api_key"] = api_key

    # Persist the change when requested.
    saved = False
    if save_to_config:
        try:
            from hermes_cli.config import save_config_value

            saved = True
            if provider:
                ok = save_config_value("delegation.provider", provider)
                saved = saved and ok
            if model:
                ok = save_config_value("delegation.model", model)
                saved = saved and ok
            if api_key:
                ok = save_config_value("delegation.api_key", api_key)
                saved = saved and ok
            if not saved:
                return DelegationSwitchResult(
                    success=False,
                    message="Runtime updated, but failed to persist to config.yaml.",
                    provider=new_provider,
                    model=new_model,
                    api_key=new_key,
                )
        except Exception as exc:
            return DelegationSwitchResult(
                success=False,
                message=f"Runtime updated, but failed to persist to config.yaml: {exc}",
                provider=new_provider,
                model=new_model,
                api_key=new_key,
            )

    message = "Delegation credentials updated."
    if warnings:
        message += " " + " ".join(warnings)
    if saved:
        message += " (saved to config.yaml)"

    return DelegationSwitchResult(
        success=True,
        message=message,
        provider=new_provider,
        model=new_model,
        api_key=new_key,
        saved_to_config=saved,
    )


def reload_delegation_key_from_env(provider: str) -> Optional[str]:
    """Reload .env and return the delegation API key for ``provider``."""
    try:
        from hermes_cli.config import reload_env

        reload_env()
    except Exception:
        pass

    if not provider:
        return None

    key_env = resolve_provider_key_env(provider)
    value = (
        __import__("os").environ.get(key_env, "").strip() or None
    )
    if value is not None:
        return value

    try:
        from hermes_cli.config import load_env

        return load_env().get(key_env, "").strip() or None
    except Exception:
        return None
