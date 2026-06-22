"""Delegation API key hotswap support for the ``/apikey-d`` slash command.

Provides argument parsing and runtime/persistent updates for the credentials
used by :func:`tools.delegate_tool.delegate_task`.  This lets users hotswap
the model/provider/key used by subagents ("droogs") independently of the main
agent.
"""

from __future__ import annotations

import argparse
import logging
import shlex
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


def get_delegation_config() -> dict:
    """Return the current delegation config dict from runtime config."""
    try:
        from cli import CLI_CONFIG

        return CLI_CONFIG.get("delegation") or {}
    except Exception:
        return {}


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
    """Apply new delegation credentials to the runtime config.

    Args:
        provider: Provider slug for subagents (empty = keep current).
        model: Model name for subagents (empty = keep current).
        api_key: API key for subagents (empty = keep current).
        save_to_config: If True, persist the changes to the active config file.

    Returns:
        :class:`DelegationSwitchResult` describing the outcome.
    """
    try:
        from cli import CLI_CONFIG
    except Exception as exc:
        return DelegationSwitchResult(
            success=False,
            message=f"Cannot access runtime config: {exc}",
        )

    delegation_cfg = CLI_CONFIG.setdefault("delegation", {})

    # Read current values so we can report them and keep unspecified fields.
    current_provider = str(delegation_cfg.get("provider") or "")
    current_model = str(delegation_cfg.get("model") or "")
    current_key = str(delegation_cfg.get("api_key") or "")

    new_provider = provider or current_provider
    new_model = model or current_model
    new_key = api_key or current_key

    # Update runtime config.
    if provider:
        delegation_cfg["provider"] = provider
    if model:
        delegation_cfg["model"] = model
    if api_key:
        delegation_cfg["api_key"] = api_key

    saved = False
    if save_to_config:
        try:
            from cli import save_config_value

            saved = True
            if provider:
                saved = save_config_value("delegation.provider", provider) and saved
            if model:
                saved = save_config_value("delegation.model", model) and saved
            if api_key:
                saved = save_config_value("delegation.api_key", api_key) and saved
        except Exception as exc:
            return DelegationSwitchResult(
                success=False,
                message=f"Runtime updated, but failed to persist to config.yaml: {exc}",
                provider=new_provider,
                model=new_model,
                api_key=new_key,
            )

    return DelegationSwitchResult(
        success=True,
        message="Delegation credentials updated.",
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
