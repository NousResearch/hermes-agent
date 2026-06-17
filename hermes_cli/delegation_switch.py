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
            When no runtime ``CLI_CONFIG`` is loaded, changes are always
            persisted so that future subagent spawns can read them.

    Returns:
        :class:`DelegationSwitchResult` describing the outcome.
    """
    runtime = _runtime_config()
    cfg = get_delegation_config()

    # Read current values so we can report them and keep unspecified fields.
    current_provider = str(cfg.get("provider") or "")
    current_model = str(cfg.get("model") or "")
    current_key = str(cfg.get("api_key") or "")

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

    # Persist the change when requested, or when there is no runtime config in
    # this process (e.g. gateway before any subagent spawn).  Without a runtime
    # config the only place future spawns can read the new credentials is disk.
    saved = False
    if save_to_config or runtime is None:
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
