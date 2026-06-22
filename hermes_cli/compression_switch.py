"""Compression API key hotswap support for the ``/apikey-c`` slash command.

Provides argument parsing and runtime/persistent updates for the credentials
used by auxiliary context compression.  This lets users hotswap the
model/provider/key used by compression independently of the main agent and
subagent delegation.

Mirrors :mod:`hermes_cli.delegation_switch` but targets ``auxiliary.compression``
instead of ``delegation``.  Unlike delegation, compression config is read via
``load_config()`` (disk-cached on mtime), not from ``CLI_CONFIG`` at runtime.
This means we always persist to disk via ``save_config_value()`` — the file
mtime change invalidates the ``load_config()`` cache so the next compression
call picks up the new key immediately, with no process restart.
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
class CompressionSwitchResult:
    """Result of a compression credential hotswap attempt."""

    success: bool
    message: str = ""
    provider: str = ""
    model: str = ""
    api_key: str = ""
    saved_to_config: bool = False


def parse_api_c_args(raw_args: str) -> tuple[argparse.Namespace, list[str]]:
    """Parse the argument string passed to ``/apikey-c``.

    Returns a tuple of (parsed_args, errors).  Errors are returned rather than
    raised so callers can present them without printing argparse help to the
    chat.
    """
    parser = argparse.ArgumentParser(prog="/apikey-c", add_help=False)
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
            ["usage: /apikey-c [--save] [--provider name] [--model name] [KEY]"],
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


def get_compression_config() -> dict:
    """Return the current compression config dict from runtime or disk."""
    runtime = _runtime_config()
    if runtime is not None:
        aux = runtime.get("auxiliary")
        if aux and isinstance(aux, dict):
            cfg = aux.get("compression")
            if cfg:
                return cfg
    persistent = _persistent_config()
    aux = persistent.get("auxiliary", {})
    if isinstance(aux, dict):
        return aux.get("compression") or {}
    return {}


def format_api_c_status(provider: str, model: str, api_key: Optional[str]) -> str:
    """Format the no-argument ``/apikey-c`` response."""
    masked = mask_api_key(api_key)
    lines = [
        f"Compression provider: {provider or '(auto)'}" if provider else "Compression provider: (auto)",
        f"Compression model:    {model or '(use provider default)'}" if model else "Compression model:    (use provider default)",
        f"Compression key:     {masked}",
    ]
    return "\n".join(lines)


def apply_api_c_switch(
    provider: str,
    model: str,
    api_key: str,
    save_to_config: bool = False,
) -> CompressionSwitchResult:
    """Apply new compression credentials.

    Unlike delegation, compression config is read via ``load_config()`` at
    runtime (disk-cached on mtime), not from ``CLI_CONFIG``.  Therefore we
    **always** persist to disk so the ``load_config()`` cache invalidates and
    the next compression call picks up the new key.  The ``save_to_config``
    flag is accepted for API consistency with ``/apikey-d`` but is effectively
    always on for compression.

    Args:
        provider: Provider slug for compression (empty = keep current).
        model: Model name for compression (empty = keep current).
        api_key: API key for compression (empty = keep current).
        save_to_config: Accepted for API consistency. Compression always
            persists to disk because ``load_config()`` reads from there.

    Returns:
        :class:`CompressionSwitchResult` describing the outcome.
    """
    runtime = _runtime_config()
    cfg = get_compression_config()

    # Read current values so we can report them and keep unspecified fields.
    current_provider = str(cfg.get("provider") or "")
    current_model = str(cfg.get("model") or "")
    current_key = str(cfg.get("api_key") or "")

    new_provider = provider or current_provider
    new_model = model or current_model
    new_key = api_key or current_key

    if not provider and not model and not api_key:
        return CompressionSwitchResult(
            success=False,
            message="No provider, model, or API key provided.",
            provider=current_provider,
            model=current_model,
            api_key=current_key,
        )

    # Update runtime CLI_CONFIG for consistency (some code paths may read it).
    if runtime is not None:
        aux_cfg = runtime.setdefault("auxiliary", {})
        comp_cfg = aux_cfg.setdefault("compression", {})
        if not isinstance(comp_cfg, dict):
            comp_cfg = {}
            aux_cfg["compression"] = comp_cfg
        if provider:
            comp_cfg["provider"] = provider
        if model:
            comp_cfg["model"] = model
        if api_key:
            comp_cfg["api_key"] = api_key

    # Always persist to disk for compression — load_config() reads from here,
    # and the mtime change invalidates its cache so the next call is immediate.
    saved = False
    try:
        from hermes_cli.config import save_config_value

        saved = True
        if provider:
            ok = save_config_value("auxiliary.compression.provider", provider)
            saved = saved and ok
        if model:
            ok = save_config_value("auxiliary.compression.model", model)
            saved = saved and ok
        if api_key:
            ok = save_config_value("auxiliary.compression.api_key", api_key)
            saved = saved and ok
        if not saved:
            return CompressionSwitchResult(
                success=False,
                message="Runtime updated, but failed to persist to config.yaml.",
                provider=new_provider,
                model=new_model,
                api_key=new_key,
            )
    except Exception as exc:
        return CompressionSwitchResult(
            success=False,
            message=f"Runtime updated, but failed to persist to config.yaml: {exc}",
            provider=new_provider,
            model=new_model,
            api_key=new_key,
        )

    return CompressionSwitchResult(
        success=True,
        message="Compression credentials updated.",
        provider=new_provider,
        model=new_model,
        api_key=new_key,
        saved_to_config=saved,
    )


def reload_compression_key_from_env(provider: str) -> Optional[str]:
    """Reload .env and return the compression API key for ``provider``."""
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
