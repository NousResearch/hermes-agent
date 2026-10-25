"""Operator-authorized process bootstrap for cold-profile 1Password fetches."""
from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


def capture_operator_op() -> Path | None:
    """Freeze authority before dotenv can redirect managed policy or executable lookup.

    Read the operator file directly: profile config, overlays and interpolation must
    never grant process-token access. A policy change takes effect after restart.
    """
    from agent.safe_worker_policy import worker_config_snapshot

    if worker_config_snapshot() is not None:
        return None
    from hermes_cli.managed_scope import get_managed_dir
    from hermes_yaml import YAMLError
    from utils import load_yaml_file_readonly

    managed_dir = get_managed_dir()
    if managed_dir is None:
        return None
    try:
        data = load_yaml_file_readonly(managed_dir / "config.yaml") or {}
        cfg = data.get("secrets", {}).get("onepassword", {})
        if cfg.get("share_process_token") is not True:
            return None
        path = Path(cfg.get("binary_path") or "")
        if not path.is_absolute():
            return None
        canonical = path.resolve(strict=True)
        return canonical if canonical.is_file() and os.access(canonical, os.X_OK) else None
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError, YAMLError):
        return None


def private_fetch_inputs(source, cfg: dict, home: Path,
                         environ: MutableMapping[str, str]) -> tuple[dict, MutableMapping[str, str]]:
    """Add a process token only to the exact bundled source's individual fetch view."""
    from hermes_cli import env_loader
    from agent.secret_sources.onepassword import OnePasswordSource, find_op
    from agent.secret_sources.registry import is_bundled_source

    approved = env_loader._PROCESS_TOKEN_OP
    token_name = "OP_SERVICE_ACCOUNT_TOKEN"
    if (environ is os.environ or approved is None or type(source) is not OnePasswordSource
            or not is_bundled_source(source, home)
            or source.token_env(cfg) != token_name or token_name in environ
            or "OP_CONNECT_HOST" in environ or "OP_CONNECT_HOST" in os.environ):
        return cfg, environ
    owned_names = env_loader.launch_dotenv_keys() | set(env_loader.source_supplied_names())
    if token_name in {name.upper() for name in owned_names}:
        return cfg, environ
    value = os.environ.get(token_name)
    if not value:
        return cfg, environ
    binary = find_op(str(cfg.get("binary_path") or ""))
    try:
        if binary is None or binary.resolve(strict=True) != approved:
            return cfg, environ
    except (OSError, ValueError, RuntimeError):
        return cfg, environ
    # Execute the very path we approved, never the mutable PATH/symlink used to find it.
    return dict(cfg, binary_path=str(approved)), dict(environ, **{token_name: value})
