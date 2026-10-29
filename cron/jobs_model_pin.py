"""The main agent's current ``(provider, model)``, which ``pinned=True`` locks onto a cron job."""

from __future__ import annotations

import contextlib
from typing import Optional

from hermes_constants import get_hermes_home


def _main_model_pin() -> tuple[Optional[str], Optional[str]]:
    """``(provider, model)`` the main agent runs on right now (``model.default`` + the provider it
    resolves to), for ``pinned=True`` jobs: the lock is a plain per-job pin, so the scheduler needs
    no second precedence axis. ``(None, None)`` when nothing is configured (the job stays unpinned).

    ``ValueError`` when the config backend cannot serve the profile (``ConfigBackendUnavailable``):
    answering "nothing configured" would silently drop the lock the caller asked for, so the
    create/update is refused like any other invalid input, and the caller's process lives on."""
    from cron.jobs import _normalize_job_optional_text  # late: the facade imports this module
    from hermes_cli.config_backend import ConfigBackendUnavailable, config_exists
    from hermes_cli.config_effective import load_user_config_effective
    cfg_path = get_hermes_home() / "config.yaml"
    try:
        cfg = load_user_config_effective(cfg_path) if config_exists(cfg_path) else {}
    except ConfigBackendUnavailable as exc:
        raise ValueError(f"cannot pin the job to the main model: {exc.code}") from exc
    model_cfg = cfg.get("model") or {}
    model = model_cfg.get("default") or model_cfg.get("model") if isinstance(model_cfg, dict) else model_cfg
    model = _normalize_job_optional_text(model)
    if not model:
        return None, None
    provider = None
    with contextlib.suppress(Exception):
        from hermes_cli.runtime_provider import resolve_runtime_provider
        provider = _normalize_job_optional_text(resolve_runtime_provider(requested=None).get("provider"))
    return (provider.lower() if provider else None), model
