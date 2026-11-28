"""The main agent's current ``(provider, model)``, which ``pinned=True`` locks onto a cron job."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home


def _model_name_from_block(model_cfg: Any) -> str:
    """The model NAME a ``model:`` config block names: ``model: <name>`` shorthand, or a dict's
    ``default`` / ``model`` / ``name`` key. ``""`` when it names none."""
    from cron.jobs import _normalize_job_optional_text  # late: the facade imports this module
    if isinstance(model_cfg, str):
        return model_cfg.strip()
    if isinstance(model_cfg, dict):
        return _normalize_job_optional_text(
            model_cfg.get("default") or model_cfg.get("model") or model_cfg.get("name")) or ""
    return ""


def _read_model_block(home: Path) -> Any:
    """``model:`` block of *home*'s ``config.yaml`` ({} when absent or empty).

    A NAMED profile's home is self-contained: the effective-config loader reads only that home's
    ``config.yaml``, so a profile with no ``model:`` block has no default model of its own.
    """
    from hermes_cli.config_backend import config_exists
    from hermes_cli.config_effective import load_user_config_effective

    cfg_path = Path(home) / "config.yaml"
    cfg = load_user_config_effective(cfg_path) if config_exists(cfg_path) else {}
    return (cfg.get("model") or {}) if isinstance(cfg, dict) else {}


def host_default_model(active_home: Optional[Path] = None) -> tuple[str, str, Any]:
    """``(model, provider, model_cfg)`` a NAMED profile inherits from the HOST root config.

    The active home's own config is the first source (see ``_main_model_pin`` and the scheduler);
    when it carries no ``model:`` block, a named profile (``<root>/profiles/<name>``) falls back to
    the HOST root's ``<root>/config.yaml`` default — the same default the root profile runs on — so
    an unpinned cron job created in that profile's store RUNS instead of dying with "has no model
    configured" (card t_92b4a684). ``("", "", {})`` when the active home IS the root, or when the
    host config names no model either.
    """
    from hermes_constants import get_default_hermes_root
    from cron.jobs import _normalize_job_optional_text  # late: the facade imports this module

    active = Path(active_home) if active_home is not None else get_hermes_home()
    try:
        root = Path(get_default_hermes_root(home=active))
        same_root = root.resolve() == active.resolve()
    except OSError:  # pragma: no cover - unreadable path
        return "", "", {}
    if same_root:
        return "", "", {}
    # Best effort: the profile's own config already answered "no model of mine", so an unreadable
    # host root means "no inheritance", never a failed pin on a profile that may be fine.
    with contextlib.suppress(Exception):
        block = _read_model_block(root)
        model = _model_name_from_block(block)
        if model:
            provider = ""
            if isinstance(block, dict):
                provider = _normalize_job_optional_text(block.get("provider")) or ""
            return model, provider, block
    return "", "", {}


def _main_model_pin() -> tuple[Optional[str], Optional[str]]:
    """``(provider, model)`` the main agent runs on right now (``model.default`` + the provider it
    resolves to), for ``pinned=True`` jobs: the lock is a plain per-job pin, so the scheduler needs
    no second precedence axis. A NAMED profile with no ``model:`` of its own inherits the HOST root
    default (``host_default_model``); ``(None, None)`` when nothing resolves anywhere — the callers
    in ``cron.jobs`` REFUSE rather than persisting a silently-dead pinned job (card t_92b4a684).

    ``ValueError`` when the config backend cannot serve the profile (``ConfigBackendUnavailable``):
    answering "nothing configured" would silently drop the lock the caller asked for, so the
    create/update is refused like any other invalid input, and the caller's process lives on.
    """
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from cron.jobs import _normalize_job_optional_text  # late: the facade imports this module
    home = get_hermes_home()
    try:
        model = _model_name_from_block(_read_model_block(home))
    except ConfigBackendUnavailable as exc:
        raise ValueError(f"cannot pin the job to the main model: {exc.code}") from exc
    requested: Optional[str] = None
    if not model:
        _host_model, _host_provider, _host_model_cfg = host_default_model(home)
        if _host_model:
            model = _host_model
            requested = _host_provider or None
    if not model:
        return None, None
    provider = None
    with contextlib.suppress(Exception):
        from hermes_cli.runtime_provider import resolve_runtime_provider
        provider = _normalize_job_optional_text(resolve_runtime_provider(requested=requested).get("provider"))
    return (provider.lower() if provider else None), model
