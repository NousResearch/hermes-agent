"""``hermes model --list``: print a provider's model catalog without the interactive picker.

The picker needs a TTY, so scripts, CI and remote shells had no way to ask which model ids a
provider serves. This reads the same catalog the picker shows (``provider_model_ids``).
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _resolve_provider(requested: str | None) -> str:
    """The provider to list: ``--provider``, else the configured one, with ``auto`` resolved."""
    from hermes_cli.models import normalize_provider
    from hermes_cli.runtime_provider import resolve_requested_provider

    provider = resolve_requested_provider(requested)
    if provider == "auto":
        from hermes_cli.auth import AuthError, resolve_provider
        try:
            provider = resolve_provider("auto")
        except AuthError:
            return "auto"
    return provider if provider.startswith("custom:") else normalize_provider(provider)


def _configured_model(config: dict[str, Any], provider: str) -> str:
    """The saved default model, only when it belongs to ``provider``."""
    from hermes_cli.models import normalize_provider

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        return model_cfg.strip() if isinstance(model_cfg, str) else ""
    saved = str(model_cfg.get("provider") or "").strip().lower()
    if saved and saved != provider and normalize_provider(saved) != provider:
        return ""
    return str(model_cfg.get("default") or model_cfg.get("name") or "").strip()


def run_model_list(args) -> int:
    """Print the catalog; JSON on stdout with ``--json``. Errors go to stderr with exit 1."""
    from hermes_cli.config import load_config
    from hermes_cli.models import list_available_providers, provider_label, provider_model_ids

    provider = _resolve_provider(getattr(args, "provider", None))
    known = {p["id"] for p in list_available_providers()}
    if provider == "auto":
        print("No provider is configured. Run `hermes model` or pass --provider <id>.", file=sys.stderr)
        return 1
    if provider not in known and not provider.startswith("custom:"):
        print(f"Unknown provider '{provider}'. Known: {', '.join(sorted(known))}", file=sys.stderr)
        return 1

    models = provider_model_ids(provider, force_refresh=bool(getattr(args, "refresh", False)))
    current = _configured_model(load_config(), provider)
    if getattr(args, "json", False):
        print(json.dumps({"schema_version": 1, "provider": provider, "current": current or None, "models": models}))
        return 0
    if not models:
        print(f"No models found for {provider_label(provider)} ({provider}).", file=sys.stderr)
        return 1
    print(f"{provider_label(provider)} ({provider}): {len(models)} models")
    for model_id in models:
        print(f"  {'*' if model_id == current else ' '} {model_id}")
    return 0
