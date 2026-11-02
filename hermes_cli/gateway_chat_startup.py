"""Frontend startup checks shared by classic chat and top-level one-shot."""
from __future__ import annotations

import os


def ensure_launch_provider(args) -> bool:
    from hermes_cli.gateway_chat import bypass_launch
    if bypass_launch(args):
        # The owner freezes code defaults for this launch; the profile is the thing under
        # investigation, so the client must not probe or repair it either.
        return True
    from hermes_cli.main import _first_run_setup_guard, _has_any_provider_configured

    # A resumed/remote session owns its provider policy. Explicit endpoint/key
    # launches can also be usable without a configured local profile.
    if (
        getattr(args, "resume", None)
        or os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
        or getattr(args, "base_url", None)
        or getattr(args, "api_key", None)
    ):
        return True
    from hermes_cli.free_tier_bootstrap import run_bootstrap

    run_bootstrap(announce=False)
    if _has_any_provider_configured():
        return True
    _first_run_setup_guard(args)
    return False


def launch_gateway_chat(args) -> int:
    try:
        from hermes_cli.main import _confirm_startup_expensive_model_override
        from hermes_cli.gateway_chat import launch_from_args, bypass_launch

        # Consent belongs before discovery/admission, even with --yolo. A bypass launch has
        # no profile-derived model/provider to guard and must not read config.yaml here.
        if not bypass_launch(args):
            _confirm_startup_expensive_model_override(args)
        return launch_from_args(args)
    except ImportError as exc:
        from hermes_constants import emit_partial_update_hint

        if emit_partial_update_hint(exc):
            return 1
        raise
