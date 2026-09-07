"""Claude Agent SDK provider wizard (subscription OAuth owned by the SDK's Claude Code CLI).

There is no Hermes login flow and no API key to collect: credential status is reported
structurally and the CLI re-checks auth authoritatively at session start. Imports of
hermes_cli.auth / hermes_cli.models stay lazy (tests patch them at call time).
"""

from __future__ import annotations


def _model_flow_claude_agent_sdk(_config, current_model=""):
    """Claude Agent SDK provider: subscription OAuth is owned by the SDK's
    Claude Code CLI subprocess — there is no Hermes login flow and no API
    key to collect. Report credential status structurally, then pick a
    model (the CLI re-checks auth authoritatively at session start)."""
    from hermes_cli.auth import (
        get_claude_agent_sdk_auth_status,
        _prompt_model_selection,
        _save_model_choice,
        _update_config_for_provider,
    )
    from hermes_cli.models import _PROVIDER_MODELS

    status = get_claude_agent_sdk_auth_status()
    if status.get("logged_in"):
        print(f"  Claude subscription credentials: ✓ ({status.get('source', '')})")
    else:
        print("No Claude subscription credential detected.")
        print("The SDK authenticates itself: run `claude setup-token` (or `claude")
        print("login`) on this machine, or set CLAUDE_CODE_OAUTH_TOKEN. Continuing —")
        print("the SDK re-checks at session start (macOS Keychain logins are not")
        print("probed here).")
    print()

    # Same Claude model ids the anthropic picker offers; the SDK also accepts
    # an unset model (CLI default), but the picker keeps an explicit choice.
    models = list(
        _PROVIDER_MODELS.get("claude-agent-sdk")
        or _PROVIDER_MODELS.get("anthropic")
        or []
    )
    selected = _prompt_model_selection(models, current_model=current_model)
    if selected:
        _save_model_choice(selected)
        # Empty base URL on purpose: the runtime resolver returns base_url ""
        # for this provider (the SDK owns the endpoint), and passing "" also
        # clears any stale base_url from a previous provider.
        _update_config_for_provider("claude-agent-sdk", "")
        print(f"Default model set to: {selected} (via Claude Agent SDK — Claude subscription)")
    else:
        print("No change.")
