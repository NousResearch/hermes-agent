"""Claude Agent SDK (subscription) row for ``hermes doctor``. Split out of ``hermes_cli/doctor.py``."""

from __future__ import annotations

from hermes_cli.doctor_report import Finding, check_info, check_ok, check_warn, doctor_check, warn_on_error


@doctor_check()
def _check_claude_agent_sdk_auth(should_fix: bool, f: Finding) -> None:
    """Structural probe (env var / ~/.claude credential files): a macOS Keychain-only login reports as
    not-detected and the hint says so. Its own best-effort block so an import failure cannot disrupt
    the OAuth rows printed before it."""
    with warn_on_error(""):
        from hermes_cli.auth import get_claude_agent_sdk_auth_status
        sdk_status = get_claude_agent_sdk_auth_status() or {}
        if sdk_status.get("logged_in"):
            check_ok("Claude Agent SDK (subscription)", f"({sdk_status.get('source', 'credentials found')})")
        else:
            check_warn("Claude Agent SDK (subscription)", "(no credential detected)")
            if sdk_status.get("hint"):
                check_info(sdk_status["hint"])
        # The SDK python package is an opt-in extra that lazy-installs at first use — mirror the
        # codex-CLI availability hint.
        with warn_on_error(""):
            from tools.lazy_deps import is_available
            if not is_available("provider.claude_agent_sdk"):
                check_info("claude-agent-sdk package not installed (optional — installs at first use, or: "
                           "pip install 'hermes-agent[claude-agent-sdk]')")
