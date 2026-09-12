"""Claude Agent SDK provider: structural subscription-credential status.

Split out of ``hermes_cli/auth.py``. The SDK-managed Claude Code CLI self-authenticates with the
Claude subscription (OAuth) — Hermes resolves NO credentials on this path by design (#25267).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def get_claude_agent_sdk_auth_status() -> Dict[str, Any]:
    """Structural auth status for the claude-agent-sdk provider.

    The SDK-managed Claude Code CLI self-authenticates with the Claude
    subscription (OAuth) — Hermes resolves NO credentials on this path by
    design (#25267). This probe only reports whether a subscription
    credential source is visibly present so setup/doctor can guide the user;
    the SDK remains the authority at session start.

    Deliberately structural (env var / credential files): the macOS Keychain
    store Claude Code uses by default is NOT probed, because a `security`
    lookup can raise an interactive Keychain prompt from `hermes doctor`.
    A Keychain-only login therefore reports logged_in=False here — the hint
    says so instead of pretending certainty.
    """
    info: Dict[str, Any] = {"provider": "claude-agent-sdk"}
    token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if token:
        info["logged_in"] = True
        info["source"] = "CLAUDE_CODE_OAUTH_TOKEN"
        return info
    home = Path.home()
    # Same credential locations agent/anthropic_adapter.py documents for the
    # Claude Code store: ~/.claude/.credentials.json, then ~/.claude.json.
    for cred_path in (home / ".claude" / ".credentials.json", home / ".claude.json"):
        try:
            if cred_path.exists():
                info["logged_in"] = True
                info["source"] = str(cred_path)
                return info
        except OSError:
            continue
    info["logged_in"] = False
    info["hint"] = (
        "No subscription credential detected: run `claude setup-token` (or "
        "`claude login`) on this machine, or set CLAUDE_CODE_OAUTH_TOKEN. "
        "macOS Keychain-stored logins are not probed here and still work at "
        "session start."
    )
    return info
