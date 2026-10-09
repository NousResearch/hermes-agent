from hermes_cli.config_credentials import (
    credential_pool_environment as _phase6_auth_environment,
)

"""Interactive Claude login commands."""
import base64
import contextlib
import functools
import hashlib
import json
import logging
import os
import platform
import re
import secrets
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from hermes_constants import get_hermes_home
from utils import atomic_json_write
from agent.secret_scope import get_secret as _get_secret
from auth.providers.anthropic import (
    _OAUTH_CLIENT_ID,
    _OAUTH_REDIRECT_URI,
    _OAUTH_SCOPES,
    _first_env,
    _generate_pkce,
    _oauth_token_state,
    _post_oauth_token,
    is_claude_code_token_valid,
    logger,
    read_claude_code_credentials,
)


def run_oauth_setup_token() -> Optional[str]:
    """Run 'claude setup-token' interactively; the resulting token or None. FileNotFoundError if no 'claude' CLI."""
    import shutil

    claude_path = shutil.which("claude")
    if not claude_path:
        raise FileNotFoundError(
            "The 'claude' CLI is not installed. Install it with: npm install -g @anthropic-ai/claude-code"
        )
    # Interactive: stdio inherited so the user can complete the OAuth prompt.  noqa: subprocess-stdin
    try:
        subprocess.run([claude_path, "setup-token"])
    except (KeyboardInterrupt, EOFError):
        return None
    creds = read_claude_code_credentials(environment=_phase6_auth_environment())
    if creds and is_claude_code_token_valid(creds):
        return creds["accessToken"]
    return _first_env("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN") or None


def run_hermes_oauth_login_pure() -> Optional[Dict[str, Any]]:
    """Run Hermes-native OAuth PKCE flow and return credential state."""
    import webbrowser
    from urllib.parse import urlencode

    verifier, challenge = _generate_pkce()
    oauth_state = secrets.token_urlsafe(32)
    params = {
        "code": "true",
        "client_id": _OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": oauth_state,
    }
    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"
    print(
        "\n".join([
            "",
            "Authorize Hermes with your Claude Pro/Max subscription.",
            "",
            "╭─ Claude Pro/Max Authorization ────────────────────╮",
            "│                                                   │",
            "│  Open this link in your browser:                  │",
            "╰───────────────────────────────────────────────────╯",
            "",
            f"  {auth_url}",
            "",
        ])
    )
    try:
        from hermes_cli.auth_device_flow import (
            _can_open_graphical_browser as _can_open_gui,
        )
    except Exception:
        _can_open_gui = lambda: True  # noqa: E731 — degrade to prior behavior
    if _can_open_gui():
        with contextlib.suppress(Exception):
            webbrowser.open(auth_url)
            print("  (Browser opened automatically)")
    print("\nAfter authorizing, you'll see a code. Paste it below.\n")
    try:
        auth_code = input("Authorization code: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if not auth_code:
        print("No code entered.")
        return None
    splits = auth_code.split("#")
    code, received_state = splits[0], (splits[1] if len(splits) > 1 else "")
    if received_state != oauth_state:  # CSRF guard (RFC 6749 §10.12)
        logger.warning("OAuth state mismatch — possible CSRF, aborting")
        return None
    try:
        exchange_data = json.dumps({
            "grant_type": "authorization_code",
            "client_id": _OAUTH_CLIENT_ID,
            "code": code,
            "state": received_state,
            "redirect_uri": _OAUTH_REDIRECT_URI,
            "code_verifier": verifier,
        }).encode()
        result = _post_oauth_token(
            exchange_data, content_type="application/json", timeout=15, what="exchange"
        )
    except Exception as e:
        print(f"Token exchange failed: {e}")
        return None
    if not result.get("access_token"):
        print("No access token in response.")
        return None
    return _oauth_token_state(result)
