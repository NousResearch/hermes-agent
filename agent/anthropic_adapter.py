"""Anthropic Messages API adapter for Hermes Agent.

Translates between Hermes's internal OpenAI-style message format and
Anthropic's Messages API. Follows the same pattern as the codex_responses
adapter — all provider-specific logic is isolated here.

Auth supports:
  - Regular API keys (sk-ant-api*) → x-api-key header
  - OAuth setup-tokens (sk-ant-oat*) → Bearer auth + beta header
  - Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json) → Bearer auth
"""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

THINKING_BUDGET = {"xhigh": 32000, "high": 16000, "medium": 8000, "low": 4000}
ADAPTIVE_EFFORT_MAP = {
    "xhigh": "max",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "minimal": "low",
}


def _supports_adaptive_thinking(model: str) -> bool:
    """Return True for Claude 4.6 models that support adaptive thinking."""
    return any(v in model for v in ("4-6", "4.6"))


# Beta headers for enhanced features (sent with ALL auth types)
_COMMON_BETAS = [
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
]

# Additional beta headers required for OAuth/subscription auth.
# Matches what Claude Code (and pi-ai / OpenCode) send.
_OAUTH_ONLY_BETAS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
]

# Claude Code identity — required for OAuth requests to be routed correctly.
# Without these, Anthropic's infrastructure intermittently 500s OAuth traffic.
# The version must stay reasonably current — Anthropic rejects OAuth requests
# when the spoofed user-agent version is too far behind the actual release.
_CLAUDE_CODE_VERSION_FALLBACK = "2.1.74"


def _detect_claude_code_version() -> str:
    """Detect the installed Claude Code version, fall back to a static constant.

    Anthropic's OAuth infrastructure validates the user-agent version and may
    reject requests with a version that's too old.  Detecting dynamically means
    users who keep Claude Code updated never hit stale-version 400s.
    """
    import subprocess as _sp

    for cmd in ("claude", "claude-code"):
        try:
            result = _sp.run(
                [cmd, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Output is like "2.1.74 (Claude Code)" or just "2.1.74"
                version = result.stdout.strip().split()[0]
                if version and version[0].isdigit():
                    return version
        except Exception:
            pass
    return _CLAUDE_CODE_VERSION_FALLBACK


_CLAUDE_CODE_VERSION = _detect_claude_code_version()
_CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."
_MCP_TOOL_PREFIX = "mcp_"


def _is_oauth_token(key: str) -> bool:
    """Check if the key is an OAuth/setup token (not a regular Console API key).

    Regular API keys start with 'sk-ant-api'. Everything else (setup-tokens
    starting with 'sk-ant-oat', managed keys, JWTs, etc.) needs Bearer auth.
    """
    if not key:
        return False
    # Regular Console API keys use x-api-key header
    if key.startswith("sk-ant-api"):
        return False
    # Everything else (setup-tokens, managed keys, JWTs) uses Bearer auth
    return True


def build_anthropic_client(api_key: str, base_url: str = None):
    """Create an Anthropic client, auto-detecting setup-tokens vs API keys.

    Returns an anthropic.Anthropic instance.
    """
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic provider. "
            "Install it with: pip install 'anthropic>=0.39.0'"
        )
    from httpx import Timeout

    kwargs = {
        "timeout": Timeout(timeout=900.0, connect=10.0),
    }
    if base_url:
        kwargs["base_url"] = base_url

    if _is_oauth_token(api_key):
        # OAuth access token / setup-token → Bearer auth + Claude Code identity.
        # Anthropic routes OAuth requests based on user-agent and headers;
        # without Claude Code's fingerprint, requests get intermittent 500s.
        all_betas = _COMMON_BETAS + _OAUTH_ONLY_BETAS
        kwargs["auth_token"] = api_key
        kwargs["default_headers"] = {
            "anthropic-beta": ",".join(all_betas),
            "user-agent": f"claude-cli/{_CLAUDE_CODE_VERSION} (external, cli)",
            "x-app": "cli",
        }
    else:
        # Regular API key → x-api-key header + common betas
        kwargs["api_key"] = api_key
        if _COMMON_BETAS:
            kwargs["default_headers"] = {"anthropic-beta": ",".join(_COMMON_BETAS)}

    return _anthropic_sdk.Anthropic(**kwargs)


def read_claude_code_credentials() -> Optional[Dict[str, Any]]:
    """Read refreshable Claude Code OAuth credentials from ~/.claude/.credentials.json.

    This intentionally excludes ~/.claude.json primaryApiKey. Opencode's
    subscription flow is OAuth/setup-token based with refreshable credentials,
    and native direct Anthropic provider usage should follow that path rather
    than auto-detecting Claude's first-party managed key.

    Returns dict with {accessToken, refreshToken?, expiresAt?} or None.
    """
    cred_path = Path.home() / ".claude" / ".credentials.json"
    if cred_path.exists():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            oauth_data = data.get("claudeAiOauth")
            if oauth_data and isinstance(oauth_data, dict):
                access_token = oauth_data.get("accessToken", "")
                if access_token:
                    return {
                        "accessToken": access_token,
                        "refreshToken": oauth_data.get("refreshToken", ""),
                        "expiresAt": oauth_data.get("expiresAt", 0),
                        "source": "claude_code_credentials_file",
                    }
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude/.credentials.json: %s", e)

    return None


def read_claude_managed_key() -> Optional[str]:
    """Read Claude's native managed key from ~/.claude.json for diagnostics only."""
    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            data = json.loads(claude_json.read_text(encoding="utf-8"))
            primary_key = data.get("primaryApiKey", "")
            if isinstance(primary_key, str) and primary_key.strip():
                return primary_key.strip()
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read ~/.claude.json: %s", e)
    return None


def is_claude_code_token_valid(creds: Dict[str, Any]) -> bool:
    """Check if Claude Code credentials have a non-expired access token."""
    import time

    expires_at = creds.get("expiresAt", 0)
    if not expires_at:
        # No expiry set (managed keys) — valid if token is present
        return bool(creds.get("accessToken"))

    # expiresAt is in milliseconds since epoch
    now_ms = int(time.time() * 1000)
    # Allow 60 seconds of buffer
    return now_ms < (expires_at - 60_000)


def _refresh_oauth_token(creds: Dict[str, Any]) -> Optional[str]:
    """Attempt to refresh an expired Claude Code OAuth token.

    Uses the same token endpoint and client_id as Claude Code / OpenCode.
    Only works for credentials that have a refresh token (from claude /login
    or claude setup-token with OAuth flow).

    Returns the new access token, or None if refresh fails.
    """
    import urllib.parse
    import urllib.request

    refresh_token = creds.get("refreshToken", "")
    if not refresh_token:
        logger.debug("No refresh token available — cannot refresh")
        return None

    # Client ID used by Claude Code's OAuth flow
    CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
    }).encode()

    req = urllib.request.Request(
        "https://console.anthropic.com/v1/oauth/token",
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": f"claude-cli/{_CLAUDE_CODE_VERSION} (external, cli)",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            new_access = result.get("access_token", "")
            new_refresh = result.get("refresh_token", refresh_token)
            expires_in = result.get("expires_in", 3600)  # seconds

            if new_access:
                import time
                new_expires_ms = int(time.time() * 1000) + (expires_in * 1000)
                # Write refreshed credentials back to ~/.claude/.credentials.json
                _write_claude_code_credentials(new_access, new_refresh, new_expires_ms)
                logger.debug("Successfully refreshed Claude Code OAuth token")
                return new_access
    except Exception as e:
        logger.debug("Failed to refresh Claude Code token: %s", e)

    return None


def _write_claude_code_credentials(access_token: str, refresh_token: str, expires_at_ms: int) -> None:
    """Write refreshed credentials back to ~/.claude/.credentials.json."""
    cred_path = Path.home() / ".claude" / ".credentials.json"
    try:
        # Read existing file to preserve other fields
        existing = {}
        if cred_path.exists():
            existing = json.loads(cred_path.read_text(encoding="utf-8"))

        existing["claudeAiOauth"] = {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
        }

        cred_path.parent.mkdir(parents=True, exist_ok=True)
        cred_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        # Restrict permissions (credentials file)
        cred_path.chmod(0o600)
    except (OSError, IOError) as e:
        logger.debug("Failed to write refreshed credentials: %s", e)


def _resolve_claude_code_token_from_credentials(creds: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve a token from Claude Code credential files, refreshing if needed."""
    creds = creds or read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        logger.debug("Using Claude Code credentials (auto-detected)")
        return creds["accessToken"]
    if creds:
        logger.debug("Claude Code credentials expired — attempting refresh")
        refreshed = _refresh_oauth_token(creds)
        if refreshed:
            return refreshed
        logger.debug("Token refresh failed — re-run 'claude setup-token' to reauthenticate")
    return None


def _prefer_refreshable_claude_code_token(env_token: str, creds: Optional[Dict[str, Any]]) -> Optional[str]:
    """Prefer Claude Code creds when a persisted env OAuth token would shadow refresh.

    Hermes historically persisted setup tokens into ANTHROPIC_TOKEN. That makes
    later refresh impossible because the static env token wins before we ever
    inspect Claude Code's refreshable credential file. If we have a refreshable
    Claude Code credential record, prefer it over the static env OAuth token.
    """
    if not env_token or not _is_oauth_token(env_token) or not isinstance(creds, dict):
        return None
    if not creds.get("refreshToken"):
        return None

    resolved = _resolve_claude_code_token_from_credentials(creds)
    if resolved and resolved != env_token:
        logger.debug(
            "Preferring Claude Code credential file over static env OAuth token so refresh can proceed"
        )
        return resolved
    return None


def get_anthropic_token_source(token: Optional[str] = None) -> str:
    """Best-effort source classification for an Anthropic credential token."""
    token = (token or "").strip()
    if not token:
        return "none"

    env_token = os.getenv("ANTHROPIC_TOKEN", "").strip()
    if env_token and env_token == token:
        return "anthropic_token_env"

    cc_env_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if cc_env_token and cc_env_token == token:
        return "claude_code_oauth_token_env"

    creds = read_claude_code_credentials()
    if creds and creds.get("accessToken") == token:
        return str(creds.get("source") or "claude_code_credentials")

    managed_key = read_claude_managed_key()
    if managed_key and managed_key == token:
        return "claude_json_primary_api_key"

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key and api_key == token:
        return "anthropic_api_key_env"

    return "unknown"


def resolve_anthropic_token() -> Optional[str]:
    """Resolve an Anthropic token from all available sources.

    Priority:
      1. ANTHROPIC_TOKEN env var (OAuth/setup token saved by Hermes)
      2. CLAUDE_CODE_OAUTH_TOKEN env var
      3. Claude Code credentials (~/.claude.json or ~/.claude/.credentials.json)
         — with automatic refresh if expired and a refresh token is available
      4. ANTHROPIC_API_KEY env var (regular API key, or legacy fallback)

    Returns the token string or None.
    """
    creds = read_claude_code_credentials()

    # 1. Hermes-managed OAuth/setup token env var
    token = os.getenv("ANTHROPIC_TOKEN", "").strip()
    if token:
        preferred = _prefer_refreshable_claude_code_token(token, creds)
        if preferred:
            return preferred
        return token

    # 2. CLAUDE_CODE_OAUTH_TOKEN (used by Claude Code for setup-tokens)
    cc_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if cc_token:
        preferred = _prefer_refreshable_claude_code_token(cc_token, creds)
        if preferred:
            return preferred
        return cc_token

    # 3. Hermes-managed OAuth credentials (~/.hermes/.anthropic_oauth.json)
    hermes_creds = read_hermes_oauth_credentials()
    if hermes_creds:
        if is_claude_code_token_valid(hermes_creds):
            logger.debug("Using Hermes-managed OAuth credentials")
            return hermes_creds["accessToken"]
        # Expired — try refresh
        logger.debug("Hermes OAuth token expired — attempting refresh")
        refreshed = refresh_hermes_oauth_token()
        if refreshed:
            return refreshed

    # 4. Claude Code credential file
    resolved_claude_token = _resolve_claude_code_token_from_credentials(creds)
    if resolved_claude_token:
        return resolved_claude_token

    # 5. Regular API key, or a legacy OAuth token saved in ANTHROPIC_API_KEY.
    # This remains as a compatibility fallback for pre-migration Hermes configs.
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key

    return None


def run_oauth_setup_token() -> Optional[str]:
    """Run 'claude setup-token' interactively and return the resulting token.

    Checks multiple sources after the subprocess completes:
      1. Claude Code credential files (may be written by the subprocess)
      2. CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_TOKEN env vars

    Returns the token string, or None if no credentials were obtained.
    Raises FileNotFoundError if the 'claude' CLI is not installed.
    """
    import shutil
    import subprocess

    claude_path = shutil.which("claude")
    if not claude_path:
        raise FileNotFoundError(
            "The 'claude' CLI is not installed. "
            "Install it with: npm install -g @anthropic-ai/claude-code"
        )

    # Run interactively — stdin/stdout/stderr inherited so user can interact
    try:
        subprocess.run([claude_path, "setup-token"])
    except (KeyboardInterrupt, EOFError):
        return None

    # Check if credentials were saved to Claude Code's config files
    creds = read_claude_code_credentials()
    if creds and is_claude_code_token_valid(creds):
        return creds["accessToken"]

    # Check env vars that may have been set
    for env_var in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN"):
        val = os.getenv(env_var, "").strip()
        if val:
            return val

    return None


# ── Hermes-native PKCE OAuth flow ────────────────────────────────────────
# Mirrors the flow used by Claude Code, pi-ai, and OpenCode.
# Stores credentials in ~/.hermes/.anthropic_oauth.json (our own file).

_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
_OAUTH_SCOPES = "org:create_api_key user:profile user:inference"
_HERMES_OAUTH_FILE = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))) / ".anthropic_oauth.json"


def _generate_pkce() -> tuple:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    import base64
    import hashlib
    import secrets

    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def run_hermes_oauth_login() -> Optional[str]:
    """Run Hermes-native OAuth PKCE flow for Claude Pro/Max subscription.

    Opens a browser to claude.ai for authorization, prompts for the code,
    exchanges it for tokens, and stores them in ~/.hermes/.anthropic_oauth.json.

    Returns the access token on success, None on failure.
    """
    import time
    import webbrowser

    verifier, challenge = _generate_pkce()

    # Build authorization URL
    params = {
        "code": "true",
        "client_id": _OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _OAUTH_REDIRECT_URI,
        "scope": _OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    from urllib.parse import urlencode
    auth_url = f"https://claude.ai/oauth/authorize?{urlencode(params)}"

    print()
    print("Authorize Hermes with your Claude Pro/Max subscription.")
    print()
    print("╭─ Claude Pro/Max Authorization ────────────────────╮")
    print("│                                                   │")
    print("│  Open this link in your browser:                  │")
    print("╰───────────────────────────────────────────────────╯")
    print()
    print(f"  {auth_url}")
    print()

    # Try to open browser automatically (works on desktop, silently fails on headless/SSH)
    try:
        webbrowser.open(auth_url)
        print("  (Browser opened automatically)")
    except Exception:
        pass

    print()
    print("After authorizing, you'll see a code. Paste it below.")
    print()
    try:
        auth_code = input("Authorization code: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

    if not auth_code:
        print("No code entered.")
        return None

    # Split code#state format
    splits = auth_code.split("#")
    code = splits[0]
    state = splits[1] if len(splits) > 1 else ""

    # Exchange code for tokens
    try:
        import urllib.request
        exchange_data = json.dumps({
            "grant_type": "authorization_code",
            "client_id": _OAUTH_CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": _OAUTH_REDIRECT_URI,
            "code_verifier": verifier,
        }).encode()

        req = urllib.request.Request(
            _OAUTH_TOKEN_URL,
            data=exchange_data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"claude-cli/{_CLAUDE_CODE_VERSION} (external, cli)",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Token exchange failed: {e}")
        return None

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = result.get("expires_in", 3600)

    if not access_token:
        print("No access token in response.")
        return None

    # Store credentials
    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    _save_hermes_oauth_credentials(access_token, refresh_token, expires_at_ms)

    # Also write to Claude Code's credential file for backward compat
    _write_claude_code_credentials(access_token, refresh_token, expires_at_ms)

    print("Authentication successful!")
    return access_token


def _save_hermes_oauth_credentials(access_token: str, refresh_token: str, expires_at_ms: int) -> None:
    """Save OAuth credentials to ~/.hermes/.anthropic_oauth.json."""
    data = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at_ms,
    }
    try:
        _HERMES_OAUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HERMES_OAUTH_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _HERMES_OAUTH_FILE.chmod(0o600)
    except (OSError, IOError) as e:
        logger.debug("Failed to save Hermes OAuth credentials: %s", e)


def read_hermes_oauth_credentials() -> Optional[Dict[str, Any]]:
    """Read Hermes-managed OAuth credentials from ~/.hermes/.anthropic_oauth.json."""
    if _HERMES_OAUTH_FILE.exists():
        try:
            data = json.loads(_HERMES_OAUTH_FILE.read_text(encoding="utf-8"))
            if data.get("accessToken"):
                return data
        except (json.JSONDecodeError, OSError, IOError) as e:
            logger.debug("Failed to read Hermes OAuth credentials: %s", e)
    return None


def refresh_hermes_oauth_token() -> Optional[str]:
    """Refresh the Hermes-managed OAuth token using the stored refresh token.

    Returns the new access token, or None if refresh fails.
    """
    import time
    import urllib.request

    creds = read_hermes_oauth_credentials()
    if not creds or not creds.get("refreshToken"):
        return None

    try:
        data = json.dumps({
            "grant_type": "refresh_token",
            "refresh_token": creds["refreshToken"],
            "client_id": _OAUTH_CLIENT_ID,
        }).encode()

        req = urllib.request.Request(
            _OAUTH_TOKEN_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"claude-cli/{_CLAUDE_CODE_VERSION} (external, cli)",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())

        new_access = result.get("access_token", "")
        new_refresh = result.get("refresh_token", creds["refreshToken"])
        expires_in = result.get("expires_in", 3600)

        if new_access:
            new_expires_ms = int(time.time() * 1000) + (expires_in * 1000)
            _save_hermes_oauth_credentials(new_access, new_refresh, new_expires_ms)
            # Also update Claude Code's credential file
            _write_claude_code_credentials(new_access, new_refresh, new_expires_ms)
            logger.debug("Successfully refreshed Hermes OAuth token")
            return new_access
    except Exception as e:
        logger.debug("Failed to refresh Hermes OAuth token: %s", e)

    return None


# ---------------------------------------------------------------------------
# Message / tool / response format conversion
# ---------------------------------------------------------------------------


def normalize_model_name(model: str) -> str:
    """Normalize a model name for the Anthropic API.

    - Strips 'anthropic/' prefix (OpenRouter format, case-insensitive)
    - Converts dots to hyphens in version numbers (OpenRouter uses dots,
      Anthropic uses hyphens: claude-opus-4.6 → claude-opus-4-6)
    """
    lower = model.lower()
    if lower.startswith("anthropic/"):
        model = model[len("anthropic/"):]
    # OpenRouter uses dots for version separators (claude-opus-4.6),
    # Anthropic uses hyphens (claude-opus-4-6). Convert dots to hyphens.
    model = model.replace(".", "-")
    return model


def _sanitize_tool_id(tool_id: str) -> str:
    """Sanitize a tool call ID for the Anthropic API.

    Anthropic requires IDs matching [a-zA-Z0-9_-]. Replace invalid
    characters with underscores and ensure non-empty.
    """
    import re
    if not tool_id:
        return "tool_0"
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
    return sanitized or "tool_0"


def _convert_openai_image_part_to_anthropic(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert an OpenAI-style image block to Anthropic's image source format."""
    image_data = part.get("image_url", {})
    url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data)
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()

    if url.startswith("data:"):
        header, sep, data = url.partition(",")
        if sep and ";base64" in header:
            media_type = header[5:].split(";", 1)[0] or "image/png"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }

    if url.startswith("http://") or url.startswith("https://"):
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        }

    return None


def _convert_user_content_part_to_anthropic(part: Any) -> Optional[Dict[str, Any]]:
    if isinstance(part, dict):
        ptype = part.get("type")
        if ptype == "text":
            block = {"type": "text", "text": part.get("text", "")}
            if isinstance(part.get("cache_control"), dict):
                block["cache_control"] = dict(part["cache_control"])
            return block
        if ptype == "image_url":
            return _convert_openai_image_part_to_anthropic(part)
        if ptype == "image" and part.get("source"):
            return dict(part)
        if ptype == "image" and part.get("data"):
            media_type = part.get("mimeType") or part.get("media_type") or "image/png"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": part.get("data", ""),
                },
            }
        if ptype == "tool_result":
            return dict(part)
    elif part is not None:
        return {"type": "text", "text": str(part)}
    return None


def convert_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Anthropic format."""
    if not tools:
        return []
    result = []
    for t in tools:
        fn = t.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _image_source_from_openai_url(url: str) -> Dict[str, str]:
    """Convert an OpenAI-style image URL/data URL into Anthropic image source."""
    url = str(url or "").strip()
    if not url:
        return {"type": "url", "url": ""}

    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                media_type = mime_part
        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }

    return {"type": "url", "url": url}


def _convert_content_part_to_anthropic(part: Any) -> Optional[Dict[str, Any]]:
    """Convert a single OpenAI-style content part to Anthropic format."""
    if part is None:
        return None
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    ptype = part.get("type")

    if ptype == "input_text":
        block: Dict[str, Any] = {"type": "text", "text": part.get("text", "")}
    elif ptype in {"image_url", "input_image"}:
        image_value = part.get("image_url", {})
        url = image_value.get("url", "") if isinstance(image_value, dict) else str(image_value or "")
        block = {"type": "image", "source": _image_source_from_openai_url(url)}
    else:
        block = dict(part)

    if isinstance(part.get("cache_control"), dict) and "cache_control" not in block:
        block["cache_control"] = dict(part["cache_control"])
    return block


def _convert_content_to_anthropic(content: Any) -> Any:
    """Convert OpenAI-style multimodal content arrays to Anthropic blocks."""
    if not isinstance(content, list):
        return content

    converted = []
    for part in content:
        block = _convert_content_part_to_anthropic(part)
        if block is not None:
            converted.append(block)
    return converted


def convert_messages_to_anthropic(
    messages: List[Dict],
) -> Tuple[Optional[Any], List[Dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    System messages are extracted since Anthropic takes them as a separate param.
    system_prompt is a string or list of content blocks (when cache_control present).
    """
    system = None
    result = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            if isinstance(content, list):
                # Preserve cache_control markers on content blocks
                has_cache = any(
                    p.get("cache_control") for p in content if isinstance(p, dict)
                )
                if has_cache:
                    system = [p for p in content if isinstance(p, dict)]
                else:
                    system = "\n".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
            else:
                system = content
            continue

        if role == "assistant":
            blocks = []
            if content:
                if isinstance(content, list):
                    converted_content = _convert_content_to_anthropic(content)
                    if isinstance(converted_content, list):
                        blocks.extend(converted_content)
                else:
                    blocks.append({"type": "text", "text": str(content)})
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else args
                except (json.JSONDecodeError, ValueError):
                    parsed_args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": _sanitize_tool_id(tc.get("id", "")),
                    "name": fn.get("name", ""),
                    "input": parsed_args,
                })
            # Anthropic rejects empty assistant content
            effective = blocks or content
            if not effective or effective == "":
                effective = [{"type": "text", "text": "(empty)"}]
            result.append({"role": "assistant", "content": effective})
            continue

        if role == "tool":
            # Sanitize tool_use_id and ensure non-empty content
            result_content = content if isinstance(content, str) else json.dumps(content)
            if not result_content:
                result_content = "(no output)"
            tool_result = {
                "type": "tool_result",
                "tool_use_id": _sanitize_tool_id(m.get("tool_call_id", "")),
                "content": result_content,
            }
            if isinstance(m.get("cache_control"), dict):
                tool_result["cache_control"] = dict(m["cache_control"])
            # Merge consecutive tool results into one user message
            if (
                result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], list)
                and result[-1]["content"]
                and result[-1]["content"][0].get("type") == "tool_result"
            ):
                result[-1]["content"].append(tool_result)
            else:
                result.append({"role": "user", "content": [tool_result]})
            continue

        # Regular user message
        if isinstance(content, list):
            converted_blocks = _convert_content_to_anthropic(content)
            result.append({
                "role": "user",
                "content": converted_blocks or [{"type": "text", "text": ""}],
            })
        else:
            result.append({"role": "user", "content": content})

    # Strip orphaned tool_use blocks (no matching tool_result follows)
    tool_result_ids = set()
    for m in result:
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if block.get("type") == "tool_result":
                    tool_result_ids.add(block.get("tool_use_id"))
    for m in result:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            m["content"] = [
                b
                for b in m["content"]
                if b.get("type") != "tool_use" or b.get("id") in tool_result_ids
            ]
            if not m["content"]:
                m["content"] = [{"type": "text", "text": "(tool call removed)"}]

    # Enforce strict role alternation (Anthropic rejects consecutive same-role messages)
    fixed = []
    for m in result:
        if fixed and fixed[-1]["role"] == m["role"]:
            if m["role"] == "user":
                # Merge consecutive user messages
                prev_content = fixed[-1]["content"]
                curr_content = m["content"]
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    fixed[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    fixed[-1]["content"] = prev_content + curr_content
                else:
                    # Mixed types — wrap string in list
                    if isinstance(prev_content, str):
                        prev_content = [{"type": "text", "text": prev_content}]
                    if isinstance(curr_content, str):
                        curr_content = [{"type": "text", "text": curr_content}]
                    fixed[-1]["content"] = prev_content + curr_content
            else:
                # Consecutive assistant messages — merge text content
                prev_blocks = fixed[-1]["content"]
                curr_blocks = m["content"]
                if isinstance(prev_blocks, list) and isinstance(curr_blocks, list):
                    fixed[-1]["content"] = prev_blocks + curr_blocks
                elif isinstance(prev_blocks, str) and isinstance(curr_blocks, str):
                    fixed[-1]["content"] = prev_blocks + "\n" + curr_blocks
                else:
                    # Mixed types — normalize both to list and merge
                    if isinstance(prev_blocks, str):
                        prev_blocks = [{"type": "text", "text": prev_blocks}]
                    if isinstance(curr_blocks, str):
                        curr_blocks = [{"type": "text", "text": curr_blocks}]
                    fixed[-1]["content"] = prev_blocks + curr_blocks
        else:
            fixed.append(m)
    result = fixed

    return system, result


def build_anthropic_kwargs(
    model: str,
    messages: List[Dict],
    tools: Optional[List[Dict]],
    max_tokens: Optional[int],
    reasoning_config: Optional[Dict[str, Any]],
    tool_choice: Optional[str] = None,
    is_oauth: bool = False,
) -> Dict[str, Any]:
    """Build kwargs for anthropic.messages.create().

    When *is_oauth* is True, applies Claude Code compatibility transforms:
    system prompt prefix, tool name prefixing, and prompt sanitization.
    """
    system, anthropic_messages = convert_messages_to_anthropic(messages)
    anthropic_tools = convert_tools_to_anthropic(tools) if tools else []

    model = normalize_model_name(model)
    effective_max_tokens = max_tokens or 16384

    # ── OAuth: Claude Code identity ──────────────────────────────────
    if is_oauth:
        # 1. Prepend Claude Code system prompt identity
        cc_block = {"type": "text", "text": _CLAUDE_CODE_SYSTEM_PREFIX}
        if isinstance(system, list):
            system = [cc_block] + system
        elif isinstance(system, str) and system:
            system = [cc_block, {"type": "text", "text": system}]
        else:
            system = [cc_block]

        # 2. Sanitize system prompt — replace product name references
        #    to avoid Anthropic's server-side content filters.
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                text = text.replace("Hermes Agent", "Claude Code")
                text = text.replace("Hermes agent", "Claude Code")
                text = text.replace("hermes-agent", "claude-code")
                text = text.replace("Nous Research", "Anthropic")
                block["text"] = text

        # 3. Prefix tool names with mcp_ (Claude Code convention)
        if anthropic_tools:
            for tool in anthropic_tools:
                if "name" in tool:
                    tool["name"] = _MCP_TOOL_PREFIX + tool["name"]

        # 4. Prefix tool names in message history (tool_use and tool_result blocks)
        for msg in anthropic_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use" and "name" in block:
                            if not block["name"].startswith(_MCP_TOOL_PREFIX):
                                block["name"] = _MCP_TOOL_PREFIX + block["name"]
                        elif block.get("type") == "tool_result" and "tool_use_id" in block:
                            pass  # tool_result uses ID, not name

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": effective_max_tokens,
    }

    if system:
        kwargs["system"] = system

    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
        # Map OpenAI tool_choice to Anthropic format
        if tool_choice == "auto" or tool_choice is None:
            kwargs["tool_choice"] = {"type": "auto"}
        elif tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            pass  # Don't send tool_choice — Anthropic will use tools if needed
        elif isinstance(tool_choice, str):
            # Specific tool name
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

    # Map reasoning_config to Anthropic's thinking parameter.
    # Claude 4.6 models use adaptive thinking + output_config.effort.
    # Older models use manual thinking with budget_tokens.
    # Haiku models do NOT support extended thinking at all — skip entirely.
    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is not False and "haiku" not in model.lower():
            effort = str(reasoning_config.get("effort", "medium")).lower()
            budget = THINKING_BUDGET.get(effort, 8000)
            if _supports_adaptive_thinking(model):
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {
                    "effort": ADAPTIVE_EFFORT_MAP.get(effort, "medium")
                }
            else:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                # Anthropic requires temperature=1 when thinking is enabled on older models
                kwargs["temperature"] = 1
                kwargs["max_tokens"] = max(effective_max_tokens, budget + 4096)

    return kwargs


def normalize_anthropic_response(
    response,
    strip_tool_prefix: bool = False,
) -> Tuple[SimpleNamespace, str]:
    """Normalize Anthropic response to match the shape expected by AIAgent.

    Returns (assistant_message, finish_reason) where assistant_message has
    .content, .tool_calls, and .reasoning attributes.

    When *strip_tool_prefix* is True, removes the ``mcp_`` prefix that was
    added to tool names for OAuth Claude Code compatibility.
    """
    text_parts = []
    reasoning_parts = []
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            reasoning_parts.append(block.thinking)
        elif block.type == "tool_use":
            name = block.name
            if strip_tool_prefix and name.startswith(_MCP_TOOL_PREFIX):
                name = name[len(_MCP_TOOL_PREFIX):]
            tool_calls.append(
                SimpleNamespace(
                    id=block.id,
                    type="function",
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(block.input),
                    ),
                )
            )

    # Map Anthropic stop_reason to OpenAI finish_reason
    stop_reason_map = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    finish_reason = stop_reason_map.get(response.stop_reason, "stop")

    return (
        SimpleNamespace(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=None,
        ),
        finish_reason,
    )
