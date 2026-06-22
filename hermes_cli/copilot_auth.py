"""GitHub Copilot authentication utilities.

Implements the OAuth device code flow used by the Copilot CLI and handles
token validation/exchange for the Copilot API.

Token type support (per GitHub docs):
  gho_          OAuth token           ✓  (default via copilot login)
  github_pat_   Fine-grained PAT      ✓  (needs Copilot Requests permission)
  ghu_          GitHub App token      ✓  (via environment variable)
  ghp_          Classic PAT           ✗  NOT SUPPORTED

Credential search order (matching Copilot CLI behaviour):
  1. COPILOT_GITHUB_TOKEN env var
  2. GH_TOKEN env var
  3. GITHUB_TOKEN env var
  4. gh auth token  CLI fallback

Catalog discovery can optionally extend that path with the Copilot
credential pool and records skipped invalid sources for auditability.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# OAuth device code flow constants (same client ID as opencode/Copilot CLI)
COPILOT_OAUTH_CLIENT_ID = "Ov23li8tweQw6odWQebz"
# Token type prefixes
_CLASSIC_PAT_PREFIX = "ghp_"
_SUPPORTED_PREFIXES = ("gho_", "github_pat_", "ghu_")

# Env var search order (matches Copilot CLI)
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

# Polling constants
_DEVICE_CODE_POLL_INTERVAL = 5  # seconds
_DEVICE_CODE_POLL_SAFETY_MARGIN = 3  # seconds


@dataclass(frozen=True)
class CopilotIdentitySkip:
    """One invalid source encountered while resolving the Copilot identity."""

    source: str
    reason: str


@dataclass(frozen=True)
class CopilotIdentityAudit:
    """Structured Copilot identity resolution result."""

    token: str = ""
    source: str = ""
    source_kind: str = ""
    skipped_sources: tuple[CopilotIdentitySkip, ...] = field(default_factory=tuple)
    error: str = ""


def validate_copilot_token(token: str) -> tuple[bool, str]:
    """Validate that a token is usable with the Copilot API.

    Returns (valid, message).
    """
    token = token.strip()
    if not token:
        return False, "Empty token"

    if token.startswith(_CLASSIC_PAT_PREFIX):
        return False, (
            "Classic Personal Access Tokens (ghp_*) are not supported by the "
            "Copilot API. Use one of:\n"
            "  → `copilot login` or `hermes model` to authenticate via OAuth\n"
            "  → A fine-grained PAT (github_pat_*) with Copilot Requests permission\n"
            "  → `gh auth login` with the default device code flow (produces gho_* tokens)"
        )

    return True, "OK"


def resolve_copilot_identity_audit(
    *,
    include_credential_pool: bool = False,
    exchange_pool_tokens: bool = False,
) -> CopilotIdentityAudit:
    """Resolve the active Copilot identity and retain an audit trail.

    ``resolve_copilot_token()`` wraps this helper for the compatibility path.
    Discovery code can opt into the credential pool and exchange behavior
    with ``include_credential_pool`` and ``exchange_pool_tokens``.
    """
    skipped_sources: list[CopilotIdentitySkip] = []

    # 1. Check env vars in priority order.
    for env_var in COPILOT_ENV_VARS:
        val = os.getenv(env_var, "").strip()
        if not val:
            continue
        valid, msg = validate_copilot_token(val)
        if not valid:
            logger.warning(
                "Token from %s is not supported: %s", env_var, msg
            )
            skipped_sources.append(CopilotIdentitySkip(source=env_var, reason=msg))
            continue
        return CopilotIdentityAudit(
            token=val,
            source=env_var,
            source_kind="env",
            skipped_sources=tuple(skipped_sources),
        )

    # 2. Optionally inspect the Copilot credential pool before gh auth.
    if include_credential_pool:
        try:
            from hermes_cli.auth import read_credential_pool
        except Exception as exc:
            logger.debug("Copilot credential pool lookup unavailable: %s", exc)
        else:
            try:
                pool_entries = read_credential_pool("copilot")
            except Exception as exc:
                logger.debug("Copilot credential pool lookup failed: %s", exc)
                skipped_sources.append(
                    CopilotIdentitySkip(
                        source="credential_pool:copilot",
                        reason=f"Failed to read credential pool: {exc}",
                    )
                )
            else:
                for index, entry in enumerate(pool_entries):
                    entry_source = f"credential_pool:copilot[{index}]"
                    if not isinstance(entry, dict):
                        skipped_sources.append(
                            CopilotIdentitySkip(
                                source=entry_source,
                                reason="Non-dict credential pool entry",
                            )
                        )
                        continue

                    raw = str(entry.get("access_token") or "").strip()
                    if not raw:
                        skipped_sources.append(
                            CopilotIdentitySkip(
                                source=entry_source,
                                reason="Missing access_token",
                            )
                        )
                        continue

                    valid, msg = validate_copilot_token(raw)
                    if not valid:
                        skipped_sources.append(
                            CopilotIdentitySkip(source=entry_source, reason=msg)
                        )
                        continue

                    if exchange_pool_tokens:
                        try:
                            api_token, _expires_at = exchange_copilot_token(raw)
                        except Exception as exc:
                            skipped_sources.append(
                                CopilotIdentitySkip(
                                    source=entry_source,
                                    reason=f"Copilot token exchange failed: {exc}",
                                )
                            )
                            continue
                        if not api_token:
                            skipped_sources.append(
                                CopilotIdentitySkip(
                                    source=entry_source,
                                    reason="Copilot token exchange returned empty token",
                                )
                            )
                            continue
                        return CopilotIdentityAudit(
                            token=api_token,
                            source=entry_source,
                            source_kind="credential_pool",
                            skipped_sources=tuple(skipped_sources),
                        )

                    return CopilotIdentityAudit(
                        token=raw,
                        source=entry_source,
                        source_kind="credential_pool",
                        skipped_sources=tuple(skipped_sources),
                    )

    # 3. Fall back to gh auth token.
    token = _try_gh_cli_token()
    if token:
        valid, msg = validate_copilot_token(token)
        if not valid:
            return CopilotIdentityAudit(
                skipped_sources=tuple(skipped_sources),
                error=(
                    "Token from `gh auth token` is a classic PAT (ghp_*). "
                    f"{msg}"
                ),
            )
        return CopilotIdentityAudit(
            token=token,
            source="gh auth token",
            source_kind="gh_auth",
            skipped_sources=tuple(skipped_sources),
        )

    return CopilotIdentityAudit(skipped_sources=tuple(skipped_sources))


def resolve_copilot_token() -> tuple[str, str]:
    """Resolve a GitHub token suitable for Copilot API use.

    Returns (token, source) where source describes where the token came from.
    Raises ValueError if only a classic PAT is available from ``gh auth token``.
    """
    audit = resolve_copilot_identity_audit()
    if audit.error:
        raise ValueError(audit.error)
    return audit.token, audit.source


def _gh_cli_candidates() -> list[str]:
    """Return candidate ``gh`` binary paths, including common Homebrew installs."""
    candidates: list[str] = []

    resolved = shutil.which("gh")
    if resolved:
        candidates.append(resolved)

    for candidate in (
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        str(Path.home() / ".local" / "bin" / "gh"),
    ):
        if candidate in candidates:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            candidates.append(candidate)

    return candidates


def _try_gh_cli_token() -> Optional[str]:
    """Return a token from ``gh auth token`` when the GitHub CLI is available.

    When COPILOT_GH_HOST is set, passes ``--hostname`` so gh returns the
    correct host's token. When COPILOT_GH_USER is set, also passes ``--user``
    so multi-account setups resolve to the intended account regardless of
    which one is currently active. Also strips GITHUB_TOKEN / GH_TOKEN from
    the subprocess environment so ``gh`` reads from its own credential store
    (hosts.yml) instead of just echoing the env var back.
    """
    hostname = os.getenv("COPILOT_GH_HOST", "").strip()
    username = os.getenv("COPILOT_GH_USER", "").strip()

    # Build a clean env so gh doesn't short-circuit on GITHUB_TOKEN / GH_TOKEN
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}

    for gh_path in _gh_cli_candidates():
        cmd = [gh_path, "auth", "token"]
        if hostname:
            cmd += ["--hostname", hostname]
        if username:
            cmd += ["--user", username]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                env=clean_env,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("gh CLI token lookup failed (%s): %s", gh_path, exc)
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


# ─── OAuth Device Code Flow ────────────────────────────────────────────────

def copilot_device_code_login(
    *,
    host: str = "github.com",
    timeout_seconds: float = 300,
) -> Optional[str]:
    """Run the GitHub OAuth device code flow for Copilot.

    Prints instructions for the user, polls for completion, and returns
    the OAuth access token on success, or None on failure/cancellation.

    This replicates the flow used by opencode and the Copilot CLI.
    """
    import urllib.request
    import urllib.parse

    domain = host.rstrip("/")
    device_code_url = f"https://{domain}/login/device/code"
    access_token_url = f"https://{domain}/login/oauth/access_token"

    # Step 1: Request device code
    data = urllib.parse.urlencode({
        "client_id": COPILOT_OAUTH_CLIENT_ID,
        "scope": "read:user",
    }).encode()

    req = urllib.request.Request(
        device_code_url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": _copilot_user_agent(),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            device_data = json.loads(resp.read().decode())
    except Exception as exc:
        logger.error("Failed to initiate device authorization: %s", exc)
        print(f"  ✗ Failed to start device authorization: {exc}")
        return None

    verification_uri = device_data.get("verification_uri", "https://github.com/login/device")
    user_code = device_data.get("user_code", "")
    device_code = device_data.get("device_code", "")
    interval = max(device_data.get("interval", _DEVICE_CODE_POLL_INTERVAL), 1)

    if not device_code or not user_code:
        print("  ✗ GitHub did not return a device code.")
        return None

    # Step 2: Show instructions
    print()
    print(f"  Open this URL in your browser: {verification_uri}")
    print(f"  Enter this code: {user_code}")
    print()
    print("  Waiting for authorization...", end="", flush=True)

    # Step 3: Poll for completion
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        time.sleep(interval + _DEVICE_CODE_POLL_SAFETY_MARGIN)

        poll_data = urllib.parse.urlencode({
            "client_id": COPILOT_OAUTH_CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }).encode()

        poll_req = urllib.request.Request(
            access_token_url,
            data=poll_data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": _copilot_user_agent(),
            },
        )

        try:
            with urllib.request.urlopen(poll_req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception:
            print(".", end="", flush=True)
            continue

        if result.get("access_token"):
            print(" ✓")
            return result["access_token"]

        error = result.get("error", "")
        if error == "authorization_pending":
            print(".", end="", flush=True)
            continue
        elif error == "slow_down":
            # RFC 8628: add 5 seconds to polling interval
            server_interval = result.get("interval")
            if isinstance(server_interval, (int, float)) and server_interval > 0:
                interval = int(server_interval)
            else:
                interval += 5
            print(".", end="", flush=True)
            continue
        elif error == "expired_token":
            print()
            print("  ✗ Device code expired. Please try again.")
            return None
        elif error == "access_denied":
            print()
            print("  ✗ Authorization was denied.")
            return None
        elif error:
            print()
            print(f"  ✗ Authorization failed: {error}")
            return None

    print()
    print("  ✗ Timed out waiting for authorization.")
    return None


# ─── Copilot Token Exchange ────────────────────────────────────────────────

# Module-level cache for exchanged Copilot API tokens.
# Maps raw_token_fingerprint -> (api_token, expires_at_epoch).
_jwt_cache: dict[str, tuple[str, float]] = {}
_JWT_REFRESH_MARGIN_SECONDS = 120  # refresh 2 min before expiry

# Token exchange endpoint. We present our single Copilot CLI identity
# (the `copilot-developer-cli` integration + `_copilot_user_agent()`), the same
# one used on the inference path, so there is exactly one identity across every
# Copilot-facing request.
# NOTE: the exchange endpoint itself is no longer used by the official
# Copilot CLI for /chat/completions or /models (those accept the raw gh
# token as a Bearer credential directly. Kept for opt-in compatibility
# (HERMES_COPILOT_FORCE_EXCHANGE=1).
_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
# Shared TTL for all on-disk version caches (CLI version + API version).
_VERSION_CACHE_TTL = 24 * 60 * 60  # 24h

# X-GitHub-Api-Version sent on Copilot API calls. Sourced (in priority order)
# from the locally-installed `@github/copilot` npm bundle, which bakes it in
# as a constant and is updated whenever the user runs `npm i -g @github/copilot`.
# Fallback is the value shipped by @github/copilot @ 1.0.57 (today's date).
_COPILOT_API_VERSION_FALLBACK = "2026-06-01"
_COPILOT_API_VERSION_CACHE_PATH = (
    Path.home() / ".cache" / "hermes" / "copilot_api_version.json"
)

# Latest released @github/copilot CLI version, used for the User-Agent we present
# to api.githubcopilot.com (we identify as the official Copilot CLI, matching the
# `copilot-developer-cli` Copilot-Integration-Id). Sourced (in priority order)
# from the GitHub releases API (authoritative upstream) then the npm registry
# `latest` dist-tag (downstream mirror, can lag); cached on disk with the same
# TTL as the other version probes. Fallback is the value shipped at the time of
# writing (2026-06-19).
_COPILOT_CLI_VERSION_FALLBACK = "1.0.63"
_COPILOT_CLI_RELEASES_URL = "https://api.github.com/repos/github/copilot-cli/releases/latest"
_COPILOT_CLI_REGISTRY_URL = "https://registry.npmjs.org/@github%2Fcopilot/latest"
_COPILOT_CLI_VERSION_CACHE_PATH = (
    Path.home() / ".cache" / "hermes" / "copilot_cli_version.json"
)

# ─────────────────────────────────────────────────────────────────────────────
# Copilot-Integration-Id — THE lever that unlocks the premium model catalog.
#
# Sent on every Copilot API call. The integration-id (NOT the User-Agent, NOT
# any X-Copilot-Agent-Slug — both proven inert) is what the GitHub backend keys
# the visible model catalog + per-model limits off of.
#
# LIVE PROBE (2026-06-19, account e126380_magh, read-only GET /models, see
# hermes/probe/integration_id_sweep.py) compared every candidate:
#   copilot-developer-cli     → 33 models  ← WINNER (strict superset)
#   copilot-cli               → 32 models
#   copilot-developer-sandbox → 32 models
#   vscode-chat               → 32 models  (NOT the gemini-hider an older
#                               comment claimed — it shows gemini-3.x too, it
#                               just isn't the most complete integrator)
#   vscode-chat-dev           → 30 models  (also needs a Request-Hmac header)
#   copilot-4-cli             → 30 models  (limits the catalog the most)
# Only `copilot-developer-cli` exposes the full set (adds gpt-5.4-nano over the
# 32-model integrators) AND exposes gemini-3.1-pro-preview + gemini-3.5-flash +
# claude-opus-4.8 with the full reasoning-effort range (opus low..max). We use
# it uniformly so the whole codebase presents ONE integrator identity.
# Override via HERMES_COPILOT_INTEGRATION_ID for an account that needs a
# different one.
#
# ★ PREMIUM-TIER REQUIREMENT: the integration-id only unlocks the catalog when
# the request carries a VALID GitHub Bearer token. The token is resolved by
# resolve_copilot_token() from (in order) COPILOT_GITHUB_TOKEN / GH_TOKEN /
# GITHUB_TOKEN env vars or `gh auth token`, then passed through
# get_copilot_api_token() and injected as `Authorization: Bearer <token>` on the
# Copilot API call (github.com tokens are used directly; the legacy
# /copilot_internal/v2/token exchange is opt-in via HERMES_COPILOT_FORCE_EXCHANGE).
# Without that Bearer token the premium models are NOT served regardless of the
# integration-id.
_COPILOT_INTEGRATION_ID_DEFAULT = "copilot-developer-cli"


def _copilot_integration_id() -> str:
    """Return the Copilot-Integration-Id to send (env-overridable)."""
    override = os.getenv("HERMES_COPILOT_INTEGRATION_ID", "").strip()
    return override or _COPILOT_INTEGRATION_ID_DEFAULT


def _copilot_node_version() -> str:
    """Return the Node version string (``v``-prefixed) for the CLI User-Agent.

    The real ``@github/copilot`` CLI runs on Node and reports
    ``process.version`` (e.g. ``v22.22.3``) in the parenthetical UA segment.
    We resolve a REAL node version from the box (via ``node --version``) so the
    value is authentic rather than fabricated — if a real Copilot CLI were
    installed here it would report the same runtime. Resolution order:
      1. ``HERMES_COPILOT_NODE_VERSION`` env override.
      2. ``node --version`` on PATH (cached in-process).
      3. Empty string → caller falls back to the short UA form.
    """
    override = os.getenv("HERMES_COPILOT_NODE_VERSION", "").strip()
    if override:
        return override if override.startswith("v") else f"v{override}"

    global _copilot_node_version_memo
    try:
        if _copilot_node_version_memo is not None:
            return _copilot_node_version_memo
    except NameError:  # pragma: no cover - module-load ordering guard
        pass

    ver = ""
    node_path = shutil.which("node")
    if node_path:
        try:
            out = subprocess.run(
                [node_path, "--version"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            cand = (out.stdout or "").strip()
            if cand.startswith("v"):
                ver = cand
        except Exception as exc:
            logger.debug("node --version probe failed: %s", exc)

    _copilot_node_version_memo = ver
    return ver


# Node-version of the platform that the real CLI's process.version reports.
_copilot_node_version_memo: Optional[str] = None

# Map Python's sys.platform to Node's process.platform tokens (the CLI builds
# the UA from process.platform: linux/darwin/win32, NOT Python's "win32"-only
# overlap — they happen to agree for the common three).
_NODE_PLATFORM_MAP = {
    "linux": "linux",
    "darwin": "darwin",
    "win32": "win32",
}

# Default TERM_PROGRAM to present when the environment has none set. The real
# CLI's builder falls back to the literal "unknown", but that reads as a
# non-interactive/bot signal; a genuine Copilot CLI user is almost always inside
# a real terminal emulator. "vscode" is the most common, valid host for the
# Copilot CLI (a GitHub/Microsoft tool) and is coherent with the
# copilot-developer-cli identity (CLI running in the VS Code integrated
# terminal). Override via HERMES_COPILOT_TERM_PROGRAM.
_COPILOT_TERM_PROGRAM_DEFAULT = "vscode"


def _copilot_term_program() -> str:
    """Return the ``TERM_PROGRAM`` token for the CLI User-Agent.

    Resolution order:
      1. ``HERMES_COPILOT_TERM_PROGRAM`` env override.
      2. A REAL ``TERM_PROGRAM`` present in the environment (most authentic —
         e.g. ``vscode``, ``iTerm.app``, ``Apple_Terminal``, ``WezTerm``).
      3. ``_COPILOT_TERM_PROGRAM_DEFAULT`` (``vscode``) — a valid, common value,
         never the bot-signalling ``unknown``.
    """
    override = os.getenv("HERMES_COPILOT_TERM_PROGRAM", "").strip()
    if override:
        return override
    real = os.environ.get("TERM_PROGRAM", "").strip()
    if real:
        return real
    return _COPILOT_TERM_PROGRAM_DEFAULT


def _copilot_user_agent() -> str:
    """User-Agent presented to api.githubcopilot.com.

    We identify as the official ``@github/copilot`` CLI, reproducing its real
    UA builder (the bundle's ``FG()`` helper, RE 2026-06-19):

        ``copilot/<ver> (<platform> <node-version>) term/<TERM_PROGRAM>``

    where ``<platform>`` is Node's ``process.platform`` (linux/darwin/win32),
    ``<node-version>`` is the ``v``-prefixed Node ``process.version``, and
    ``<TERM_PROGRAM>`` identifies the host terminal. We source a REAL node
    version + platform from this box so the value is authentic (the CLI
    installed here would report the same), not fabricated. If node cannot be
    resolved we degrade to the honest short core ``copilot/<ver>`` rather than
    invent a runtime. ``TERM_PROGRAM`` uses a real environment value when set,
    else a valid default (``vscode``) rather than the CLI's literal ``unknown``
    fallback (which reads as a non-interactive/bot signal).

    A 2026-06-19 live probe proved the User-Agent does NOT affect the /models
    catalog (every UA value, including none, returned the same 33 models) — it
    is cosmetic for unlock; we send the faithful CLI value for identity
    consistency, not capability. Version is env-overridable via
    HERMES_COPILOT_CLI_VERSION; node version via HERMES_COPILOT_NODE_VERSION;
    terminal via HERMES_COPILOT_TERM_PROGRAM.
    """
    ver = _latest_copilot_cli_version()
    node_ver = _copilot_node_version()
    if not node_ver:
        # No authentic Node runtime to report — send the honest short form.
        return f"copilot/{ver}"
    platform = _NODE_PLATFORM_MAP.get(sys.platform, sys.platform)
    term = _copilot_term_program()
    return f"copilot/{ver} ({platform} {node_ver}) term/{term}"

# Candidate paths for the @github/copilot CLI bundle (global npm install).
_COPILOT_CLI_BUNDLE_CANDIDATES = (
    "/usr/local/lib/node_modules/@github/copilot/sdk/index.js",
    "/usr/lib/node_modules/@github/copilot/sdk/index.js",
)

# In-process caches so we don't hit disk on every header build.
_copilot_api_version_memo: tuple[str, float] | None = None
_copilot_cli_version_memo: tuple[str, float] | None = None


def _latest_copilot_cli_version() -> str:
    """Return the latest released ``@github/copilot`` CLI version.

    Used to build the ``copilot/<ver>`` User-Agent. Resolution order:
      1. ``HERMES_COPILOT_CLI_VERSION`` env override.
      2. In-process memo (TTL ``_VERSION_CACHE_TTL``).
      3. On-disk cache at ``_COPILOT_CLI_VERSION_CACHE_PATH``.
      4. npm registry ``latest`` dist-tag for ``@github/copilot``.
      5. Hard fallback ``_COPILOT_CLI_VERSION_FALLBACK``.
    """
    override = os.getenv("HERMES_COPILOT_CLI_VERSION", "").strip()
    if override:
        return override

    global _copilot_cli_version_memo
    now = time.time()
    if (
        _copilot_cli_version_memo
        and now - _copilot_cli_version_memo[1] < _VERSION_CACHE_TTL
    ):
        return _copilot_cli_version_memo[0]

    cache_path = _COPILOT_CLI_VERSION_CACHE_PATH
    try:
        if cache_path.is_file():
            data = json.loads(cache_path.read_text())
            ver = str(data.get("version") or "").lstrip("v").strip()
            ts = float(data.get("fetched_at") or 0)
            if ver and now - ts < _VERSION_CACHE_TTL:
                _copilot_cli_version_memo = (ver, ts)
                return ver
    except Exception as exc:
        logger.debug("copilot-cli version cache read failed: %s", exc)

    ver = _COPILOT_CLI_VERSION_FALLBACK
    try:
        import urllib.request

        latest = ""
        # 1) GitHub releases API (authoritative upstream). tag_name like "v1.0.63".
        try:
            req = urllib.request.Request(
                _COPILOT_CLI_RELEASES_URL,
                headers={"Accept": "application/vnd.github+json", "User-Agent": "hermes-agent"},
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                payload = json.loads(resp.read().decode())
            latest = str(payload.get("tag_name") or "").lstrip("v").strip()
        except Exception as exc:
            logger.debug("copilot-cli GitHub releases fetch failed: %s", exc)

        # 2) npm registry `latest` dist-tag (downstream mirror) if GH didn't answer.
        if not latest:
            req = urllib.request.Request(
                _COPILOT_CLI_REGISTRY_URL,
                headers={"Accept": "application/json", "User-Agent": "hermes-agent"},
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                payload = json.loads(resp.read().decode())
            latest = str(payload.get("version") or "").lstrip("v").strip()

        if latest:
            ver = latest
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"version": ver, "fetched_at": now})
                )
            except Exception as exc:
                logger.debug("copilot-cli version cache write failed: %s", exc)
    except Exception as exc:
        logger.debug(
            "failed to fetch latest copilot-cli version, using fallback %s: %s",
            _COPILOT_CLI_VERSION_FALLBACK,
            exc,
        )

    _copilot_cli_version_memo = (ver, now)
    return ver


def _discover_copilot_cli_bundles() -> list[Path]:
    """Locate plausible ``@github/copilot/sdk/index.js`` paths on this host.

    Walks common global-npm locations (incl. nvm). Returns existing files only.
    """
    seen: set[Path] = set()
    out: list[Path] = []

    def _add(p: Path) -> None:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen or not rp.is_file():
            return
        seen.add(rp)
        out.append(rp)

    # Static candidates.
    for s in _COPILOT_CLI_BUNDLE_CANDIDATES:
        _add(Path(s))

    # nvm-managed installs: ~/.nvm/versions/node/*/lib/node_modules/@github/copilot/sdk/index.js
    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    if nvm_root.is_dir():
        try:
            for node_dir in nvm_root.iterdir():
                _add(node_dir / "lib" / "node_modules" / "@github" / "copilot" / "sdk" / "index.js")
        except Exception as exc:
            logger.debug("nvm scan failed: %s", exc)

    # User-local global install (npm prefix override).
    _add(Path.home() / ".npm-global" / "lib" / "node_modules" / "@github" / "copilot" / "sdk" / "index.js")

    return out


def _extract_api_version_from_bundle(bundle: Path) -> str | None:
    """Grep the Copilot CLI bundle for the X-GitHub-Api-Version constant.

    The bundle defines it as e.g. ``Mss="X-GitHub-Api-Version",Oss="2026-06-01"``.
    We extract every adjacent date literal, drop the github.com REST date
    ``2022-11-28`` (used only for gist/asset uploads), and return the newest.
    """
    import re
    try:
        text = bundle.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.debug("copilot CLI bundle read failed (%s): %s", bundle, exc)
        return None
    matches = re.findall(
        r'"X-GitHub-Api-Version"\s*,\s*[A-Za-z0-9_$]+\s*=\s*"(\d{4}-\d{2}-\d{2})"',
        text,
    )
    # Filter out the github.com REST API version (different surface).
    candidates = sorted({m for m in matches if m != "2022-11-28"}, reverse=True)
    return candidates[0] if candidates else None


def _latest_copilot_api_version() -> str:
    """Return the X-GitHub-Api-Version value used by the Copilot API.

    Resolution order:
      1. ``HERMES_COPILOT_API_VERSION`` env override.
      2. In-process memo (TTL ``_VERSION_CACHE_TTL``).
      3. On-disk cache at ``_COPILOT_API_VERSION_CACHE_PATH``.
      4. Local ``@github/copilot`` npm bundle (the live source of truth,
         updates whenever the user runs ``npm i -g @github/copilot``).
      5. Hard fallback ``_COPILOT_API_VERSION_FALLBACK``.
    """
    override = os.getenv("HERMES_COPILOT_API_VERSION", "").strip()
    if override:
        return override

    global _copilot_api_version_memo
    now = time.time()
    if (
        _copilot_api_version_memo
        and now - _copilot_api_version_memo[1] < _VERSION_CACHE_TTL
    ):
        return _copilot_api_version_memo[0]

    cache_path = _COPILOT_API_VERSION_CACHE_PATH
    try:
        if cache_path.is_file():
            data = json.loads(cache_path.read_text())
            ver = str(data.get("version") or "").strip()
            ts = float(data.get("fetched_at") or 0)
            if ver and now - ts < _VERSION_CACHE_TTL:
                _copilot_api_version_memo = (ver, ts)
                return ver
    except Exception as exc:
        logger.debug("copilot api-version cache read failed: %s", exc)

    ver = _COPILOT_API_VERSION_FALLBACK
    for bundle in _discover_copilot_cli_bundles():
        extracted = _extract_api_version_from_bundle(bundle)
        if extracted:
            ver = extracted
            logger.debug("copilot api-version %s from %s", ver, bundle)
            break
    else:
        logger.debug(
            "no @github/copilot bundle found, using fallback api-version %s",
            _COPILOT_API_VERSION_FALLBACK,
        )

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"version": ver, "fetched_at": now}))
    except Exception as exc:
        logger.debug("copilot api-version cache write failed: %s", exc)

    _copilot_api_version_memo = (ver, now)
    return ver


def _token_fingerprint(raw_token: str) -> str:
    """Short fingerprint of a raw token for cache keying (avoids storing full token)."""
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()[:16]


def exchange_copilot_token(raw_token: str, *, timeout: float = 10.0) -> tuple[str, float]:
    """Exchange a raw GitHub token for a short-lived Copilot API token.

    Calls ``GET https://api.github.com/copilot_internal/v2/token`` with
    the raw GitHub token and returns ``(api_token, expires_at)``.

    The returned token is a semicolon-separated string (not a standard JWT)
    used as ``Authorization: Bearer <token>`` for Copilot API requests.

    Results are cached in-process and reused until close to expiry.
    Raises ``ValueError`` on failure.
    """
    import urllib.request

    fp = _token_fingerprint(raw_token)

    # Check cache first
    cached = _jwt_cache.get(fp)
    if cached:
        api_token, expires_at = cached
        if time.time() < expires_at - _JWT_REFRESH_MARGIN_SECONDS:
            return api_token, expires_at

    req = urllib.request.Request(
        _TOKEN_EXCHANGE_URL,
        method="GET",
        headers={
            "Authorization": f"Bearer {raw_token}",
            "User-Agent": _copilot_user_agent(),
            "Accept": "application/json",
            "Copilot-Integration-Id": _copilot_integration_id(),
            "X-GitHub-Api-Version": _latest_copilot_api_version(),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        raise ValueError(f"Copilot token exchange failed: {exc}") from exc

    api_token = data.get("token", "")
    expires_at = data.get("expires_at", 0)
    if not api_token:
        raise ValueError("Copilot token exchange returned empty token")

    # Convert expires_at to float if needed
    expires_at = float(expires_at) if expires_at else time.time() + 1800

    _jwt_cache[fp] = (api_token, expires_at)
    logger.debug(
        "Copilot token exchanged, expires_at=%s",
        expires_at,
    )
    return api_token, expires_at


def get_copilot_api_token(raw_token: str) -> str:
    """Return the API token to use against ``api.githubcopilot.com``.

    The Copilot API accepts the raw GitHub OAuth/PAT token directly as
    ``Authorization: Bearer ***``; no exchange step is required.
    This was verified against the official ``@github/copilot`` CLI bundle:
    its SDK calls Copilot endpoints with ``Bearer <gh-token>`` directly.

    The legacy ``GET /copilot_internal/v2/token`` exchange endpoint on
    ``api.github.com`` is not used by the CLI and now returns 404 (the
    REST router treats ``/copilot_internal/...`` as a repo sub-path).

    Set ``HERMES_COPILOT_FORCE_EXCHANGE=1`` to opt in to the legacy
    exchange flow (will fall back to the raw token on failure).
    """
    if not raw_token:
        return raw_token
    if os.getenv("HERMES_COPILOT_FORCE_EXCHANGE", "").strip() in ("1", "true", "yes"):
        try:
            api_token, _ = exchange_copilot_token(raw_token)
            return api_token
        except Exception as exc:
            logger.debug("Copilot token exchange failed, using raw token: %s", exc)
    return raw_token


# ─── Copilot API Headers ───────────────────────────────────────────────────

def copilot_request_headers(
    *,
    is_agent_turn: bool = True,
    is_vision: bool = False,
    model: str = "",
    intent: str = "conversation-panel",
    interaction_id: Optional[str] = None,
) -> dict[str, str]:
    """Build the standard headers for Copilot API requests.

    Presents as the official ``@github/copilot`` CLI (matching the
    ``copilot-developer-cli`` Copilot-Integration-Id), NOT the VS Code Chat
    extension. Verified against the real CLI bundle (``@github/copilot`` 1.0.63,
    its ``que()`` inference-header builder, RE 2026-06-19): the CLI sends
    ``Copilot-Integration-Id`` + ``Authorization: Bearer`` + ``Runtime-Client-Version``
    and does NOT send the ``Editor-Version`` / ``Editor-Plugin-Version`` pair
    (those are VS Code Chat extension headers). We follow the CLI shape so the
    whole identity (integration-id + UA + headers) is internally consistent.
    """
    import uuid as _uuid
    headers: dict[str, str] = {
        "User-Agent": _copilot_user_agent(),
        "Copilot-Integration-Id": _copilot_integration_id(),
        # The real CLI sends this in place of the VS Code Editor-* pair; value
        # is the @github/copilot CLI version we're identifying as.
        "Runtime-Client-Version": _latest_copilot_cli_version(),
        "Openai-Intent": intent,
        # Mirror of Openai-Intent (extension sends both unless overridden).
        "X-Interaction-Type": intent,
        # Inference-path Copilot API version (currently 2026-06-01).
        "X-GitHub-Api-Version": _latest_copilot_api_version(),
        "x-initiator": "agent" if is_agent_turn else "user",
        # Per-call request id + stable per-session interaction id (server uses
        # them for trace/log correlation and may key some quotas off
        # X-Interaction-Id).
        "X-Request-Id": str(_uuid.uuid4()),
        "X-Interaction-Id": interaction_id or str(_uuid.uuid4()),
    }

    # NOTE: hermes previously injected `X-Copilot-Agent-Slug: copilot-1m-context`
    # here, believing it mapped the token to the developer-app integrator and
    # unlocked 1M context / Gemini-3.x. Live probing (2026-06-07) proved that
    # slug is INERT: it changes neither catalog visibility nor per-model limits.
    # What actually exposes gemini-3.x and the full limits is the
    # Copilot-Integration-Id (`copilot-developer-cli`, matching the official CLI). The
    # slug was removed to avoid sending a misleading no-op header. The official
    # @github/copilot CLI sends `copilot-developer-sandbox` only on specific
    # (non-inference) endpoints; we don't need it for chat/messages/responses.

    if is_vision:
        headers["Copilot-Vision-Request"] = "true"

    return headers
