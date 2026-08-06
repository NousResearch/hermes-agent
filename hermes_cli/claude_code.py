"""Claude subscription provider (official Claude Agent SDK) — identity + auth surface.

This module owns everything core Hermes needs to know about the ``claude-code``
provider *except* the inference runtime (which routes through the official
``claude-agent-sdk``).

**Credential boundary — the whole point of this provider.**  Hermes never
reads, writes, refreshes, or deletes Claude credentials.  There is no
authorize URL, no client id, no PKCE, no token exchange, no token storage, and
no credential-file inspection anywhere in this module.  The user runs
``claude auth login`` themselves and the SDK resolves credentials on its own;
Hermes only asks the official CLI *whether* a login exists so the UI can say
so.

Two constraints that look like details and are not:

* Config isolation, when a caller needs it, goes through ``CLAUDE_CONFIG_DIR``
  — never by overriding ``HOME``.  Overriding ``HOME`` also relocates the
  macOS login-keychain lookup (``$HOME/Library/Keychains``), so the spawned
  CLI cannot find its stored credentials and reports "Not logged in".
* Every probe spawns with ``stdin`` set to ``DEVNULL``.  Claude Code probes
  stdin and blocks indefinitely when it inherits an unusable parent stdin
  (e.g. from a GUI parent); ``--print`` does not override this.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Identity ────────────────────────────────────────────────────────────────

CLAUDE_CODE_PROVIDER_ID = "claude-code"
CLAUDE_CODE_DISPLAY_NAME = "Claude subscription (Agent SDK)"
CLAUDE_CODE_DESCRIPTION = (
    "Claude subscription (Agent SDK) — uses your Claude Pro/Max/Team/Enterprise "
    "plan through the official Claude Agent SDK; run `claude auth login` first"
)
CLAUDE_CODE_API_MODE = "claude_agent_sdk"

# Internal scheme, mirroring ``acp://copilot``.  Deliberately not an HTTP URL:
# nothing in Hermes may treat this provider as a reachable REST endpoint.
CLAUDE_CODE_BASE_URL = "claude-sdk://subscription"

# Curated default when no model is configured for this provider. Every
# surface that can hand the runtime an empty model (TUI startup resolution,
# bare provider switches) falls back to this rather than letting the CLI's
# own default silently pick the model.
DEFAULT_SUBSCRIPTION_MODEL = "claude-sonnet-5"

CLAUDE_BINARY = "claude"
CLAUDE_LOGIN_COMMAND = "claude auth login"
CLAUDE_LOGOUT_COMMAND = "claude auth logout"
CLAUDE_DOCS_URL = "https://docs.claude.com/en/docs/claude-code"

# Slugs that meant "anthropic, authenticated with a Claude Code OAuth token"
# before this provider existed.  They keep resolving to ``anthropic`` while the
# subscription gate is closed so existing configs are not silently repointed at
# a different billing source; see :func:`legacy_alias_target`.
LEGACY_ANTHROPIC_ALIASES = ("claude-code", "claude-oauth")

# CodeMux's probe budget: 5s per subprocess, 10s for the whole status call.
_PROBE_TIMEOUT_SECONDS = 5.0
_STATUS_TIMEOUT_SECONDS = 10.0


# ── Config gate (PR1's ``claude_subscription.enabled``) ─────────────────────

_gate_cache: Optional[bool] = None


def subscription_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True when the Claude subscription provider is turned on.

    Thin, never-raising wrapper around
    ``hermes_cli.claude_subscription.claude_subscription_enabled``.  The result
    is cached per process: the gate decides module-import-time membership in
    ``CANONICAL_PROVIDERS``, so re-reading it mid-process could only produce a
    universe that disagrees with itself.
    """
    global _gate_cache
    if config is None and _gate_cache is not None:
        return _gate_cache
    try:
        from hermes_cli.claude_subscription import claude_subscription_enabled
    except ImportError:
        # The gate module ships alongside this provider; treat a missing gate
        # as "off" so a partial checkout degrades to today's behavior.
        if config is None:
            _gate_cache = False
        return False
    resolved = config
    if resolved is None:
        # claude_subscription_enabled() reads a dict and treats None as off, so
        # a no-arg call has to load the config itself or the gate is pinned
        # False no matter what config.yaml says. Read-only: this runs from the
        # provider catalog and alias resolution, which must not write config.
        try:
            from hermes_cli.config import load_config_readonly

            resolved = load_config_readonly()
        except Exception as exc:
            logger.debug("load_config_readonly() failed for the gate: %s", exc)
            resolved = None
    try:
        enabled = bool(claude_subscription_enabled(resolved))
    except Exception as exc:
        logger.debug("claude_subscription_enabled() failed: %s", exc)
        enabled = False
    if config is None:
        _gate_cache = enabled
    return enabled


def reset_subscription_gate_cache() -> None:
    """Drop the cached gate value (tests and config-mutating flows)."""
    global _gate_cache
    _gate_cache = None


def legacy_alias_target(slug: str) -> Optional[str]:
    """Return ``"anthropic"`` when *slug* is a pre-SDK Claude alias to keep.

    ``claude-code`` used to be an alias of ``anthropic``.  It is now a provider
    in its own right, but only once the user opts in — until then the old
    mapping stays live so an existing ``provider: claude-code`` config keeps
    resolving instead of erroring out.
    """
    if (slug or "").strip().lower() not in LEGACY_ANTHROPIC_ALIASES:
        return None
    return None if subscription_enabled() else "anthropic"


# ── Official CLI probes ─────────────────────────────────────────────────────


def resolve_claude_binary() -> Optional[str]:
    """Absolute path to the ``claude`` CLI, or None when none is available.

    PATH first (a user-managed install is newer or deliberately pinned), then
    the executable bundled inside the ``claude-agent-sdk`` wheel. The bundled
    fallback matters for exactly the install this provider documents:
    ``pip install 'hermes-agent[claude-code]'`` on a machine that never
    installed Claude Code separately. The SDK spawns its own bundled CLI in
    that layout, so the auth probe must find the same binary or preflight
    refuses a setup that would actually work.
    """
    try:
        found = shutil.which(CLAUDE_BINARY)
        if found:
            return found
    except Exception:
        pass
    try:
        import importlib.util

        spec = importlib.util.find_spec("claude_agent_sdk")
        if spec and spec.origin:
            bundled = os.path.join(
                os.path.dirname(spec.origin), "_bundled", CLAUDE_BINARY
            )
            if os.path.isfile(bundled) and os.access(bundled, os.X_OK):
                return bundled
    except Exception:
        pass
    return None


def _run_claude(
    binary: str, args: list[str], timeout: float
) -> tuple[Optional[int], str, str]:
    """Run ``claude <args>``; return ``(returncode, stdout, stderr)``.

    ``returncode`` is None when the call timed out or could not be spawned.
    Never raises.
    """
    try:
        proc = subprocess.run(
            [binary, *args],
            # Claude Code probes stdin and hangs forever on an unusable
            # inherited stdin (GUI parent, closed fd). DEVNULL is required.
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.debug("claude %s timed out after %ss", " ".join(args), timeout)
        return None, "", "timed out"
    except Exception as exc:
        logger.debug("claude %s failed to spawn: %s", " ".join(args), exc)
        return None, "", str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _parse_auth_status(returncode: Optional[int], stdout: str, stderr: str) -> Dict[str, Any]:
    """Map a ``claude auth status`` result onto Hermes' status fields.

    ``claude auth status`` emits JSON (``{"loggedIn": true, "authMethod":
    "claude.ai", ...}``) and exits 0 when logged in, 1 when not.  JSON is the
    target; the documented text patterns are the fallback for older CLIs whose
    output is prose.
    """
    if returncode is None:
        return {
            "status": "unknown",
            "logged_in": False,
            "auth_method": "",
            "account": "",
            "organization": "",
            "subscription_type": "",
            "subscription": False,
            "message": f"Could not read `claude auth status` ({stderr.strip() or 'no output'}).",
        }

    payload: Any = None
    text = (stdout or "").strip()
    if text:
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            payload = None

    if isinstance(payload, dict):
        logged_in = bool(payload.get("loggedIn"))
        auth_method = str(payload.get("authMethod") or "").strip()
        info: Dict[str, Any] = {
            "status": "logged_in" if logged_in else "logged_out",
            "logged_in": logged_in,
            "auth_method": auth_method,
            "account": str(payload.get("email") or "").strip(),
            "organization": str(payload.get("orgName") or "").strip(),
            "subscription_type": str(payload.get("subscriptionType") or "").strip(),
        }
        info["subscription"] = logged_in and auth_method != "api-key"
        info["message"] = _status_message(info)
        return info

    # Fallback: prose output. Exit code is the authoritative signal.
    lowered = (text + " " + (stderr or "")).lower()
    logged_in = returncode == 0 and "not logged in" not in lowered
    auth_method = ""
    if "api key" in lowered or "api-key" in lowered:
        auth_method = "api-key"
    elif logged_in:
        auth_method = "claude.ai"
    info = {
        "status": "logged_in" if logged_in else "logged_out",
        "logged_in": logged_in,
        "auth_method": auth_method,
        "account": "",
        "organization": "",
        "subscription_type": "",
        "subscription": logged_in and auth_method != "api-key",
    }
    info["message"] = _status_message(info)
    return info


def _status_message(info: Dict[str, Any]) -> str:
    """Human-readable one-liner for a parsed auth status."""
    if not info.get("logged_in"):
        return f"Not signed in to Claude — run `{CLAUDE_LOGIN_COMMAND}`."
    if info.get("auth_method") == "api-key":
        # Load-bearing distinction: an API-key login is metered API billing,
        # not the user's Claude plan. Saying "subscription" here would be a lie.
        return (
            "Signed in with an Anthropic API key, not a Claude plan — requests are "
            f"billed as API usage. Run `{CLAUDE_LOGIN_COMMAND}` to use your subscription."
        )
    plan = info.get("subscription_type") or ""
    account = info.get("account") or ""
    detail = " ".join(p for p in (f"({plan})" if plan else "", account) if p).strip()
    return f"Signed in to Claude{' — ' + detail if detail else ''}."


_PROBE_CACHE_TTL_SECONDS = 45.0
_probe_cache: Optional[Dict[str, Any]] = None
_probe_cache_at: float = 0.0


def probe_claude_auth_cached(ttl: float = _PROBE_CACHE_TTL_SECONDS) -> Dict[str, Any]:
    """`probe_claude_auth`, memoized for *ttl* seconds.

    The probe spawns two subprocesses (`auth status` + `--version`, up to 5s
    each). The per-turn preflight and the provider picker both consult it, so
    without a cache an interactive session pays that cost on every turn and
    every picker open. Login state changes on human timescales; a short TTL
    keeps the answer honest without the spawn cost.
    """
    global _probe_cache, _probe_cache_at
    now = time.monotonic()
    if _probe_cache is not None and (now - _probe_cache_at) < ttl:
        return _probe_cache
    result = probe_claude_auth()
    _probe_cache, _probe_cache_at = result, now
    return result


def reset_probe_cache() -> None:
    """Drop the memoized auth probe (tests and login/logout flows)."""
    global _probe_cache, _probe_cache_at
    _probe_cache, _probe_cache_at = None, 0.0


def probe_claude_auth() -> Dict[str, Any]:
    """Ask the official CLI whether a Claude login exists.

    Reads no credential file and holds no token.  Never raises — an
    uninstalled CLI, a timeout, or malformed output all degrade to a status
    dict carrying an actionable ``message``.
    """
    resolved = resolve_claude_binary()
    if not resolved:
        return {
            "status": "cli_missing",
            "logged_in": False,
            "subscription": False,
            "auth_method": "",
            "account": "",
            "organization": "",
            "subscription_type": "",
            "cli_version": "",
            "command": CLAUDE_BINARY,
            "resolved_command": None,
            "message": (
                "Claude Code is not installed — install it, then run "
                f"`{CLAUDE_LOGIN_COMMAND}`."
            ),
        }

    # Per-probe cap, plus an overall cap so two slow probes can't stall a
    # provider list for twice the per-probe timeout.
    deadline = time.monotonic() + _STATUS_TIMEOUT_SECONDS
    rc, out, err = _run_claude(resolved, ["auth", "status"], _PROBE_TIMEOUT_SECONDS)
    info = _parse_auth_status(rc, out, err)

    version = ""
    remaining = deadline - time.monotonic()
    if remaining > 0:
        vrc, vout, _ = _run_claude(
            resolved, ["--version"], min(_PROBE_TIMEOUT_SECONDS, remaining)
        )
        if vrc == 0 and vout.strip():
            version = vout.strip().splitlines()[0].strip()

    info.setdefault("subscription", False)
    info.update(
        {
            "cli_version": version,
            "command": CLAUDE_BINARY,
            "resolved_command": resolved,
        }
    )
    return info


def provider_status(provider_name: str = CLAUDE_CODE_DISPLAY_NAME) -> Dict[str, Any]:
    """Status snapshot in the shape ``get_external_process_provider_status`` returns.

    Keeps ``configured`` / ``provider`` / ``name`` / ``command`` / ``args`` /
    ``resolved_command`` / ``base_url`` / ``logged_in`` so every existing
    external-process consumer keeps working, and adds the Claude-specific
    fields the UI needs to distinguish a plan login from an API-key login.
    """
    probe = probe_claude_auth()
    return {
        "configured": bool(probe.get("resolved_command")),
        "provider": CLAUDE_CODE_PROVIDER_ID,
        "name": provider_name,
        "command": probe.get("command", CLAUDE_BINARY),
        # No args: Hermes does not spawn the CLI for inference, the SDK does.
        "args": [],
        "resolved_command": probe.get("resolved_command"),
        "base_url": CLAUDE_CODE_BASE_URL,
        "logged_in": bool(probe.get("logged_in")),
        "status": probe.get("status", "unknown"),
        "auth_method": probe.get("auth_method", ""),
        "subscription": bool(probe.get("subscription")),
        "subscription_type": probe.get("subscription_type", ""),
        "account": probe.get("account", ""),
        "organization": probe.get("organization", ""),
        "cli_version": probe.get("cli_version", ""),
        "message": probe.get("message", ""),
        "login_command": CLAUDE_LOGIN_COMMAND,
        "logout_command": CLAUDE_LOGOUT_COMMAND,
    }


# ── Migration notice for the pre-SDK direct-OAuth path ──────────────────────

_LEGACY_NOTICE_EMITTED = False


def _anthropic_would_use_a_claude_login(environ: Dict[str, Any]) -> bool:
    """True when picking ``anthropic`` would ride the user's Claude login.

    The ``anthropic`` picker row reports itself authenticated when a Claude
    Code credential exists on disk (``hermes_cli/model_switch.py``), because
    the legacy direct-OAuth path can use it. Selecting that row then bills the
    plan's *extra-usage* pool rather than the plan — the surprise this whole
    provider split exists to end — and it looks identical to a configured API
    key in the picker.

    Detected without reading any credential: an API key set in the environment
    wins outright, so there is nothing to warn about; otherwise ask the
    official CLI whether a login exists.
    """
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if (environ.get(name) or "").strip():
            return False
    try:
        return bool(probe_claude_auth_cached().get("logged_in"))
    except Exception:
        return False


def legacy_provider_notice(
    provider: str,
    *,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Return a migration notice for configs still on the direct-OAuth path.

    Applies when the user asked for one of the legacy Claude slugs, or is on
    ``anthropic`` with a Claude Code OAuth token supplying the credential.
    Those two setups bill against the Claude subscription's extra-usage meter,
    which is not the same source as either replacement — so the user has to
    choose, we do not choose for them.  Returns "" when nothing applies.
    """
    environ = os.environ if env is None else env
    slug = (provider or "").strip().lower()
    # A legacy slug only *rides* the legacy path while the gate is shut and
    # the alias still rewrites it to the direct-OAuth provider. Once the gate
    # is open the same slug names the SDK runtime, and warning about it would
    # tell a correctly-configured user their setup is the thing it replaced.
    is_legacy_slug = (
        slug in LEGACY_ANTHROPIC_ALIASES and legacy_alias_target(slug) is not None
    )
    # Presence check only — the value is never read into a request. This is how
    # we recognise a user still on the legacy path so we can tell them about it.
    legacy_token = environ.get("CLAUDE_CODE_OAUTH_TOKEN")  # claude-boundary: ok — presence only
    has_oauth_token = bool((legacy_token or "").strip())
    anthropic_on_oauth = slug == "anthropic" and (
        has_oauth_token or _anthropic_would_use_a_claude_login(environ)
    )
    if not is_legacy_slug and not anthropic_on_oauth:
        return ""

    label = f"'{slug}'" if slug else "your Claude provider"
    return (
        f"Notice: {label} still uses Hermes' direct Claude OAuth path, which bills "
        "against your Claude plan's extra-usage credits.\n"
        "  Two supported replacements, billed differently — pick one:\n"
        "    • Anthropic API  — provider 'anthropic' with an ANTHROPIC_API_KEY "
        "(metered API billing).\n"
        f"    • Claude subscription — provider '{CLAUDE_CODE_PROVIDER_ID}' via the "
        f"official Claude Agent SDK; run `{CLAUDE_LOGIN_COMMAND}`, then enable "
        "`claude_subscription.enabled` in config.yaml.\n"
        "  Nothing has changed automatically; your current setup keeps working."
    )


def warn_legacy_provider_once(provider: str) -> bool:
    """Print :func:`legacy_provider_notice` to stderr at most once per process.

    Returns True when a notice was emitted.  Never raises — a notice must not
    be able to break provider resolution.
    """
    global _LEGACY_NOTICE_EMITTED
    if _LEGACY_NOTICE_EMITTED:
        return False
    try:
        notice = legacy_provider_notice(provider)
        if not notice:
            return False
        _LEGACY_NOTICE_EMITTED = True
        sys.stderr.write("\n" + notice + "\n\n")
        return True
    except Exception:
        return False
