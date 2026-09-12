"""Authority profile — controls what the agent can do per session.

Two profiles, opt-in by Austin via ``config.yaml``::

    authority_profile: production   # default — current safety floors
    authority_profile: principal    # lifts edit/create/install rules

The principal profile exists so that on Austin's own machine, the agent can
install dependencies, edit canonical config files, and author new skills
**without** requiring an extra round-trip per action. Every action that is
allowed under principal is logged (see :func:`log_action`) so the audit trail
is intact.

Both profiles preserve the **absolute deny** set (cookie/secret hygiene,
smart-approval gate, ``hermes-agent/`` upstream checkout, 1mcp aggregator,
FCPS customer-facing cron deliveries, cross-repo ``.hermes.md`` files).

Per-action allow/deny decisions go through :func:`allows`. SOUL.md and
runtime patch-tool guards consult this function instead of hard-coding rules.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import yaml

# HERMES_HOME is resolved lazily so we don't import the heavy hermes_constants
# at module import time (this module is imported by config.py on every load).
HERMES_HOME = Path(
    os.environ.get("HERMES_HOME", r"C:\Users\bbask\AppData\Local\hermes")
)
CONFIG_PATH = HERMES_HOME / "config.yaml"

Profile = Literal["production", "principal"]

#: Actions that **never** execute under either profile.
ABSOLUTE_DENIES: frozenset[str] = frozenset(
    {
        "commit_cookie_json",
        "commit_session_token_json",
        "bypass_approvals_deny",
        "disable_smart_approval",
        "modify_fcps_cron_delivery",
        "modify_1mcp_runtime",
        "edit_other_repo_hermes_md",
        "modify_hermes_agent_upstream_checkout",
        "skip_error_class",
        "act_outside_authority",
    }
)

#: Actions denied under production but allowed under principal.
#: Each entry corresponds to a "Hard rule" the plan moves out of SOUL.md
#: prose into code.
PRODUCTION_DENIES: frozenset[str] = frozenset(
    {
        "edit_config_yaml",
        "edit_soul_md",
        "edit_env",
        "edit_plugin_yaml",
        "edit_auth_json",
        "edit_memory_md",
        "edit_user_md",
        "edit_other_repo_hermes_md",  # also in absolute_denies; defensive double-list
        "pip_install_venv",
        "pipx_install",
        "winget_install_user",
        "npm_install_global",
        "git_clone_tools",
        "download_binary",
        "mcp_server_install",
        "plugin_install",
        "skill_create",
        "skill_modify",
    }
)

#: Actions explicitly allowed under principal. (Anything not in this set is
#: still denied under principal — explicit allow-list, not deny-list.)
PRINCIPAL_ALLOWS: frozenset[str] = frozenset(
    {
        "edit_config_yaml",
        "edit_soul_md",
        "edit_env",
        "edit_plugin_yaml",
        "edit_memory_md",
        "edit_user_md",
        "pip_install_venv",
        "pipx_install",
        "winget_install_user",
        "npm_install_global",
        "git_clone_tools",
        "download_binary",
        "mcp_server_install",
        "plugin_install",
        "skill_create",
        "skill_modify",
    }
)

#: Cache so we don't re-read config.yaml on every ``allows()`` call.
#: 5-second TTL keeps the change visible within one turn but avoids hot-path
#: disk reads. Test code calls ``reset_cache()`` to control this.
_cache_profile: Profile | None = None
_cache_ts: float = 0.0
_CACHE_TTL_SECONDS = 5.0


def reset_cache() -> None:
    """Clear the cached profile. Useful for tests and after a config flip."""
    global _cache_profile, _cache_ts
    _cache_profile = None
    _cache_ts = 0.0


def _read_profile_from_config() -> Profile:
    if not CONFIG_PATH.exists():
        return "production"
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return "production"
    raw = cfg.get("authority_profile", "production")
    if isinstance(raw, str) and raw in ("production", "principal"):
        return raw  # type: ignore[return-value]
    return "production"


def get_profile() -> Profile:
    """Return the active authority profile, cached for 5 seconds."""
    global _cache_profile, _cache_ts
    now = time.time()
    if _cache_profile is None or (now - _cache_ts) > _CACHE_TTL_SECONDS:
        _cache_profile = _read_profile_from_config()
        _cache_ts = now
    return _cache_profile


def allows(action: str) -> bool:
    """Return True if ``action`` is permitted under the active profile.

    Order of checks:
      1. Absolute denies win over everything (always fail-closed).
      2. ``production`` denies anything in :data:`PRODUCTION_DENIES`.
      3. ``principal`` allows only what's in :data:`PRINCIPAL_ALLOWS`.
    """
    if action in ABSOLUTE_DENIES:
        return False
    profile = get_profile()
    if profile == "principal":
        return action in PRINCIPAL_ALLOWS
    # production default
    return action not in PRODUCTION_DENIES


def deny_reason(action: str) -> str:
    """Human-readable reason why an action would be denied. Useful for error msgs."""
    if action in ABSOLUTE_DENIES:
        return f"action '{action}' is an absolute deny (both profiles)"
    profile = get_profile()
    if profile == "production" and action in PRODUCTION_DENIES:
        return (
            f"action '{action}' is denied under authority_profile='production'. "
            "Set authority_profile: principal in config.yaml to enable."
        )
    if profile == "principal" and action not in PRINCIPAL_ALLOWS:
        return f"action '{action}' is not in the principal allow-list"
    return ""


def log_action(action: str, target: str | None = None) -> None:
    """Append a privileged-action line to today's audit log.

    Per-turn authorization for specific actions still flows through this
    function even outside the principal profile (the production profile
    allows per-turn authorizations, just not blanket ones).

    Best-effort logging: filesystem errors are NOT silently swallowed —
    they print to stderr so the failure is visible. Logging never blocks
    the action itself (no exception propagates).
    """
    log_dir = HERMES_HOME / "cache" / "authority-actions"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        log_file = log_dir / f"{now.date().isoformat()}.log"
        line = (
            f"{now.isoformat(timespec='seconds')}Z\t"
            f"profile={get_profile()}\taction={action}\ttarget={target or ''}\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        # Visible failure — print to stderr but don't propagate.
        print(
            f"[authority] WARN: failed to log action '{action}' -> {exc}",
            file=__import__("sys").stderr,
        )


def format_constraints() -> str:
    """One-paragraph description of the active profile's permissions.

    Used by SOUL.md kickoff and the agent's self-model when introspecting
    what it can and can't do without round-tripping.
    """
    profile = get_profile()
    if profile == "principal":
        allows_str = ", ".join(sorted(PRINCIPAL_ALLOWS))
        return (
            "Authority profile: principal. The agent may edit canonical "
            "config files, install dependencies via pip/winget/npm/git, "
            "create new skills, and add new plugins — all logged for audit. "
            f"Allowed actions: {allows_str}. "
            f"Always denied: {', '.join(sorted(ABSOLUTE_DENIES))}."
        )
    return (
        "Authority profile: production. The agent follows SOUL.md hard rules: "
        "no edits to canonical files, no dependency installs, no skill authoring "
        "without explicit per-turn Austin authorization. "
        "Set authority_profile: principal in config.yaml for the expanded mode."
    )


def bootstrap_session(
    packages: list[tuple[str, str]] | None = None,
    *,
    dry_run: bool = True,
) -> dict:
    """Session-start bootstrap — principal profile installs missing
    dependencies transparently; production profile is a no-op unless
    ``packages`` is explicitly named (per-turn authorization).

    Args:
        packages: List of ``(backend, spec)`` pairs. ``backend`` is one of
            ``"pip"``, ``"pipx"``, ``"npm"``, ``"winget"``, ``"git"``;
            ``spec`` is the package name (pip: ``name`` or ``name>=version``;
            npm: ``name`` or ``name@version``; git: clone URL).
            ``None`` defaults to ``DEFAULT_SESSION_PACKAGES`` (sentence-
            transformers + jieba for Eagle Eye L4).
        dry_run: ``True`` (default — safe for production) reports what
            WOULD be installed without running any installer. ``False``
            actually invokes installers.

    Returns:
        Dict ``{backend_spec: status}`` where status is one of
        ``"installed"``, ``"would_install"``, ``"skipped_production"``,
        ``"blocked_profile"``, ``"failed: <reason>"``.

    Called by the session-start hook when the active profile is
    ``principal``. Marriage of "Austin explicitly opted in" and "production
    never auto-installs; per-turn named packages only".
    """
    if packages is None:
        packages = DEFAULT_SESSION_PACKAGES

    profile = get_profile()
    results: dict = {}
    for backend, spec in packages:
        action = _action_for_backend(backend)
        if not allows(action):
            results[f"{backend}:{spec}"] = (
                "skipped_production" if profile == "production"
                else "blocked_profile"
            )
            continue
        if dry_run:
            results[f"{backend}:{spec}"] = "would_install"
            continue
        try:
            log_action(action, target=f"{backend}:{spec}")
            _run_installer(backend, spec)
            results[f"{backend}:{spec}"] = "installed"
        except Exception as exc:
            results[f"{backend}:{spec}"] = f"failed: {exc}"
    return results


def _action_for_backend(backend: str) -> str:
    """Map an installer name to the authority-action key it triggers."""
    return {
        "pip": "pip_install_venv",
        "pipx": "pipx_install",
        "npm": "npm_install_global",
        "winget": "winget_install_user",
        "git": "git_clone_tools",
    }.get(backend, "download_binary")


def _run_installer(backend: str, spec: str) -> None:
    """Invoke the actual installer. Subprocess; blocks the caller.

    Hardened against shell injection: each spec is passed as a list of
    argv tokens, never interpolated into a string. Only reachable via
    ``bootstrap_session`` for backends that ``allows()`` returned True for.
    """
    import subprocess
    import sys
    invocations = {
        "pip": [sys.executable, "-m", "pip", "install", spec],
        "pipx": ["pipx", "install", spec],
        "npm": ["npm", "install", "-g", spec],
        "winget": ["winget", "install", "--scope", "user", "-e", spec],
        "git": ["git", "clone", spec],
    }
    argv = invocations.get(backend)
    if argv is None:
        raise ValueError(f"unknown backend: {backend!r}")
    subprocess.check_call(argv)


# Default session-start dependency set. Short list — every entry is one
# more failure mode if the install machine has no network.
DEFAULT_SESSION_PACKAGES: list[tuple[str, str]] = [
    # sentence-transformers is the L4 dense-embedding engine Eagle Eye needs.
    # jieba is its Chinese-segmentation dependency.
    ("pip", "sentence-transformers==6.0.1"),
    ("pip", "jieba==0.42.1"),
]
