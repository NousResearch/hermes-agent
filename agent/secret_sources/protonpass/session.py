"""``pass-cli`` session establishment + subprocess plumbing.

Every ``pass-cli`` child runs with a MINIMAL, isolated environment built by
:func:`_child_env`: just the handful of ambient vars a CLI needs plus the
``PROTON_PASS_*`` vars, with an isolated ``PROTON_PASS_SESSION_DIR`` so hermes
never clobbers the user's interactive session.  The service token lives ONLY in
that minimal dict and is NEVER logged, stored, or surfaced in a warning.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List

from .install import _hermes_bin_dir

logger = logging.getLogger(__name__)

# Timeout for a single pass-cli subprocess, in seconds.
_PASS_CLI_RUN_TIMEOUT = 30

# Default reason recorded on every scoped (agent-token) fetch.  Harmless under
# a full personal session; REQUIRED under scoped agent-token sessions.
_DEFAULT_AGENT_REASON = "hermes-agent startup secret injection"

# Full CSI escape-sequence matcher: ESC [ <params> <intermediates> <final>.
# Strips colour / cursor sequences pass-cli might emit even with NO_COLOR set.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _token_fingerprint(token: str) -> str:
    """SHA-256 prefix used as a cache key — never logged, never displayed."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _session_dir(token: str = "") -> Path:
    """Isolated pass-cli session dir under ``<hermes_home>/protonpass-session``.

    Confirmed: pass-cli honours ``PROTON_PASS_SESSION_DIR``.  We point it at a
    dir we own so hermes never clobbers the user's interactive session.
    ``_hermes_bin_dir()`` is ``<hermes_home>/bin``, so its parent is
    hermes_home.

    The directory is ALWAYS suffixed with the token fingerprint (never the token
    itself) so two concurrent processes authenticating with DIFFERENT tokens get
    DIFFERENT session dirs and cannot clobber each other's session material.  The
    fingerprint is a SHA-256 prefix and is never logged or displayed.  A bare
    ``token=""`` simply fingerprints the empty string — every real caller passes
    a non-empty token, so the suffix is unconditional and the "session
    isolation" guarantee always holds.
    """
    parent = _hermes_bin_dir().parent
    return parent / f"protonpass-session-{_token_fingerprint(token)}"


def _establish_session(token: str, binary: Path) -> List[str]:
    """Establish a non-interactive ``pass-cli`` session from ``token``.

    Returns a list of non-fatal warnings; raises :class:`RuntimeError` on a
    fatal auth failure.  The token is NEVER included in any returned warning,
    raised message, or log line.

    Confirmed (pass-cli 2.1.1): the token is read from
    ``PROTON_PASS_PERSONAL_ACCESS_TOKEN`` in the child env, ``pass-cli login``
    establishes the session non-interactively from it, and ``pass-cli info``
    (exit 0) verifies it.  On auth failure we do ONE ``pass-cli logout
    --force`` + retry login, then give up.  All invocations share the one
    isolated ``PROTON_PASS_SESSION_DIR`` built by :func:`_child_env`.
    """
    env = _child_env(token)
    warnings: List[str] = []

    # login → info, with one logout --force + relogin recovery on failure.
    ok, _ = _try_login_and_verify(binary, env)
    if ok:
        return warnings

    # Recovery: clear any stale session in our isolated dir and retry once.
    _run_pass_cli([str(binary), "logout", "--force"], env)
    ok, login_err = _try_login_and_verify(binary, env)
    if ok:
        warnings.append("pass-cli session recovered after a logout/relogin retry")
        return warnings

    # Surface the (redacted, ANSI-stripped) login stderr to aid debugging.  The
    # token is scrubbed defensively even though it should never appear here.
    detail = _redact_token(_clean_stream(login_err), token)
    message = (
        "pass-cli could not establish a session (login + verify failed after "
        "one logout/relogin retry); the token may be invalid or expired"
    )
    if detail:
        message = f"{message}: {detail}"
    raise RuntimeError(message)


def _try_login_and_verify(binary: Path, env: Dict[str, str]):
    """Run ``pass-cli login`` then ``pass-cli info``.

    Returns ``(ok, login_stderr)`` where ``ok`` is True iff both exit 0.  The
    login stderr (raw, possibly containing ANSI; never the token) is returned so
    a caller can surface a redacted form on final failure.
    """
    login = _run_pass_cli([str(binary), "login"], env)
    if login is None:
        return False, ""
    if login.returncode != 0:
        # Consistency with the stderr-only secret-command rule: surface ONLY
        # stderr (login is not secret-bearing, but we keep the convention so no
        # path ever leans on stdout for a diagnostic detail).
        return False, (login.stderr or "")
    info = _run_pass_cli([str(binary), "info"], env)
    ok = info is not None and info.returncode == 0
    return ok, ""


def _run_pass_cli(cmd: List[str], env: Dict[str, str]):
    """Run a pass-cli command, returning the CompletedProcess or None.

    None signals a transport failure (timeout / OSError).  The token never
    appears here — it already lives in ``env``.  Callers decide whether a
    failure is fatal.
    """
    try:
        return subprocess.run(  # noqa: S603 — pass-cli path is trusted
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",  # pass-cli emits UTF-8; don't decode via the locale codepage
            errors="replace",  # invalid UTF-8 in output can't raise a decode error
            timeout=_PASS_CLI_RUN_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None


# Ambient vars a CLI legitimately needs to find shared libs / a home dir.  This
# allow-list is the ONLY parent-env passthrough — everything else (the token
# and every other loaded secret) is intentionally excluded.
_AMBIENT_ENV_NAMES = (
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TMP", "TMPDIR",
    "LANG", "LC_ALL",
)


def _minimal_env() -> Dict[str, str]:
    """Build a MINIMAL, scrubbed subprocess env with NO secrets in it.

    Rather than ``os.environ.copy()`` (which would hand a child every other
    secret in the parent process), we carry only the handful of ambient vars a
    CLI legitimately needs plus ``NO_COLOR``.  Crucially this contains NO token
    and NONE of the inherited secrets, so it is safe to hand to a probe (e.g. a
    ``--version`` check) of a binary we have not yet verified.

    :func:`_child_env` builds on this and is the ONLY place that adds the token.
    """
    env: Dict[str, str] = {}
    for name in _AMBIENT_ENV_NAMES:
        val = os.environ.get(name)
        if val is not None:
            env[name] = val
    env["NO_COLOR"] = "1"
    return env


def _ensure_private_session_dir(session_dir: Path) -> None:
    """Create the isolated session dir and lock it to 0o700, or RAISE.

    Rejects a symlinked final component BEFORE chmod/write so we never follow it
    to an attacker-chosen target.  The path embeds the token fingerprint
    (unguessable without the token), so this is best-effort hardening against a
    pre-created symlink — it checks only the final component (HERMES_HOME is
    trusted) and is not TOCTOU-proof.  The RuntimeError is caught upstream so an
    unverifiable session store skips Proton Pass without blocking startup; the
    message carries only the path, never the token.
    """
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Could not create the Proton Pass session directory; skipping: {exc}"
        ) from exc
    if session_dir.is_symlink():
        raise RuntimeError(
            "Proton Pass session directory is a symlink; refusing to use it for "
            "session storage and skipping Proton Pass."
        )
    try:
        os.chmod(session_dir, 0o700)
    except OSError as exc:
        raise RuntimeError(
            f"Could not secure the Proton Pass session directory; skipping: {exc}"
        ) from exc


def _parent_env_or_default(name: str, default: str) -> str:
    """Parent-env value for ``name``, or ``default`` if unset/blank.

    Our child env starts minimal, so an explicit passthrough is the only way to
    honor a user override; an empty/whitespace value falls back to the secure
    default rather than silently disabling it.
    """
    return (os.environ.get(name) or "").strip() or default


def _child_env(token: str) -> Dict[str, str]:
    """Build a MINIMAL subprocess env for a pass-cli invocation.

    Starts from :func:`_minimal_env` (no inherited secrets) and adds only the
    ``PROTON_PASS_*`` vars pass-cli needs; the token lives ONLY in this dict.
    The session dir is isolated + 0o700 (see :func:`_ensure_private_session_dir`)
    so hermes never touches the user's interactive session.  ``KEY_PROVIDER``
    defaults to ``fs`` because pass-cli's default OS keyring is absent on
    headless servers/containers (where `login` would otherwise fail); the key
    then lives on disk in the 0o700 dir — acceptable since the bootstrap token
    already sits at 0600 in the same home — and a keyring host can override it.
    """
    env = _minimal_env()
    env["PROTON_PASS_PERSONAL_ACCESS_TOKEN"] = token
    session_dir = _session_dir(token)
    _ensure_private_session_dir(session_dir)
    env["PROTON_PASS_SESSION_DIR"] = str(session_dir)
    env["PROTON_PASS_AGENT_REASON"] = _DEFAULT_AGENT_REASON
    env["PROTON_PASS_KEY_PROVIDER"] = _parent_env_or_default(
        "PROTON_PASS_KEY_PROVIDER", "fs"
    )
    env["PROTON_PASS_DISABLE_TELEMETRY"] = _parent_env_or_default(
        "PROTON_PASS_DISABLE_TELEMETRY", "1"
    )
    return env


def _clean_stream(text: str) -> str:
    """Strip ANSI escapes and surrounding whitespace from a CLI stream.

    Uses a full CSI matcher (``ESC [ ... <final>``) rather than just dropping
    the ESC byte, so a leftover ``[0m`` tail can't survive into a warning.
    """
    return _ANSI_CSI_RE.sub("", text or "").replace("\x1b", "").strip()


def _redact_token(text: str, token: str) -> str:
    """Remove any occurrence of the token from a string before surfacing it.

    Defense-in-depth: we never *intend* to pass the token to a stream that
    ends up in a warning, but if pass-cli ever echoes it we scrub it here so
    it can't leak into logs/errors.
    """
    if not token:
        return text
    return text.replace(token, "***REDACTED***")
