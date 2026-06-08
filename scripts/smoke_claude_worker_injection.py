#!/usr/bin/env python3
"""Smoke test for scoped Claude Code OAuth token injection.

Verifies the narrow, opt-in passthrough of ``CLAUDE_CODE_OAUTH_TOKEN`` to
direct ``claude -p`` worker subprocesses without globally unblocking secrets.

This is a *diagnostic*, not a pytest. Run it by hand on a machine where the
Claude CLI is installed to confirm worker auth reliability end to end:

    venv/bin/python scripts/smoke_claude_worker_injection.py

What it checks
--------------
  1. Config paths + terminal backend are resolvable.
  2. The ``delegation.claude_code_pass_oauth_token`` flag value.
  3. A direct LocalEnvironment ``claude -p`` worker receives an injected
     **fake** token (via a shim ``claude`` on PATH).
  4. Non-Claude commands do NOT receive the token.
  5. A fake token actually reaches the *real* Claude CLI — the CLI rejects it
     with a 401 / "invalid bearer token" rather than "not logged in" (proving
     the token is delivered, not stripped).
  6. A real PONG round-trips when real CLI auth / token is available.

Safety
------
  - Never prints token values or OAuth URLs/codes; all sensitive strings are
    masked before display.
  - Forces the passthrough flag ON only in-process (monkeypatching the config
    loader); it never writes to the user's real config.
  - Steps 5 and 6 are SKIPPED when the real ``claude`` binary is absent.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Make the repo importable when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _maybe_reexec_with_repo_python() -> None:
    """When launched as an executable, switch to a repo venv with deps.

    The diagnostic imports Hermes modules, which depend on PyYAML. macOS/system
    Python often lacks those deps, while the repo venv has them. `python
    scripts/...` already uses the caller's interpreter; this only improves the
    shebang path (`scripts/smoke_claude_worker_injection.py`).
    """
    if os.environ.get("HERMES_SMOKE_NO_VENV_REEXEC"):
        return
    try:
        import yaml  # noqa: F401
        return
    except Exception:
        pass

    current = Path(sys.executable).resolve()
    for rel in ("venv/bin/python", ".venv/bin/python"):
        # Keep the venv symlink path intact. Resolving it jumps to the base
        # Python binary and drops the venv's site-packages from sys.path.
        candidate = _REPO_ROOT / rel
        if not candidate.exists() or candidate.resolve() == current:
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import yaml"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            env = os.environ.copy()
            env["HERMES_SMOKE_NO_VENV_REEXEC"] = "1"
            os.execve(str(candidate), [str(candidate), *sys.argv], env)


_maybe_reexec_with_repo_python()

FAKE_TOKEN = "cc-worker-FAKE-do-not-use-0000000000000000"
_SENSITIVE = [FAKE_TOKEN]


# ── output helpers ────────────────────────────────────────────────────────

class _Tally:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0


TALLY = _Tally()


def _register_secret(value: str | None) -> None:
    if value and value.strip() and value not in _SENSITIVE:
        _SENSITIVE.append(value)


def _mask(text: str) -> str:
    """Redact any known sensitive substrings before display."""
    out = text or ""
    for secret in _SENSITIVE:
        if secret:
            out = out.replace(secret, "***REDACTED***")
    # Belt-and-suspenders: blank out anything that looks like an oauth token
    # or an authorize URL/code we might not have registered.
    out = re.sub(r"(CLAUDE_CODE_OAUTH_TOKEN=)\S+", r"\1***REDACTED***", out)
    out = re.sub(r"https://\S*oauth\S*", "https://***REDACTED-OAUTH-URL***", out)
    return out


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _ok(msg: str) -> None:
    TALLY.passed += 1
    print(f"  PASS  {_mask(msg)}")


def _fail(msg: str) -> None:
    TALLY.failed += 1
    print(f"  FAIL  {_mask(msg)}")


def _skip(msg: str) -> None:
    TALLY.skipped += 1
    print(f"  SKIP  {_mask(msg)}")


def _info(msg: str) -> None:
    print(f"  ..    {_mask(msg)}")


# ── token-source / auth-error classification ──────────────────────────────

def _looks_like_not_logged_in(text: str) -> bool:
    low = (text or "").lower()
    return any(
        marker in low
        for marker in (
            "not logged in",
            "please run /login",
            "run `claude /login`",
            "no api key",
            "no authentication",
        )
    )


def _looks_like_bad_bearer(text: str) -> bool:
    low = (text or "").lower()
    return any(
        marker in low
        for marker in (
            "401",
            "invalid bearer token",
            "invalid_bearer",
            "unauthorized",
            "authentication_error",
            "invalid api key",
            "oauth token has expired",
        )
    )


# ── check 1: config paths + backend ───────────────────────────────────────

def check_config_paths() -> None:
    _section("1. Config paths + terminal backend")
    try:
        from hermes_cli.config import get_config_path, get_env_path, load_config
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        cfg_path = get_config_path()
        env_path = get_env_path()
        _info(f"HERMES_HOME : {home}")
        _info(f"config.yaml : {cfg_path} (exists={cfg_path.exists()})")
        _info(f".env        : {env_path} (exists={env_path.exists()})")

        cfg = load_config() or {}
        backend = ((cfg.get("terminal") or {}).get("backend")) or "local"
        _info(f"terminal.backend : {backend}")
        if backend != "local":
            _info(
                "backend is not 'local' — token injection only applies to the "
                "LocalEnvironment path; remote backends are out of scope."
            )
        _ok("config paths and terminal backend resolved")
    except Exception as exc:  # pragma: no cover - diagnostic
        _fail(f"could not resolve config: {exc!r}")


# ── check 2: flag value ───────────────────────────────────────────────────

def check_flag_value() -> None:
    _section("2. delegation.claude_code_pass_oauth_token")
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        delegation = cfg.get("delegation") or {}
        value = delegation.get("claude_code_pass_oauth_token", False)
        _info(f"configured value : {value!r}")
        if value:
            _ok("passthrough is ENABLED in real config")
        else:
            _ok(
                "passthrough is DISABLED in real config (default-safe). "
                "Injection checks below force it on in-process only."
            )
    except Exception as exc:  # pragma: no cover - diagnostic
        _fail(f"could not read flag: {exc!r}")


# ── shared: force flag on in-process, point PATH at a shim claude ──────────

def _write_shim_claude(dir_path: Path) -> Path:
    """A fake `claude` that reports whether the token reached its env."""
    shim = dir_path / "claude"
    shim.write_text(
        "#!/bin/sh\n"
        'if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then\n'
        "  echo TOKEN_PRESENT\n"
        "else\n"
        "  echo TOKEN_ABSENT\n"
        "fi\n"
    )
    shim.chmod(0o755)
    return shim


def _force_flag_on() -> None:
    import hermes_cli.config as cfg_mod

    cfg_mod.load_config = lambda: {  # type: ignore[assignment]
        "delegation": {"claude_code_pass_oauth_token": True}
    }


# ── check 3 + 4: shim worker receives token; non-claude does not ──────────

def check_local_injection_with_shim() -> None:
    _section("3+4. LocalEnvironment injection (fake token, shim claude)")
    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = FAKE_TOKEN
    _register_secret(FAKE_TOKEN)
    _force_flag_on()

    try:
        from tools.environments.local import LocalEnvironment
    except Exception as exc:  # pragma: no cover
        _fail(f"could not import LocalEnvironment: {exc!r}")
        return

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        shim = _write_shim_claude(tmp)
        env = LocalEnvironment(cwd=str(tmp), timeout=20)
        try:
            allowed = env.execute(f"{shim} -p 'Reply exactly: ok'", timeout=20)
            denied = env.execute(f"{shim} -p hi & env", timeout=20)
            env_hijack = env.execute(f"env PATH={tmp} claude -p hi", timeout=20)
            non_claude = env.execute("env", timeout=20)
            subsequent = env.execute("env", timeout=20)
        finally:
            env.cleanup()

        # 3: direct worker gets the token.
        if "TOKEN_PRESENT" in allowed["output"]:
            _ok("direct `claude -p` worker received the injected token")
        else:
            _fail(f"worker did not receive token (output: {allowed['output'].strip()!r})")

        # 4a: shell-control wrapper must NOT receive the token.
        if "TOKEN_PRESENT" not in denied["output"]:
            _ok("`claude -p ... & env` leak form did NOT receive the token")
        else:
            _fail("shell-control wrapper leaked the token")

        # 4a.1: env-prefixed PATH/loader overrides must NOT receive the token.
        if "TOKEN_PRESENT" not in env_hijack["output"]:
            _ok("`env PATH=... claude -p` hijack form did NOT receive the token")
        else:
            _fail("env-prefixed command leaked the token")

        # 4b: plain non-claude command must not see it.
        if "CLAUDE_CODE_OAUTH_TOKEN" not in non_claude["output"]:
            _ok("non-claude `env` command did not receive the token")
        else:
            _fail("non-claude command saw CLAUDE_CODE_OAUTH_TOKEN")

        # 4c: token must not persist into the session snapshot.
        if "CLAUDE_CODE_OAUTH_TOKEN" not in subsequent["output"]:
            _ok("token did not persist into the session env snapshot")
        else:
            _fail("token leaked into a subsequent command via the snapshot")


# ── check 5: fake token reaches the real Claude CLI → 401, not "logged out" ─

def check_real_cli_rejects_fake_token() -> None:
    _section("5. Real Claude CLI rejects the fake token (401, not logged-out)")
    claude = shutil.which("claude")
    if not claude:
        _skip("`claude` binary not on PATH")
        return

    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = FAKE_TOKEN
    _register_secret(FAKE_TOKEN)
    _force_flag_on()

    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=tempfile.gettempdir(), timeout=60)
    try:
        result = env.execute("claude -p 'Return exactly PONG'", timeout=60)
    finally:
        env.cleanup()

    out = result["output"]
    _info(f"exit={result['returncode']} output={out.strip()[:300]!r}")

    if "PONG" in out and not _looks_like_bad_bearer(out):
        _skip(
            "real CLI returned PONG with a FAKE token — a valid stored login "
            "took precedence over the env token, so token delivery can't be "
            "isolated here. Re-run after `claude /logout` to assert the 401 path."
        )
        return
    if _looks_like_not_logged_in(out) and not _looks_like_bad_bearer(out):
        _fail(
            "CLI reports 'not logged in' — the fake token did NOT reach the CLI "
            "(it was stripped before exec)"
        )
        return
    if _looks_like_bad_bearer(out):
        _ok(
            "CLI rejected the fake token with an auth/401 error — token IS being "
            "delivered to the real CLI"
        )
        return
    _skip(
        "CLI produced an unrecognized response; inspect masked output above "
        "to classify manually"
    )


# ── check 6: real PONG when genuine auth is available ─────────────────────

def check_real_pong() -> None:
    _section("6. Real PONG round-trip (genuine CLI auth)")
    claude = shutil.which("claude")
    if not claude:
        _skip("`claude` binary not on PATH")
        return

    # Use the real stored login / real token, NOT the fake one.
    os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    real_token = os.environ.get("_HERMES_SMOKE_REAL_OAUTH_TOKEN", "").strip()
    if real_token:
        _register_secret(real_token)

    import subprocess

    sub_env = os.environ.copy()
    sub_env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    if real_token:
        sub_env["CLAUDE_CODE_OAUTH_TOKEN"] = real_token
        _info("using real token from _HERMES_SMOKE_REAL_OAUTH_TOKEN")
    else:
        _info("no explicit real token; relying on stored CLI login")

    try:
        proc = subprocess.run(
            [claude, "-p", "Return exactly PONG"],
            env=sub_env,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired:
        _fail("real claude CLI timed out")
        return

    combined = (proc.stdout or "") + (proc.stderr or "")
    _info(f"exit={proc.returncode} output={combined.strip()[:300]!r}")

    if "PONG" in combined:
        _ok("real Claude CLI returned PONG — worker auth is healthy")
    elif _looks_like_not_logged_in(combined) or _looks_like_bad_bearer(combined):
        _skip(
            "no valid CLI auth available in this environment — run "
            "`claude /login` (or set _HERMES_SMOKE_REAL_OAUTH_TOKEN) to exercise"
        )
    else:
        _fail("real CLI did not return PONG; inspect masked output above")


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("Claude Code worker OAuth-token injection smoke test")
    print("(no token values or OAuth URLs are printed)")

    check_config_paths()
    check_flag_value()
    check_local_injection_with_shim()
    check_real_cli_rejects_fake_token()
    check_real_pong()

    _section("Summary")
    print(f"  passed={TALLY.passed} failed={TALLY.failed} skipped={TALLY.skipped}")
    return 1 if TALLY.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
