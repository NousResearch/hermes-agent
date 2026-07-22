#!/usr/bin/env python3
"""
Secure credential input — prompted with **** masking, never printed or logged.

Usage:
    python3 secure-input.py --key OPENAI_API_KEY
    python3 secure-input.py --key DATABASE_PASSWORD --no-confirm
    python3 secure-input.py --key GITHUB_TOKEN --file /path/to/.env

The script prompts for the secret value with masked echo (****), optionally
confirms it, then appends KEY=VALUE to the target .env file (default:
~/.hermes/.env). The actual value is never written to stdout, stderr, or any
log — only a success/error message is printed.

Exit codes:
    0 — credential stored successfully
    1 — input error (empty, mismatch, interrupted)
    2 — file write error
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Securely store a credential in an .env file."
    )
    parser.add_argument(
        "--key", required=True,
        help="Environment variable name (e.g. OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--file", default=None,
        help="Target .env file (default: ~/.hermes/.env)"
    )
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Skip confirmation prompt (useful when pasting long tokens)"
    )
    args = parser.parse_args()

    # Validate key name
    key = args.key.strip()
    if not _is_valid_env_name(key):
        print(f"ERROR: invalid env var name: {key!r}", file=sys.stderr)
        sys.exit(1)

    env_path = Path(args.file) if args.file else _get_env_path()

    # Read secret
    value = _read_secret(f"{key}")
    if not value or not value.strip():
        print("ERROR: empty value — nothing stored.", file=sys.stderr)
        sys.exit(1)
    value = value.strip()

    # Confirm
    if not args.no_confirm:
        value2 = _read_secret(f"{key} (confirm)")
        if value != value2:
            print("ERROR: values do not match.", file=sys.stderr)
            sys.exit(1)

    # Write .env
    try:
        _upsert_env(env_path, key, value)
    except OSError as exc:
        print(f"ERROR: could not write {env_path}: {exc}", file=sys.stderr)
        sys.exit(2)
    finally:
        # Python strings are immutable — the value never entered stdout/stderr
        # so the best defence is already in place
        pass

    print(f"OK: {key} → {env_path}")


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def _get_env_path() -> Path:
    """Return the profile-aware Hermes .env path."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / ".env"
    except ImportError:
        # Fallback for standalone use without Hermes installed
        env = os.environ.get("HERMES_HOME", "")
        if env:
            return Path(env) / ".env"
        return Path.home() / ".hermes" / ".env"


def _read_secret(prompt: str) -> str:
    """Read a secret with **** masked echo. Falls back to getpass."""
    try:
        # Preferred: full masked_secret_prompt (raw terminal, **** per char)
        from hermes_cli.secret_prompt import masked_secret_prompt
        return masked_secret_prompt(f"{prompt}: ", mask="*")
    except Exception:
        import getpass
        return getpass.getpass(f"{prompt}: ")


def _upsert_env(env_path: Path, key: str, value: str) -> None:
    """Upsert KEY=value into env_path atomically."""
    import secrets as _secrets

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.touch(mode=0o600, exist_ok=True)
    env_path.chmod(0o600)

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        lines = []

    new_line = f"{key}={value}\n"
    replaced = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            existing_key = stripped.split("=", 1)[0]
            if existing_key == key or existing_key.removeprefix("export ") == key:
                out.append(new_line)
                replaced = True
                continue
        out.append(line)
    if not replaced:
        # Guard against .env files missing trailing newline
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        out.append(new_line)

    # os.open with 0o600 — strict from creation, no umask race window
    tmp = env_path.with_suffix(env_path.suffix + f".tmp.{_secrets.token_hex(4)}")
    data = "".join(out)
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, env_path)


if __name__ == "__main__":
    main()
