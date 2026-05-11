#!/usr/bin/env python3
"""Non-extracting GitHub broker for Daimon sandbox containers."""
from __future__ import annotations

import json
import os
import pwd
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

BROKER_HOST = os.environ.get("DAIMON_GH_BROKER_HOST", "0.0.0.0")  # nosec B104 — intentional: container-internal only, isolated Docker network
BROKER_PORT = int(os.environ.get("DAIMON_GH_BROKER_PORT", "7842"))
TOKEN_PATH = os.environ.get("GH_TOKEN_FILE", "/run/secrets/gh_token")
GH_REAL = os.environ.get("GH_REAL", "/usr/bin/gh")
ALLOWED_REPO = os.environ.get("DAIMON_GH_ALLOWED_REPO", "NousResearch/hermes-agent")
GH_CONFIG_DIR = os.environ.get("DAIMON_GH_CONFIG_DIR", "/tmp/daimon-gh-config")
DEFAULT_TIMEOUT_SEC = 60
MAX_TIMEOUT_SEC = 120
MAX_OUTPUT_BYTES = 1_000_000

ALLOWED_COMMANDS = {
    ("issue", "list"),
    ("issue", "view"),
    ("issue", "create"),
    ("issue", "comment"),
    ("issue", "close"),
    ("issue", "edit"),
    ("pr", "list"),
    ("pr", "view"),
    ("pr", "create"),
    ("pr", "comment"),
    ("pr", "diff"),
    ("pr", "checks"),
    ("search", "issues"),
    ("search", "prs"),
    ("search", "code"),
}

DENIED_COMMANDS = {
    "alias",
    "api",
    "auth",
    "config",
    "extension",
    "gpg-key",
    "secret",
    "ssh-key",
}

DENIED_FLAGS = {
    "--hostname",
    "--with-token",
}

REPO_FLAGS = {"-R", "--repo"}


class BrokerError(Exception):
    """User-facing broker denial."""


def _json_response(ok: bool, exit_code: int, stdout: str = "", stderr: str = "") -> bytes:
    return (
        json.dumps(
            {
                "ok": ok,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            },
            ensure_ascii=False,
        )
        + "\n"
    ).encode()


def _limited_text(data: bytes) -> str:
    if len(data) > MAX_OUTPUT_BYTES:
        data = data[:MAX_OUTPUT_BYTES] + b"\n[broker output truncated]\n"
    return data.decode("utf-8", errors="replace")


def _extract_repo(argv: list[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg in REPO_FLAGS and index + 1 < len(argv):
            return argv[index + 1]
        for prefix in ("-R=", "--repo="):
            if arg.startswith(prefix):
                return arg[len(prefix):]
    return None


def validate_argv(argv: Any) -> list[str]:
    if not isinstance(argv, list) or len(argv) < 2:
        raise BrokerError("Denied: expected a gh subcommand and action.")
    if not all(isinstance(arg, str) and arg for arg in argv):
        raise BrokerError("Denied: argv must contain non-empty strings only.")

    subcommand, action = argv[0], argv[1]
    if subcommand == "auth" and action == "status":
        return argv
    if subcommand in DENIED_COMMANDS:
        raise BrokerError(f"Denied: 'gh {subcommand}' is not allowed.")
    if (subcommand, action) not in ALLOWED_COMMANDS:
        raise BrokerError(f"Denied: 'gh {subcommand} {action}' is not an allowed operation.")

    for arg in argv:
        if arg in DENIED_FLAGS or any(arg.startswith(flag + "=") for flag in DENIED_FLAGS):
            raise BrokerError(f"Denied: flag '{arg.split('=', 1)[0]}' is not allowed.")

    repo = _extract_repo(argv)
    if repo is None:
        argv = [*argv, "-R", ALLOWED_REPO]
    elif repo != ALLOWED_REPO:
        raise BrokerError(f"Denied: repo must be {ALLOWED_REPO}.")

    return argv


def _validate_token_file(path: str) -> str:
    stat_result = os.stat(path)
    mode = stat_result.st_mode & 0o777
    if stat_result.st_uid != 0 or stat_result.st_gid != 0 or mode != 0o600:
        raise BrokerError(
            "Token file must be owned by root:root with mode 0600; "
            f"found {stat_result.st_uid}:{stat_result.st_gid}:{mode:o}."
        )
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token:
        raise BrokerError("Token file is empty.")
    return token


def _drop_privileges(user: str = "broker") -> None:
    if os.getuid() != 0:
        return
    pw_record = pwd.getpwnam(user)
    os.setgroups([])
    os.setgid(pw_record.pw_gid)
    os.setuid(pw_record.pw_uid)


def run_gh(argv: list[str], token: str, cwd: str | None, timeout_sec: int) -> dict[str, Any]:
    timeout_sec = max(1, min(timeout_sec, MAX_TIMEOUT_SEC))
    os.makedirs(GH_CONFIG_DIR, mode=0o700, exist_ok=True)
    env = dict(os.environ)
    env["GH_TOKEN"] = token
    env["GH_CONFIG_DIR"] = GH_CONFIG_DIR
    env["HOME"] = str(Path(GH_CONFIG_DIR).parent)
    env.pop("GITHUB_TOKEN", None)

    result = subprocess.run(
        [GH_REAL] + argv,
        cwd=cwd if cwd and os.path.isdir(cwd) else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
        check=False,
    )
    stdout = _limited_text(result.stdout)
    stderr = _limited_text(result.stderr)
    return {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def handle_request(raw: bytes, token: str) -> bytes:
    try:
        request = json.loads(raw.decode("utf-8"))
        argv = validate_argv(request.get("argv"))
        if argv[:2] == ["auth", "status"]:
            return _json_response(
                True,
                0,
                f"github.com\n  Authenticated via Daimon GitHub broker for {ALLOWED_REPO}\n",
                "",
            )
        cwd = request.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise BrokerError("Denied: cwd must be a string.")
        timeout_sec = request.get("timeout_sec", DEFAULT_TIMEOUT_SEC)
        if not isinstance(timeout_sec, int):
            raise BrokerError("Denied: timeout_sec must be an integer.")
        response = run_gh(argv, token, cwd, timeout_sec)
        return _json_response(
            bool(response["ok"]),
            int(response["exit_code"]),
            str(response["stdout"]),
            str(response["stderr"]),
        )
    except BrokerError as exc:
        return _json_response(False, 1, "", str(exc))
    except subprocess.TimeoutExpired:
        return _json_response(False, 124, "", "GitHub command timed out.")
    except Exception:
        return _json_response(False, 1, "", "Broker request failed.")


def serve(host: str = BROKER_HOST, port: int = BROKER_PORT, token_path: str = TOKEN_PATH) -> None:
    token = _validate_token_file(token_path)
    _drop_privileges()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(16)
        while True:
            conn, _addr = server.accept()
            with conn:
                conn.settimeout(5)
                chunks = []
                too_large = False
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if sum(len(part) for part in chunks) > 256_000:
                        conn.sendall(_json_response(False, 1, "", "Denied: request too large."))
                        too_large = True
                        break
                if chunks and not too_large:
                    conn.sendall(handle_request(b"".join(chunks), token))


def main() -> int:
    try:
        serve()
    except BrokerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
