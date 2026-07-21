#!/usr/bin/env python3
"""Fetch a MyBrandMetrics Google token and execute gws."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

TOKEN_ENDPOINT = "https://api.mybrandmetrics.com/internal/token/access"
SOURCE_KEY_MAP = {
    "calendar": "google_calendar",
    "sheets": "google_sheets",
    "drive": "google_drive",
}


def load_api_key() -> str | None:
    api_key = os.environ.get("GWS_SKILL_API_KEY")
    if api_key:
        return api_key.strip()

    key_path = Path.home() / ".google_workspace_api_key"
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    return None


def get_token(source_key: str, api_key: str, timeout: int = 15) -> str:
    payload = json.dumps({"source_key": source_key}).encode("utf-8")
    request = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-API_KEY": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(
            f"ERROR: Token request failed with HTTP {exc.code}: {detail}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"ERROR: Token request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print("ERROR: Token endpoint returned non-JSON response.", file=sys.stderr)
        raise SystemExit(1)

    token = data.get("access_token")
    if not token:
        print("ERROR: Token endpoint response did not include access_token.", file=sys.stderr)
        raise SystemExit(1)
    return token


def resolve_gws_binary() -> str:
    found = shutil.which("gws")
    if found:
        return found

    local_binary = Path(__file__).with_name("gws_musl")
    if local_binary.exists():
        return str(local_binary)

    print(
        "ERROR: gws was not found. Run scripts/install_gws.sh first.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run gws with a MyBrandMetrics Google access token.",
    )
    parser.add_argument("service", choices=sorted(SOURCE_KEY_MAP))
    parser.add_argument("gws_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.gws_args:
        parser.error("missing gws resource/method arguments")

    api_key = load_api_key()
    if not api_key:
        print(
            "REQUIRED_ACTION: Provide GWS_SKILL_API_KEY or save it to "
            "~/.google_workspace_api_key.",
            file=sys.stderr,
        )
        return 1

    token = get_token(SOURCE_KEY_MAP[args.service], api_key)
    env = os.environ.copy()
    env["GOOGLE_WORKSPACE_CLI_TOKEN"] = token

    cmd = [resolve_gws_binary(), args.service, *args.gws_args]
    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
