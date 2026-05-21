#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import requests


STATUS_URL = "https://api.mybrandmetrics.com/tiktok/publishing/status"


def resolve_config_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()

    env_path = os.environ.get("TIKTOK_PUBLISHER_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent.parent
    return workspace_root / "config.json"


def load_config(config_path: Path) -> dict:
    try:
        return json.loads(config_path.read_text())
    except FileNotFoundError:
        print(f"Error: config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Error: could not decode JSON from {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)


def resolve_api_key(config: dict) -> str:
    tiktok_config = config.get("tiktok", {})
    api_key = (
        os.environ.get("TIKTOK_API_KEY")
        or os.environ.get("TIKTOK_AUTHORIZATION_TOKEN")
        or tiktok_config.get("api_key")
        or tiktok_config.get("authorization_token")
    )

    if not api_key:
        print(
            "Error: TikTok API key not found. Provide it via config.json or environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    return api_key


def check_status(publish_id: str, api_key: str) -> None:
    headers = {"X-API-Key": api_key}
    response = requests.get(STATUS_URL, headers=headers, params={"publish_id": publish_id}, timeout=60)

    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
        return

    print(f"Error: HTTP {response.status_code}", file=sys.stderr)
    print(response.text, file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TikTok publishing status")
    parser.add_argument("--publish-id", required=True, help="Publish job ID to inspect")
    parser.add_argument("--config", help="Path to config.json")

    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    api_key = resolve_api_key(config)
    check_status(args.publish_id, api_key)


if __name__ == "__main__":
    main()
