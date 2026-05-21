#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import requests


API_URL = "https://api.mybrandmetrics.com/tiktok/publishing/post"


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


def build_headers(api_key: str) -> dict:
    return {
        "X-API-Key": api_key,
        "X-API_KEY": api_key,
    }


def publish_tiktok(args: argparse.Namespace, api_key: str) -> None:
    is_url = args.source.startswith(("http://", "https://"))
    headers = build_headers(api_key)

    if is_url:
        headers["Content-Type"] = "application/json"
        payload = {
            "title": args.title,
            "video_url": args.source,
            "privacy_level": args.privacy_level,
            "wait_for_published": args.wait_for_published,
        }
        if args.poll_interval is not None:
            payload["poll_interval_ms"] = args.poll_interval
        if args.poll_timeout is not None:
            payload["poll_timeout_ms"] = args.poll_timeout

        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
    else:
        source_path = Path(args.source).expanduser()
        if not source_path.exists():
            print(f"Error: local file not found at {source_path}", file=sys.stderr)
            sys.exit(1)

        data = {
            "title": args.title,
            "privacy_level": args.privacy_level,
            "wait_for_published": str(args.wait_for_published).lower(),
        }
        if args.poll_interval is not None:
            data["poll_interval_ms"] = str(args.poll_interval)
        if args.poll_timeout is not None:
            data["poll_timeout_ms"] = str(args.poll_timeout)

        with source_path.open("rb") as handle:
            files = {"file": handle}
            response = requests.post(API_URL, headers=headers, files=files, data=data, timeout=300)

    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
        return

    print(f"Error: HTTP {response.status_code}", file=sys.stderr)
    print(response.text, file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a TikTok video via MyBrandMetrics API")
    parser.add_argument("--source", required=True, help="Direct video URL or local file path")
    parser.add_argument("--title", required=True, help="Title of the TikTok post")
    parser.add_argument("--privacy-level", default="SELF_ONLY", help="Privacy level such as PUBLIC or SELF_ONLY")
    parser.add_argument("--wait-for-published", action="store_true", help="Wait for the publish job to complete")
    parser.add_argument("--poll-interval", type=int, help="Polling interval in milliseconds")
    parser.add_argument("--poll-timeout", type=int, help="Polling timeout in milliseconds")
    parser.add_argument("--config", help="Path to config.json")

    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    api_key = resolve_api_key(config)
    publish_tiktok(args, api_key)


if __name__ == "__main__":
    main()
