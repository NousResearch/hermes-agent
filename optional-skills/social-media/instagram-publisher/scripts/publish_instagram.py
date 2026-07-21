#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests


BASE_URL = "https://api.mybrandmetrics.com/instagram/publishing"


def resolve_config_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()

    env_path = os.environ.get("INSTAGRAM_PUBLISH_CONFIG")
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


def resolve_credentials(args: argparse.Namespace, config: dict) -> tuple[str, str, str]:
    instagram_config = config.get("instagram", {})
    api_key = (
        args.api_key
        or os.environ.get("INSTAGRAM_API_KEY")
        or os.environ.get("INSTAGRAM_AUTHORIZATION_TOKEN")
        or instagram_config.get("api_key")
        or instagram_config.get("authorization_token")
    )
    connection_id = (
        args.connection_id
        or os.environ.get("INSTAGRAM_CONNECTION_ID")
        or instagram_config.get("connection_id")
    )
    account_id = (
        args.account_id
        or os.environ.get("INSTAGRAM_ACCOUNT_ID")
        or instagram_config.get("account_id")
    )

    if not all([api_key, connection_id, account_id]):
        print(
            "Error: missing Instagram credentials. Provide API key, connection_id, and "
            "account_id via arguments, environment variables, or config.json.",
            file=sys.stderr,
        )
        sys.exit(1)

    return api_key, connection_id, account_id


class InstagramPublisher:
    def __init__(self, api_key: str, connection_id: str, account_id: str) -> None:
        self.api_key = api_key
        self.connection_id = connection_id
        self.account_id = account_id
        self.headers = {"X-API_KEY": self.api_key}

    def _post(self, endpoint: str, data=None, files=None, is_json: bool = True):
        url = f"{BASE_URL}/{endpoint}"
        headers = self.headers.copy()
        if is_json:
            return requests.post(url, headers=headers, json=data, timeout=300)
        data_str = {k: str(v) for k, v in data.items() if v is not None}
        return requests.post(url, headers=headers, data=data_str, files=files, timeout=300)

    def _download_to_tempfile(self, url: str) -> Path:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        suffix = Path(url.split("?")[0]).suffix or ".bin"
        fd, temp_path = tempfile.mkstemp(prefix="instagram_publish_", suffix=suffix)
        os.close(fd)
        temp_file = Path(temp_path)
        with temp_file.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
        return temp_file

    def upload_media_item(self, media_type: str, url: str | None = None, path: str | None = None, thumb_offset: int = 1000, wait: bool = True):
        temp_path = None
        try:
            if url:
                temp_path = self._download_to_tempfile(url)
                path = str(temp_path)

            if not path or not Path(path).exists():
                print("Error: no valid local file was available for upload.", file=sys.stderr)
                return None

            data = {
                "connection_id": self.connection_id,
                "account_id": self.account_id,
                "media_type": media_type,
                "is_carousel_item": "true",
                "wait_for_finished": "true" if wait else "false",
            }
            if media_type == "VIDEO":
                data["thumb_offset"] = str(thumb_offset)

            with Path(path).open("rb") as handle:
                files = {"file": (Path(path).name, handle)}
                response = self._post("media", data=data, files=files, is_json=False)

            if response.status_code == 200:
                result = response.json()
                return result.get("creation_id") or result.get("id") or result.get("media_id")

            print(f"Error uploading carousel item: HTTP {response.status_code}", file=sys.stderr)
            print(response.text, file=sys.stderr)
            return None
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    def publish_once(
        self,
        media_type: str,
        caption: str,
        url: str | None = None,
        path: str | None = None,
        children=None,
        share_to_feed: bool = True,
        thumb_offset: int = 1000,
        wait: bool = True,
        retries: int = 2,
    ):
        data = {
            "connection_id": self.connection_id,
            "account_id": self.account_id,
            "media_type": media_type,
            "caption": caption,
            "wait_for_finished": wait,
        }

        if media_type == "CAROUSEL":
            if not children:
                raise ValueError("Children IDs are required for CAROUSEL")
            data["children"] = children
            is_json = True
        elif path:
            source_path = Path(path).expanduser()
            if not source_path.exists():
                raise FileNotFoundError(f"File not found: {source_path}")
            is_json = False
            data["wait_for_finished"] = "true" if wait else "false"
            if media_type == "REELS":
                data["share_to_feed"] = "true" if share_to_feed else "false"
                data["thumb_offset"] = str(thumb_offset)
        elif url:
            is_json = True
            if media_type == "REELS":
                data["video_url"] = url
                data["share_to_feed"] = share_to_feed
                data["thumb_offset"] = thumb_offset
            else:
                data["image_url"] = url
        else:
            raise ValueError("Provide either url, path, or children.")

        for attempt in range(retries + 1):
            if is_json:
                response = self._post("publish-once", data=data, is_json=True)
            else:
                with Path(path).open("rb") as handle:
                    files = {"file": (Path(path).name, handle)}
                    response = self._post("publish-once", data=data, files=files, is_json=False)

            if response.status_code == 200:
                return response.json()

            if response.status_code in {500, 502, 503, 504} or (
                response.status_code == 400 and "unexpected" in response.text.lower()
            ):
                if attempt < retries:
                    time.sleep(30 * (attempt + 1))
                    continue

            print(f"Error publishing: HTTP {response.status_code}", file=sys.stderr)
            print(response.text, file=sys.stderr)
            return None

        return None

    def check_status(self, publish_id: str):
        response = requests.get(
            f"{BASE_URL}/status",
            headers=self.headers,
            params={
                "publish_id": publish_id,
                "connection_id": self.connection_id,
                "account_id": self.account_id,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        print(f"Error checking status: HTTP {response.status_code}", file=sys.stderr)
        print(response.text, file=sys.stderr)
        return None


def detect_media_type(item: str) -> str:
    lowered = item.lower()
    ext = Path(item.split("?")[0]).suffix.lower()
    if ext in {".mp4", ".mov", ".avi", ".m4v"} or "video" in lowered or "mp4" in lowered:
        return "VIDEO"
    return "IMAGE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish to Instagram via MyBrandMetrics API")
    parser.add_argument("--api-key", help="MyBrandMetrics API key")
    parser.add_argument("--connection-id", help="Instagram connection ID")
    parser.add_argument("--account-id", help="MyBrandMetrics account ID")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--type", choices=["IMAGE", "REELS", "CAROUSEL"], default="IMAGE")
    parser.add_argument("--caption", default="", help="Post caption")
    parser.add_argument("--url", help="Media URL for single image or reel")
    parser.add_argument("--path", help="Local file path for single image or reel")
    parser.add_argument("--items", nargs="+", help="List of URLs or paths for carousel items")
    parser.add_argument("--no-feed", action="store_false", dest="share_to_feed", help="Do not share reels to the feed")
    parser.add_argument("--thumb-offset", type=int, default=1000, help="Thumbnail offset in milliseconds")
    parser.add_argument("--no-wait", action="store_false", dest="wait", help="Do not wait for completion")
    parser.add_argument("--check-id", help="Check the status of an existing publish ID")
    args = parser.parse_args()

    config = load_config(resolve_config_path(args.config))
    api_key, connection_id, account_id = resolve_credentials(args, config)
    publisher = InstagramPublisher(api_key, connection_id, account_id)

    if args.check_id:
        status = publisher.check_status(args.check_id)
        if status is not None:
            print(json.dumps(status, indent=2))
            return
        sys.exit(1)

    if args.type == "CAROUSEL":
        if not args.items:
            print("Error: --items is required for CAROUSEL", file=sys.stderr)
            sys.exit(1)

        children_ids = []
        for item in args.items:
            media_type = detect_media_type(item)
            if item.startswith(("http://", "https://")):
                child_id = publisher.upload_media_item(media_type, url=item, thumb_offset=args.thumb_offset, wait=args.wait)
            else:
                child_id = publisher.upload_media_item(media_type, path=item, thumb_offset=args.thumb_offset, wait=args.wait)
            if not child_id:
                print("Error: failed to upload at least one carousel item.", file=sys.stderr)
                sys.exit(1)
            children_ids.append(child_id)

        has_video = any(detect_media_type(item) == "VIDEO" for item in args.items)
        time.sleep(90 if has_video else 30)
        result = publisher.publish_once("CAROUSEL", args.caption, children=children_ids, wait=args.wait)
    else:
        result = publisher.publish_once(
            args.type,
            args.caption,
            url=args.url,
            path=args.path,
            share_to_feed=args.share_to_feed,
            thumb_offset=args.thumb_offset,
            wait=args.wait,
        )

    if result is None:
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
