#!/usr/bin/env python3
"""Imgflip free-tier helper.

Documented free-tier endpoints used:
- GET /get_memes
- POST /caption_image

Credentials are read from IMGFLIP_USERNAME and IMGFLIP_PASSWORD. The script never
prints credential values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

API_BASE = "https://api.imgflip.com"


def die(message: str, code: int = 1) -> NoReturn:
    print(json.dumps({"success": False, "error": message}, indent=2))
    raise SystemExit(code)


def http_get_json(url: str, params: dict[str, str] | None = None, timeout: int = 20) -> dict[str, Any]:
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Hermes-Agent-Imgflip-Skill/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except HTTPError as exc:
        die(f"GET failed: HTTP {exc.code}: {exc.reason}")
    except URLError as exc:
        die(f"GET failed: {exc.reason}")
    try:
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        die(f"GET returned non-JSON response: HTTP {status}: {exc}")
    if status != 200:
        die(f"GET failed: HTTP {status}: {data}")
    return data


def http_post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    encoded = urlencode({k: str(v) for k, v in payload.items()}).encode("utf-8")
    request = Request(url, data=encoded, method="POST", headers={"User-Agent": "Hermes-Agent-Imgflip-Skill/1.0"})
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except HTTPError as exc:
        die(f"POST failed: HTTP {exc.code}: {exc.reason}")
    except URLError as exc:
        die(f"POST failed: {exc.reason}")
    try:
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        die(f"POST returned non-JSON response: HTTP {status}: {exc}")
    if status != 200:
        die(f"POST failed: HTTP {status}: {data}")
    return data


def validate_imgflip_image_url(url: str) -> None:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if parsed.scheme != "https" or not (hostname == "imgflip.com" or hostname.endswith(".imgflip.com")):
        die("Imgflip returned an unexpected image URL host; refusing to download it")


def download(url: str, timeout: int = 20) -> tuple[int, str, bytes]:
    validate_imgflip_image_url(url)
    request = Request(url, headers={"User-Agent": "Hermes-Agent-Imgflip-Skill/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.headers.get("content-type", ""), response.read()
    except HTTPError as exc:
        die(f"Image download failed: HTTP {exc.code}: {exc.reason}")
    except URLError as exc:
        die(f"Image download failed: {exc.reason}")


def get_memes(template_type: str = "image") -> list[dict[str, Any]]:
    params = {"type": template_type} if template_type else {}
    data = http_get_json(f"{API_BASE}/get_memes", params=params)
    if not data.get("success"):
        die(f"GET /get_memes failed: {data.get('error_message') or data}")
    return data.get("data", {}).get("memes", [])


def resolve_template(template_id: str | None, template_name: str | None, template_type: str) -> dict[str, Any]:
    memes = get_memes(template_type)
    if template_id:
        for meme in memes:
            if str(meme.get("id")) == str(template_id):
                return meme
        # Known IDs can still work even when they are not in the current top list.
        return {"id": template_id, "name": None, "url": None, "box_count": None}

    if template_name:
        query = template_name.lower()
        for meme in memes:
            if query in meme.get("name", "").lower():
                return meme
        die(
            f"No top-template match for {template_name!r}. "
            "Free tier has no /search_memes; use a known template ID or Premium search."
        )

    die("Provide --template-id or --template-name")


def list_templates(args: argparse.Namespace) -> None:
    memes = get_memes(args.type)
    print(json.dumps({"success": True, "count": len(memes), "templates": memes[: args.limit]}, indent=2))


def load_credentials() -> tuple[str, str]:
    username = os.getenv("IMGFLIP_USERNAME")
    password = os.getenv("IMGFLIP_PASSWORD")
    if not username or not password:
        die("Missing IMGFLIP_USERNAME/IMGFLIP_PASSWORD. Load credentials from environment variables.", code=2)
    return str(username), str(password)


def add_boxes_to_payload(payload: dict[str, Any], boxes_json: str) -> None:
    try:
        boxes = json.loads(boxes_json)
    except json.JSONDecodeError as exc:
        die(f"--boxes must be valid JSON: {exc}")
    if not isinstance(boxes, list):
        die("--boxes must be a JSON list")
    if len(boxes) > 20:
        die("Imgflip currently limits caption_image to 20 text boxes per image")
    for index, box in enumerate(boxes):
        if not isinstance(box, dict) or "text" not in box:
            die("Each box must be an object with at least a text field")
        for key, value in box.items():
            payload[f"boxes[{index}][{key}]"] = value


def make_meme(args: argparse.Namespace) -> None:
    username, password = load_credentials()
    template = resolve_template(args.template_id, args.template_name, "image")
    payload: dict[str, Any] = {
        "template_id": str(template["id"]),
        "username": username,
        "password": password,
    }

    if args.boxes:
        add_boxes_to_payload(payload, args.boxes)
    else:
        if args.text0 is None and args.text1 is None:
            die("Provide --text0/--text1 or --boxes")
        payload["text0"] = args.text0 or ""
        payload["text1"] = args.text1 or ""

    if args.font:
        payload["font"] = args.font
    if args.max_font_size:
        payload["max_font_size"] = str(args.max_font_size)

    data = http_post_json(f"{API_BASE}/caption_image", payload)
    if not data.get("success"):
        die(f"POST /caption_image failed: {data.get('error_message') or data}")

    result = data.get("data", {})
    verification: dict[str, Any] = {}
    if result.get("url") and args.verify_download:
        status, content_type, content = download(result["url"])
        looks_like_image = status == 200 and content_type.startswith("image/")
        verification = {
            "download_status": status,
            "content_type": content_type,
            "bytes": len(content),
            "looks_like_image": looks_like_image,
        }
        if args.output and looks_like_image:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
            verification["saved_to"] = str(output)

    print(
        json.dumps(
            {
                "success": True,
                "template": {key: template.get(key) for key in ("id", "name", "url", "box_count")},
                "url": result.get("url"),
                "page_url": result.get("page_url"),
                "verified": verification,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Imgflip free-tier meme helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List popular captionable templates via GET /get_memes")
    list_parser.add_argument("--type", default="image", choices=["image", "gif", "gif,image", "image,gif"])
    list_parser.add_argument("--limit", type=int, default=10)
    list_parser.set_defaults(func=list_templates)

    make_parser = subparsers.add_parser("make", help="Caption an image template via POST /caption_image")
    template_group = make_parser.add_mutually_exclusive_group(required=True)
    template_group.add_argument("--template-id")
    template_group.add_argument("--template-name")
    make_parser.add_argument("--text0")
    make_parser.add_argument("--text1")
    make_parser.add_argument("--boxes", help='JSON list of box objects, for example: [{"text":"TOP"},{"text":"BOTTOM"}]')
    make_parser.add_argument("--font", default="impact")
    make_parser.add_argument("--max-font-size", type=int)
    make_parser.add_argument("--verify-download", action=argparse.BooleanOptionalAction, default=True)
    make_parser.add_argument("--output", help="Optional local path to save the returned image")
    make_parser.set_defaults(func=make_meme)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
