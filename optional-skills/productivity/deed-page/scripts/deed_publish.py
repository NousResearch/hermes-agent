#!/usr/bin/env python3
"""Publish a file or folder to deed.page, or delete a site. Stdlib only.

Usage:
  deed_publish.py PATH --slug SLUG [--ttl SECONDS] [--spa]
  deed_publish.py --delete SLUG

Reads DEEDPAGE_TOKEN (optional for a first publish), DEED_REF (default
skill-hermes), and DEEDPAGE_URL (default https://deed.page). Prints one JSON
object. The token is never printed except in the API's first-publish response.
"""
import argparse
import io
import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def base_url() -> str:
    return os.environ.get("DEEDPAGE_URL", "https://deed.page").rstrip("/")


def build_body(path: Path):
    """Return (bytes, content_type) for a file or a folder tarball."""
    if path.is_dir():
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    tar.add(child, arcname=child.relative_to(path).as_posix())
        return buf.getvalue(), "application/gzip"
    data = path.read_bytes()
    if path.suffix.lower() in (".html", ".htm"):
        return data, "text/html; charset=utf-8"
    return data, "application/octet-stream"


def request(method: str, url: str, body=None, headers=None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        try:
            payload = json.loads(err.read() or b"{}")
        except ValueError:
            payload = {"ok": False, "error": "http_error"}
        return err.code, payload


def auth_headers():
    token = os.environ.get("DEEDPAGE_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def publish(path: Path, slug: str, ttl=None, spa=False):
    if not SLUG_RE.match(slug):
        return 400, {"ok": False, "error": "slug_invalid"}
    body, ctype = build_body(path)
    query = urllib.parse.urlencode({"slug": slug, "ref": os.environ.get("DEED_REF", "skill-hermes")})
    headers = {"Content-Type": ctype, "X-Client": "Hermes", **auth_headers()}
    if path.is_file() and ctype == "application/octet-stream":
        headers["Content-Disposition"] = f'attachment; filename="{path.name}"'
    if ttl:
        headers["X-Ttl"] = str(int(ttl))
    if spa:
        headers["X-Spa"] = "1"
    return request("PUT", f"{base_url()}/v1/sites?{query}", body, headers)


def delete(slug: str):
    if not os.environ.get("DEEDPAGE_TOKEN"):
        return 401, {"ok": False, "error": "DEEDPAGE_TOKEN is not set"}
    return request("DELETE", f"{base_url()}/v1/sites/{urllib.parse.quote(slug)}", None, auth_headers())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Publish to deed.page")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--slug")
    ap.add_argument("--ttl", type=int)
    ap.add_argument("--spa", action="store_true")
    ap.add_argument("--delete", metavar="SLUG")
    args = ap.parse_args(argv)
    if args.delete:
        status, payload = delete(args.delete)
    else:
        if not args.path or not args.slug:
            ap.error("PATH and --slug are required to publish")
        status, payload = publish(Path(args.path), args.slug, args.ttl, args.spa)
    print(json.dumps(payload))
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
