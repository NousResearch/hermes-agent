#!/usr/bin/env python3
"""Print the first unambiguous GitHub token in a git credential-store file."""

from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


_TOKEN_PREFIXES = ("ghp_", "github_pat_", "gho_", "ghu_", "ghs_", "ghr_")
_INVALID_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def _decode(value: str | None) -> str:
    if value is None or _INVALID_ESCAPE.search(value):
        return ""
    decoded = unquote(value)
    if not decoded or any(ord(char) <= 0x1F or 0x7F <= ord(char) <= 0x9F for char in decoded):
        return ""
    return decoded


def _token_from_url(line: str) -> str:
    if "\r" in line or "\n" in line:
        return ""
    try:
        credential = urlsplit(line)
        port = credential.port
    except ValueError:
        return ""
    if credential.scheme != "https" or credential.hostname != "github.com" or port not in (None, 443):
        return ""

    username = _decode(credential.username)
    password = _decode(credential.password)
    if not username:
        return ""
    if password and password != "x-oauth-basic":
        return password
    if password == "x-oauth-basic":
        return username
    return username if username.startswith(_TOKEN_PREFIXES) else ""


def main() -> int:
    path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / ".git-credentials"
    try:
        lines = path.read_bytes().split(b"\n")
    except OSError:
        return 1

    for raw_line in lines:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            continue
        token = _token_from_url(line)
        if token:
            print(token)
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
