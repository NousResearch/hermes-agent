#!/usr/bin/env python3
"""
scrape_helper.py — stdlib-only helper for the web-price-scraper skill.

Subcommands
-----------
validate <url>
    Check HTTP accessibility and robots.txt compliance.
    Prints a JSON status object. Exits 0 if accessible and allowed, 1 otherwise.

save <url> <json-file>
    Read JSON result data from stdin (or from <json-file> path) and save it
    atomically under ~/.hermes/web-price-scraper/<domain>/<ISO-datetime>/.
    Prints the saved path to stdout.

list [domain]
    List previously saved scrape results, optionally filtered by domain.
    Prints a JSON array of result metadata.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path


_TIMEOUT = 15
_USER_AGENT = "HermesAgent/1.0 (+https://github.com/NousResearch/hermes-agent)"


def _hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME", "")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", text)[:80]


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def cmd_validate(url: str) -> int:
    result: dict = {
        "url": url,
        "accessible": False,
        "robots_allows_crawl": False,
        "status_code": None,
        "error": None,
    }

    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            result["status_code"] = resp.status
            result["accessible"] = True
    except urllib.error.HTTPError as exc:
        result["status_code"] = exc.code
        if exc.code in (301, 302, 303, 307, 308):
            result["accessible"] = True
        else:
            result["error"] = f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        result["error"] = str(exc.reason)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    if not result["accessible"]:
        print(json.dumps(result, indent=2))
        return 1

    parsed = urllib.parse.urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        allowed = rp.can_fetch(_USER_AGENT, url)
        if not allowed:
            allowed = rp.can_fetch("*", url)
        result["robots_allows_crawl"] = bool(allowed)
    except Exception:  # noqa: BLE001
        result["robots_allows_crawl"] = True

    print(json.dumps(result, indent=2))
    return 0 if result["robots_allows_crawl"] else 1


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

def cmd_save(url: str) -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print("ERROR: no JSON data received on stdin", file=sys.stderr)
        return 1

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON on stdin: {exc}", file=sys.stderr)
        return 1

    parsed = urllib.parse.urlparse(url)
    domain_slug = _safe_slug(parsed.netloc or "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    out_dir = _hermes_home() / "web-price-scraper" / domain_slug / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "result.json"
    meta = {
        "url": url,
        "domain": parsed.netloc,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "item_count": len(data) if isinstance(data, list) else 1,
    }
    meta_path = out_dir / "meta.json"

    _atomic_write(result_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    _atomic_write(meta_path, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(str(result_path))
    return 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(domain_filter: str | None = None) -> int:
    base = _hermes_home() / "web-price-scraper"
    if not base.exists():
        print("[]")
        return 0

    entries = []
    for meta_file in sorted(base.rglob("meta.json"), reverse=True):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if domain_filter and domain_filter not in meta.get("domain", ""):
                continue
            meta["result_path"] = str(meta_file.parent / "result.json")
            entries.append(meta)
        except Exception:  # noqa: BLE001
            continue

    print(json.dumps(entries, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, content: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    subcommand = args[0]

    if subcommand == "validate":
        if len(args) < 2:
            print("Usage: scrape_helper.py validate <url>", file=sys.stderr)
            return 1
        return cmd_validate(args[1])

    if subcommand == "save":
        if len(args) < 2:
            print("Usage: scrape_helper.py save <url>  (JSON data on stdin)", file=sys.stderr)
            return 1
        return cmd_save(args[1])

    if subcommand == "list":
        domain = args[1] if len(args) > 1 else None
        return cmd_list(domain)

    print(f"Unknown subcommand: {subcommand!r}", file=sys.stderr)
    print("Available: validate, save, list", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
