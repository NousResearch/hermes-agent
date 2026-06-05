#!/usr/bin/env python3
"""
ci_recon.py — stdlib-only helper for the competitive-intelligence skill.

Subcommands
-----------
validate <url>
    Check HTTP accessibility and robots.txt compliance.
    Prints a JSON status object. Exits 0 if accessible and allowed, 1 otherwise.

save_report <slug>
    Read a markdown report from stdin and save it atomically under
    ~/.hermes/competitive-intelligence/<slug>/<ISO-date>/.
    Prints the saved path to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path


_TIMEOUT = 10
_USER_AGENT = "HermesAgent/1.0 (+https://github.com/NousResearch/hermes-agent)"


def _hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME", "")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


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

    # 1. Check accessibility
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            result["status_code"] = resp.status
            result["accessible"] = True
    except urllib.error.HTTPError as exc:
        result["status_code"] = exc.code
        # Treat redirect targets as accessible; treat 4xx/5xx as failures
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

    # 2. Check robots.txt
    from urllib.parse import urlparse
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        allowed = rp.can_fetch(_USER_AGENT, url)
        # Also check the wildcard agent
        if not allowed:
            allowed = rp.can_fetch("*", url)
        result["robots_allows_crawl"] = bool(allowed)
    except Exception:  # noqa: BLE001
        # If robots.txt is unreachable, assume crawling is allowed (common convention)
        result["robots_allows_crawl"] = True

    print(json.dumps(result, indent=2))
    return 0 if result["robots_allows_crawl"] else 1


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def cmd_save_report(slug: str) -> int:
    report_md = sys.stdin.read()
    if not report_md.strip():
        print("ERROR: no report content received on stdin", file=sys.stderr)
        return 1

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = _hermes_home() / "competitive-intelligence" / slug / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.md"
    meta_path = out_dir / "meta.json"

    meta = {
        "slug": slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_urls": [],
    }

    _atomic_write(report_path, report_md)
    _atomic_write(meta_path, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(str(report_path))
    return 0


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
            print("Usage: ci_recon.py validate <url>", file=sys.stderr)
            return 1
        return cmd_validate(args[1])

    if subcommand == "save_report":
        if len(args) < 2:
            print("Usage: ci_recon.py save_report <slug>", file=sys.stderr)
            return 1
        return cmd_save_report(args[1])

    print(f"Unknown subcommand: {subcommand!r}", file=sys.stderr)
    print("Available: validate, save_report", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
