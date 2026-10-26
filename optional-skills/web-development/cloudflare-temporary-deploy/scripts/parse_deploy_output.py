#!/usr/bin/env python3
"""Parse `wrangler deploy --temporary` output into structured JSON.

Reads wrangler's stdout/stderr from STDIN and extracts the live workers.dev
URL, the claim URL, the temporary account name/state, the claim window, and
whether a deploy actually happened. Stdlib only — no dependencies.

Usage:
    npx wrangler@latest deploy --temporary 2>&1 | python3 parse_deploy_output.py
    python3 parse_deploy_output.py --selftest
"""

from __future__ import annotations

import json
import re
import sys

# Match the live workers.dev URL (subdomain.subdomain.workers.dev).
_LIVE_URL = re.compile(r"https://[A-Za-z0-9._-]+\.workers\.dev\S*")
# Match the claim URL. Cloudflare uses dash.cloudflare.com/claim-preview?claimToken=...
# Keep it broad enough to survive minor path changes while still requiring a claim token.
_CLAIM_URL = re.compile(r"https://\S*claim\S*claimToken=\S+", re.IGNORECASE)
# "Account: Serene Temple (created)"  /  "Account:  example-name (reused)"
# Account names can contain spaces (e.g. "Serene Temple"), so capture everything
# up to the trailing "(state)" marker rather than a single token.
_ACCOUNT = re.compile(
    r"Account:\s*(?P<name>.+?)\s*\((?P<state>created|reused)\)", re.IGNORECASE
)
# "Claim within:   60 minutes"
_CLAIM_WITHIN = re.compile(r"Claim within:\s*(?P<minutes>\d+)\s*minutes?", re.IGNORECASE)
# A successful deploy prints a "Deployed" / "Uploaded" line.
_DEPLOYED = re.compile(r"^\s*(Deployed|Uploaded)\b", re.IGNORECASE | re.MULTILINE)


def _first(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    if not m:
        return None
    # Strip trailing punctuation that often clings to a URL in log lines.
    return m.group(0).rstrip(".,);]")


def parse(text: str) -> dict:
    """Extract deploy facts from wrangler output text."""
    account = _ACCOUNT.search(text)
    claim_within = _CLAIM_WITHIN.search(text)
    return {
        "live_url": _first(_LIVE_URL, text),
        "claim_url": _first(_CLAIM_URL, text),
        "account": account.group("name") if account else None,
        "account_state": account.group("state").lower() if account else None,
        "expires_minutes": int(claim_within.group("minutes")) if claim_within else None,
        "deployed": bool(_DEPLOYED.search(text)),
    }


_SAMPLE = """\
Continuing means you accept Cloudflare's Terms of Service and Privacy Policy.

Temporary account ready:
     Account:        example-name (created)
     Claim within:   60 minutes
     Claim URL:      https://dash.cloudflare.com/claim-preview?claimToken=abc123XYZ

Uploaded example-worker
Deployed example-worker triggers
     https://example-worker.example-name.workers.dev
"""

_SAMPLE_REUSED = """\
Temporary account ready:
     Account:        example-name (reused)
     Claim within:   42 minutes
     Claim URL:      https://dash.cloudflare.com/claim-preview?claimToken=def456
Deployed example-worker triggers
     https://example-worker.example-name.workers.dev
"""

_SAMPLE_NO_TEMP = """\
✘ [ERROR] You are not logged in.

To continue without logging in, rerun this command with `--temporary`.
"""


def _selftest() -> int:
    r = parse(_SAMPLE)
    assert r["live_url"] == "https://example-worker.example-name.workers.dev", r
    assert r["claim_url"] == "https://dash.cloudflare.com/claim-preview?claimToken=abc123XYZ", r
    assert r["account"] == "example-name", r
    assert r["account_state"] == "created", r
    assert r["expires_minutes"] == 60, r
    assert r["deployed"] is True, r

    r2 = parse(_SAMPLE_REUSED)
    assert r2["account_state"] == "reused", r2
    assert r2["expires_minutes"] == 42, r2
    assert r2["deployed"] is True, r2

    r3 = parse(_SAMPLE_NO_TEMP)
    assert r3["live_url"] is None, r3
    assert r3["claim_url"] is None, r3
    assert r3["account"] is None, r3
    assert r3["deployed"] is False, r3

    print("selftest: OK")
    return 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    text = sys.stdin.read()
    result = parse(text)
    print(json.dumps(result, indent=2))
    # Non-zero exit if no live URL was found, so callers can branch on it.
    return 0 if result["live_url"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
