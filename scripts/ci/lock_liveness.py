"""Pin liveness census: report every pinned input whose source has moved on.

Pins are treated as immutable facts; the sources behind them are not. Every
reported install failure in the #125350 class was one projection of that gap:
BtbN pruned a retained-daily ffmpeg tag, the Termux pool rotated two .debs out
of pool/main/, an inbox tar could not decompress a bz2 pin, a wrapper spoke a
switch the script had dropped. Each was fixed as a single artifact; nothing
watched the pipeline's relationship to its suppliers.

The Termux pool already has a repair loop (`hermes pm update --termux`), but it
only runs when a human runs it, and only for the pool rows. The CI archive job
only fires when pm/lock.json or pm/artifact-mirror.json changes. A pin whose
supplier retires it *between* bumps rots silently until a user report.

This module is the missing detector, in the same shape as the
install-e2e-red tracker: a scheduled HEAD census of every pinned HTTP source,
one open issue while anything is dead. It is read-only — it never rewrites a
pin. The repair stays `pm update --termux` and the lock bump, so a census run
can never mutate what it measures.

Consumers:

    python3 -m scripts.ci.lock_liveness --format json     # machine-readable rows
    python3 -m scripts.ci.lock_liveness                   # human summary

Exit status is 0 when every source answers, 1 when at least one is confirmed
dead. Unknown/blocked sources (403, 429, TLS, timeout) are reported but do not
fail the census: a regional edge denial is not evidence that a pin has rotated,
and a red census that cannot be trusted is worse than no census.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# The runner invokes this before setup-pm has installed the checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pm.lock import SCHEMA

# A HEAD is what liveness needs; bodies are never read. The UA follows
# pm/downloader.py: asset CDNs 403 unknown tool UAs from CI runner IP ranges.
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) hermes-pm/1.0"}
DEAD = "dead"
ALIVE = "alive"
UNKNOWN = "unknown"

# Statuses that mean "this exact object is gone": the pin must be repaired.
DEAD_STATUSES = (404, 410)
# Statuses that mean "this vantage point was refused, the object may exist":
# regional edges (Cloudflare), rate limits, and transport failures.
UNKNOWN_STATUSES = (401, 403, 429)


@dataclass(frozen=True)
class Pin:
    """One pinned HTTP input, named the way its owner names it."""

    scope: str  # "lock" (a pm/lock.json row) or "table" (a runtime-lib row)
    name: str  # "<package>@<target>" for lock rows, the table key otherwise
    url: str
    sha256: str
    role: str = "primary"  # "primary" (the supplier URL) or "mirror" (our content-addressed copy)


@dataclass
class Row:
    pin: Pin
    status: str  # alive | dead | unknown
    detail: str = ""


@dataclass
class Census:
    rows: list[Row] = field(default_factory=list)

    @property
    def dead(self) -> list[Row]:
        return [row for row in self.rows if row.status == DEAD]

    @property
    def unknown(self) -> list[Row]:
        return [row for row in self.rows if row.status == UNKNOWN]

    @property
    def alive(self) -> list[Row]:
        return [row for row in self.rows if row.status == ALIVE]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lock_pins(repo: Path, *, target: str | None = None) -> list[Pin]:
    """Every HTTP pin in pm/lock.json, or just one target's rows.

    `docker://` rows are OCI digests owned by a registry, not HTTP objects;
    they are skipped here the same way the archive job skips them.
    """
    lock = _read_json(repo / "pm" / "lock.json")
    if not isinstance(lock, dict) or lock.get("schema") != SCHEMA or not isinstance(lock.get("packages"), dict):
        raise ValueError("Invalid PM lockfile")
    out: list[Pin] = []
    for name, package in sorted(lock["packages"].items()):
        artifacts = package.get("artifacts") or {}
        keys = [target] if target in artifacts else sorted(artifacts)
        for row_target in keys:
            rows = artifacts.get(row_target)
            if rows is None:
                continue
            for row in rows if isinstance(rows, list) else [rows]:
                url = (row or {}).get("url") if isinstance(row, dict) else None
                if not isinstance(url, str) or url.startswith("docker://"):
                    continue
                out.append(Pin("lock", f"{name}@{row_target}", url, str(row.get("sha256") or "")))
    return out


def table_pins(repo: Path, *, target: str | None = None) -> list[Pin]:
    """Every pinned runtime-lib / license HTTP source.

    These are Termux-pool archives consumed by the payload stager; `target`
    only gates whether they are in scope at all (they are bionic-only).
    """
    if target not in (None, "linux-arm64-bionic"):
        return []
    table = _read_json(repo / "pm" / "termux_runtime_libs.json")
    out: list[Pin] = []
    for name, row in sorted((table.get("libs") or {}).items()):
        if isinstance(row, dict) and isinstance(row.get("url"), str):
            out.append(Pin("table", name, row["url"], str(row.get("sha256") or "")))
    licenses = table.get("licenses")
    if isinstance(licenses, dict) and isinstance(licenses.get("url"), str):
        out.append(Pin("table", "termux-licenses", licenses["url"], str(licenses.get("sha256") or "")))
    return out


def pinned_inputs(repo: Path, *, target: str | None = None, include_mirrors: bool = True) -> list[Pin]:
    """Every pin reference, plus (by default) each pin's mirror object.

    The mirror matters: the combination that actually bricks users is a dead
    primary AND a denied/absent mirror. A runner cannot reproduce a user's
    regional denial, but an absent mirror object is unambiguous — it means the
    archive step never seeded this pin, which is itself a defect the census
    should name.
    """
    from pm.artifact_mirror import mirror_url

    out: list[Pin] = []
    for pin in lock_pins(repo, target=target) + table_pins(repo, target=target):
        out.append(pin)
        if include_mirrors and pin.sha256:
            try:
                out.append(Pin(pin.scope, pin.name, mirror_url(pin.sha256), pin.sha256, role="mirror"))
            except ValueError:
                # A malformed digest is the lockfile validator's finding, not
                # the census's; probe the primary, skip the mirror.
                pass
    return out


def probe(url: str, *, timeout: float = 30.0) -> tuple[str, str]:
    """HEAD one URL. Returns (status, detail) — never raises."""
    request = urllib.request.Request(url, headers=_UA, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https pins are validated upstream)
            code = response.status
            if 200 <= code < 400:
                return ALIVE, f"HTTP {code}"
            if code in DEAD_STATUSES:
                return DEAD, f"HTTP {code}"
            return UNKNOWN, f"HTTP {code}"
    except urllib.error.HTTPError as exc:
        if exc.code in DEAD_STATUSES:
            return DEAD, f"HTTP {exc.code}"
        return UNKNOWN, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - transport classification, never fatal
        return UNKNOWN, f"{type(exc).__name__}: {exc}"


def census(repo: Path, *, target: str | None = None, include_mirrors: bool = True, workers: int = 16) -> Census:
    pins = pinned_inputs(repo, target=target, include_mirrors=include_mirrors)
    result = Census()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe, pin.url): pin for pin in pins}
        for future in as_completed(futures):
            pin = futures[future]
            status, detail = future.result()
            result.rows.append(Row(pin, status, detail))
    result.rows.sort(key=lambda row: (row.status != DEAD, row.status != UNKNOWN, row.pin.scope, row.pin.role, row.pin.name))
    return result


def render(census_result: Census) -> str:
    lines = [
        f"pin liveness: {len(census_result.alive)} alive / {len(census_result.dead)} dead / "
        f"{len(census_result.unknown)} unknown of {len(census_result.rows)} pinned sources",
    ]
    for row in census_result.dead:
        lines.append(f"  DEAD    {row.pin.scope:5} {row.pin.role:7} {row.pin.name:32} {row.detail:8} {row.pin.url}")
    for row in census_result.unknown:
        lines.append(f"  UNKNOWN {row.pin.scope:5} {row.pin.role:7} {row.pin.name:32} {row.detail:8} {row.pin.url}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--target", default=None, help="restrict lock rows to one target (tables stay as scoped)")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    result = census(args.repo.resolve(), target=args.target)
    if args.format == "json":
        payload = {
            "alive": len(result.alive),
            "dead": len(result.dead),
            "unknown": len(result.unknown),
            "total": len(result.rows),
            "rows": [
                {"scope": row.pin.scope, "name": row.pin.name, "role": row.pin.role, "url": row.pin.url,
                 "status": row.status, "detail": row.detail}
                for row in result.rows
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render(result))
    return 1 if result.dead else 0


if __name__ == "__main__":
    raise SystemExit(main())
