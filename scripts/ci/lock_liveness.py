"""Report source availability separately from pin recoverability.

A pin is dead only when every source in its verified-download ladder is dead.
Unknown observations never establish recovery. This HEAD census measures
availability, not byte integrity; PM verifies SHA256 during actual downloads.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

# The runner invokes this before setup-pm has installed the checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pm.lock import SCHEMA

# HEAD establishes availability only. A different body can still fail PM's hash gate.
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) hermes-pm/1.0"}
DEAD, ALIVE, UNKNOWN = "dead", "alive", "unknown"
DEAD_STATUSES = (404, 410)


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
    inventory: list[Pin] | None = None

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


def _valid_https_url(url: str) -> bool:
    if not isinstance(url, str) or any(char.isspace() for char in url):
        return False
    try:
        parsed = urlsplit(url)
        parsed.port
        return (parsed.scheme == "https" and bool(parsed.hostname) and parsed.username is None
                and parsed.password is None and not parsed.fragment)
    except ValueError:
        return False


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
        if not isinstance(package, dict) or not isinstance(package.get("artifacts"), dict) or not package["artifacts"]:
            raise ValueError(f"Invalid package inventory: {name}")
        artifacts = package["artifacts"]
        keys = ([target] if target in artifacts else ["any"] if "any" in artifacts else []) if target is not None else sorted(artifacts)
        for row_target in keys:
            rows = artifacts.get(row_target)
            if rows is None or rows == []:
                raise ValueError(f"Empty artifact inventory: {name}@{row_target}")
            for row in rows if isinstance(rows, list) else [rows]:
                url = (row or {}).get("url") if isinstance(row, dict) else None
                if isinstance(url, str) and url.startswith("docker://"):
                    continue
                if not _valid_https_url(url):
                    raise ValueError(f"Invalid HTTP pin: {name}@{row_target}")
                out.append(Pin("lock", f"{name}@{row_target}", url, row.get("sha256")))
    return out


def table_pins(repo: Path, *, target: str | None = None) -> list[Pin]:
    """Every pinned runtime-lib / license HTTP source.

    These are Termux-pool archives consumed by the payload stager; `target`
    only gates whether they are in scope at all (they are bionic-only).
    """
    if target not in (None, "linux-arm64-bionic"):
        return []
    table = _read_json(repo / "pm" / "termux_runtime_libs.json")
    if not isinstance(table, dict) or not isinstance(table.get("libs"), dict):
        raise ValueError("Invalid runtime-lib inventory")
    entries = dict(table["libs"])
    if "licenses" in table:
        entries["termux-licenses"] = table["licenses"]
    out: list[Pin] = []
    for name, row in sorted(entries.items()):
        if not isinstance(row, dict) or not _valid_https_url(row.get("url")):
            raise ValueError(f"Invalid runtime-lib pin: {name}")
        out.append(Pin("table", name, row["url"], row.get("sha256")))
    return out


def pinned_inputs(repo: Path, *, target: str | None = None, include_mirrors: bool = True) -> list[Pin]:
    """Every owner and digest, with the same ladder as pinned_source()."""
    from pm.artifact_mirror import historical_url, object_key, public_prefix

    layout = _read_json(repo / "pm" / "artifact-mirror.json")
    mirror_prefix = public_prefix(layout)
    out: list[Pin] = []
    for pin in lock_pins(repo, target=target) + table_pins(repo, target=target):
        object_key(pin.sha256)  # Invalid inventory must not produce a reassuring partial report.
        out.append(pin)
        if include_mirrors:
            out.append(Pin(pin.scope, pin.name, mirror_prefix + pin.sha256, pin.sha256, role="mirror"))
            historical = historical_url(pin.url)
            if historical:
                out.append(Pin(pin.scope, pin.name, historical, pin.sha256, role="historical"))
    return list(dict.fromkeys(out))


def probe(url: str, *, timeout: float = 30.0) -> tuple[str, str]:
    """HEAD one URL. Returns (status, detail) — never raises."""
    request = urllib.request.Request(url, headers=_UA, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https pins are validated upstream)
            code = response.status
            if code == 200:
                return ALIVE, f"HTTP {code}"
            if code in DEAD_STATUSES:
                return DEAD, f"HTTP {code}"
            return UNKNOWN, f"HTTP {code}"
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        return (DEAD if code in DEAD_STATUSES else UNKNOWN), f"HTTP {code}"
    except Exception as exc:  # noqa: BLE001 - transport classification, never fatal
        return UNKNOWN, f"{type(exc).__name__}: {exc}"


def census(repo: Path, *, target: str | None = None, include_mirrors: bool = True, workers: int = 16) -> Census:
    pins = pinned_inputs(repo, target=target, include_mirrors=include_mirrors)
    result = Census(inventory=pins)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # A digest can have many consumers. Probe the URL once, retain every owner.
        owners: dict[str, list[Pin]] = {}
        for pin in pins:
            owners.setdefault(pin.url, []).append(pin)
        futures = {pool.submit(probe, url): url for url in owners}
        for future in as_completed(futures):
            status, detail = future.result()
            result.rows.extend(Row(pin, status, detail) for pin in owners[futures[future]])
    result.rows.sort(key=lambda row: (row.status != DEAD, row.status != UNKNOWN, row.pin.scope, row.pin.role, row.pin.name, row.pin.sha256, row.pin.url))
    return result


def pin_groups(rows: list[dict]) -> list[dict]:
    """Identity includes the digest: one target may contain several artifacts."""
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["scope"], row["name"], row["sha256"])
        grouped.setdefault(key, []).append(row)
    groups = []
    for (scope, name, digest), sources in sorted(grouped.items()):
        statuses = {source["status"] for source in sources}
        status = ALIVE if ALIVE in statuses else UNKNOWN if UNKNOWN in statuses else DEAD
        groups.append({"scope": scope, "name": name, "sha256": digest, "status": status,
                       "sources": sources})
    return groups


def report(result: Census, *, complete: bool = True) -> dict:
    rows = [{**asdict(row.pin), "status": row.status, "detail": row.detail} for row in result.rows]
    return {
        "schema": 2, "complete": complete,
        "alive": len(result.alive), "dead": len(result.dead), "unknown": len(result.unknown),
        "total": len(rows),
        # The inventory receipt catches dropped, duplicated, or retargeted observations.
        "inventory": [asdict(pin) for pin in (result.inventory if result.inventory is not None else [row.pin for row in result.rows])],
        "rows": rows,
    }


def validate_report(payload: dict, *, expected_inventory: list[Pin] | None = None) -> list[dict]:
    """Refuse absent, incomplete or inconsistent input before issue mutation."""
    from pm.artifact_mirror import object_key

    if not isinstance(payload, dict) or payload.get("schema") != 2 or payload.get("complete") is not True:
        raise ValueError("Tracker requires a complete schema-2 census")
    rows, inventory = payload.get("rows"), payload.get("inventory")
    if not isinstance(rows, list) or not rows or not isinstance(inventory, list):
        raise ValueError("Census inventory must be nonempty")
    keys = ("scope", "name", "sha256", "role", "url")
    identities = []
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key] for key in keys):
            raise ValueError("Invalid census observation")
        if row["scope"] not in ("lock", "table") or row["role"] not in ("primary", "mirror", "historical"):
            raise ValueError("Unknown census owner or source role")
        if row.get("status") not in (ALIVE, DEAD, UNKNOWN):
            raise ValueError("Unknown census status")
        if not _valid_https_url(row["url"]):
            raise ValueError("Invalid census source URL")
        object_key(row["sha256"])
        identities.append(tuple(row[key] for key in keys))
    try:
        if any(not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key] for key in keys) for row in inventory):
            raise ValueError("Invalid census inventory")
        expected = [tuple(row[key] for key in keys) for row in inventory]
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid census inventory") from exc
    if len(set(identities)) != len(identities) or sorted(expected) != sorted(identities):
        raise ValueError("Census does not cover its exact inventory")
    if expected_inventory is not None:
        authoritative = [tuple(asdict(pin)[key] for key in keys) for pin in expected_inventory]
        if sorted(authoritative) != sorted(identities):
            raise ValueError("Census does not cover the checkout's pinned inputs")
    for status in (ALIVE, DEAD, UNKNOWN):
        if type(payload.get(status)) is not int or payload[status] != sum(row["status"] == status for row in rows):
            raise ValueError("Census counts disagree with observations")
    if type(payload.get("total")) is not int or payload["total"] != len(rows):
        raise ValueError("Census total disagrees with inventory")
    groups = pin_groups(rows)
    for group in groups:
        roles = [source["role"] for source in group["sources"]]
        if roles.count("primary") != 1 or roles.count("mirror") != 1 or len(set(roles)) != len(roles):
            raise ValueError("Incomplete pin ladder")
        primary = next(source for source in group["sources"] if source["role"] == "primary")
        from pm.artifact_mirror import historical_url
        historical = historical_url(primary["url"])
        observed_historical = next((source["url"] for source in group["sources"] if source["role"] == "historical"), None)
        if historical != observed_historical:
            raise ValueError("Incomplete historical source coverage")
    return groups


def render(census_result: Census) -> str:
    payload = report(census_result)
    groups = pin_groups(payload["rows"])
    lines = [f"pin liveness: {sum(pin['status'] == DEAD for pin in groups)} unrecoverable / "
             f"{sum(pin['status'] == UNKNOWN for pin in groups)} unknown of {len(groups)} pins; "
             f"{payload['dead']} retired source URLs"]
    for row in census_result.rows:
        if row.status != ALIVE:
            lines.append(f"  {row.status.upper():7} {row.pin.scope:5} {row.pin.role:10} {row.pin.name:32} {row.detail} {row.pin.url}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--target", default=None, help="restrict lock rows to one target (tables stay as scoped)")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    try:
        result = census(args.repo.resolve(), target=args.target)
        if not result.rows:
            raise ValueError("No pinned HTTP inputs found")
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    payload = report(result, complete=args.target is None)
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(render(result))
    return 1 if any(pin["status"] == DEAD for pin in pin_groups(payload["rows"])) else 0



if __name__ == "__main__":
    raise SystemExit(main())
