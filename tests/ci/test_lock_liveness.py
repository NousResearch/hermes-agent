"""The pin-liveness census and its tracker: pure logic, no network.

The census exists because pins are treated as immutable facts while sources
are not: a supplier pruning a tag, a rolling pool deleting an archive, and a
CDN rotating a build all retire the exact object a pin names, and nothing else
in CI notices until a user does. These tests pin the classifier, the source
inventory (lock rows + runtime-lib table + mirror objects), and the tracker's
open/update/close policy.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.ci import lock_liveness, pin_liveness_tracker
from pm.artifact_mirror import mirror_url


def write_pins(repo: Path, packages: dict, libs: dict | None = None) -> None:
    (repo / "pm").mkdir(parents=True, exist_ok=True)
    (repo / "pm/lock.json").write_text(
        json.dumps({"schema": 1, "packages": packages}), encoding="utf-8"
    )
    (repo / "pm/termux_runtime_libs.json").write_text(
        json.dumps(libs or {"libs": {}}), encoding="utf-8"
    )


def row(url: str, digest: str) -> dict:
    return {"url": url, "sha256": digest}


# ---------------------------------------------------------------------------
# Inventory: every pinned source on both authorities, plus the mirror object
# ---------------------------------------------------------------------------


def test_inventory_covers_lock_rows_table_rows_and_mirrors(tmp_path):
    digest = "a" * 64
    write_pins(
        tmp_path,
        {
            "portable": {"version": "1", "artifacts": {"any": row("https://upstream.test/portable.tgz", digest)}},
            "container": {"version": "sha256:x", "artifacts": {"linux-arm64-bionic": {"url": "docker://termux/termux-docker@sha256:" + "b" * 64}}},
        },
        {"libs": {"lib": row("https://packages.termux.dev/apt/termux-main/pool/main/l/lib/lib_1_aarch64.deb", "c" * 64)},
         "licenses": row("https://packages.termux.dev/apt/termux-main/pool/main/t/termux-licenses/termux-licenses_2.2_all.deb", "d" * 64)},
    )
    pins = lock_liveness.pinned_inputs(tmp_path)
    names = {(pin.scope, pin.name, pin.role) for pin in pins}
    # The docker row is an OCI digest, not an HTTP object: excluded.
    assert ("lock", "container@linux-arm64-bionic", "primary") not in names
    assert ("lock", "portable@any", "primary") in names
    assert ("table", "lib", "primary") in names
    assert ("table", "termux-licenses", "primary") in names
    # Every primary with a digest comes with its content-addressed mirror.
    assert ("lock", "portable@any", "mirror") in names
    assert ("table", "lib", "mirror") in names
    mirrors = {pin.url for pin in pins if pin.role == "mirror"}
    assert mirror_url("a" * 64) in mirrors
    assert mirror_url("c" * 64) in mirrors


def test_target_filter_scopes_lock_rows_but_not_tables(tmp_path):
    write_pins(
        tmp_path,
        {"tool": {"version": "1", "artifacts": {
            "win32-x64": row("https://upstream.test/win.zip", "a" * 64),
            "linux-arm64-bionic": row("https://packages.termux.dev/apt/termux-main/pool/main/t/tool/tool_1_aarch64.deb", "b" * 64),
        }}},
        {"libs": {"lib": row("https://packages.termux.dev/apt/termux-main/pool/main/l/lib/lib_1_aarch64.deb", "c" * 64)}},
    )
    win = lock_liveness.pinned_inputs(tmp_path, target="win32-x64")
    assert {pin.name for pin in win if pin.scope == "lock"} == {"tool@win32-x64"}
    # The runtime-lib table is bionic-only: excluded entirely on other targets.
    assert not [pin for pin in win if pin.scope == "table"]
    bionic = lock_liveness.pinned_inputs(tmp_path, target="linux-arm64-bionic")
    assert {pin.name for pin in bionic if pin.scope == "table"} == {"lib"}


# ---------------------------------------------------------------------------
# Classification: dead = the exact object is gone; unknown = vantage refused
# ---------------------------------------------------------------------------


def test_probe_classifies_dead_alive_and_unknown(monkeypatch):
    import urllib.error

    def raises(code):
        def _open(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, code, "boom", {}, None)

        return _open

    class Ok:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(lock_liveness.urllib.request, "urlopen", lambda request, timeout=None: Ok())
    assert lock_liveness.probe("https://x.test/ok")[0] == lock_liveness.ALIVE

    for code, expected in ((404, lock_liveness.DEAD), (410, lock_liveness.DEAD),
                           (403, lock_liveness.UNKNOWN), (429, lock_liveness.UNKNOWN), (500, lock_liveness.UNKNOWN)):
        monkeypatch.setattr(lock_liveness.urllib.request, "urlopen", raises(code))
        status, detail = lock_liveness.probe("https://x.test/y")
        assert status == expected, (code, status)
        assert str(code) in detail


def test_transport_failure_is_unknown_not_dead(monkeypatch):
    def refuses(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(lock_liveness.urllib.request, "urlopen", refuses)
    status, detail = lock_liveness.probe("https://x.test/any")
    assert status == lock_liveness.UNKNOWN
    assert "OSError" in detail


# ---------------------------------------------------------------------------
# Tracker policy: one issue in step, two repair classes kept apart
# ---------------------------------------------------------------------------

DEAD_PRIMARY = {"scope": "lock", "name": "uv@linux-arm64-bionic", "role": "primary",
                "url": "https://packages.termux.dev/apt/termux-main/pool/main/u/uv/uv_0.12.15_aarch64.deb",
                "status": "dead", "detail": "HTTP 404"}
DEAD_MIRROR = {"scope": "table", "name": "lib", "role": "mirror",
               "url": "https://hermes-assets.nousresearch.com/upstream/sha256/" + "c" * 64,
               "status": "dead", "detail": "HTTP 404"}
ALIVE = {"scope": "lock", "name": "portable@any", "role": "primary", "url": "https://u.test/a",
         "status": "alive", "detail": "HTTP 200"}
UNKNOWN = {"scope": "lock", "name": "x@any", "role": "primary", "url": "https://u.test/x",
           "status": "unknown", "detail": "HTTP 403"}


def report(rows):
    return {"alive": sum(1 for r in rows if r["status"] == "alive"),
            "dead": sum(1 for r in rows if r["status"] == "dead"),
            "unknown": sum(1 for r in rows if r["status"] == "unknown"),
            "total": len(rows), "rows": rows}


def test_all_alive_no_open_issue_is_none():
    assert pin_liveness_tracker.plan(report([ALIVE]), None)["action"] == "none"


def test_all_alive_with_open_issue_closes_it():
    change = pin_liveness_tracker.plan(report([ALIVE]), {"number": 7})
    assert change["action"] == "close"


def test_dead_primary_opens_and_names_the_repair_path():
    change = pin_liveness_tracker.plan(report([DEAD_PRIMARY, ALIVE]), None)
    assert change["action"] == "open"
    assert "uv@linux-arm64-bionic" in change["body"]
    assert "pm update --termux" in change["body"]
    assert "Retired pins" in change["body"]


def test_dead_mirror_opens_with_the_archive_path_not_the_termux_path():
    change = pin_liveness_tracker.plan(report([DEAD_MIRROR, ALIVE]), None)
    assert change["action"] == "open"
    assert "Unseeded mirrors" in change["body"]
    assert "archive-inputs.yml" in change["body"]


def test_unknown_rows_never_open_or_hold_the_issue():
    assert pin_liveness_tracker.plan(report([UNKNOWN, ALIVE]), None)["action"] == "none"
    assert pin_liveness_tracker.plan(report([UNKNOWN, ALIVE]), {"number": 3})["action"] == "close"


def test_open_issue_is_rewritten_in_place_not_duplicated():
    change = pin_liveness_tracker.plan(report([DEAD_PRIMARY]), {"number": 9})
    assert change["action"] == "update"
    assert change["title"].startswith("Pin liveness:")
    assert "DEAD" not in change["body"]  # tables carry the rows, not raw status dumps


def test_tracker_is_idempotent_for_identical_census():
    first = pin_liveness_tracker.plan(report([DEAD_PRIMARY, DEAD_MIRROR]), None)
    second = pin_liveness_tracker.plan(report([DEAD_PRIMARY, DEAD_MIRROR]), {"number": 1})
    assert first["title"] == second["title"]
    assert first["body"] == second["body"]
