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
import pytest
from pathlib import Path

from scripts.ci import lock_liveness, pin_liveness_tracker
from pm.artifact_mirror import mirror_url


def write_pins(repo: Path, packages: dict, libs: dict | None = None) -> None:
    (repo / "pm").mkdir(parents=True, exist_ok=True)
    (repo / "pm/artifact-mirror.json").write_text(json.dumps({"origin": "https://hermes-assets.nousresearch.com", "prefix": "upstream/sha256/"}), encoding="utf-8")
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
    # Generate the exact production receipt rather than hand-writing summary counts.
    observations = []
    for index, r in enumerate(rows):
        digest = r.get("sha256", str(index % 10) * 64)
        primary_url = r["url"] if r["role"] == "primary" else f"https://upstream.test/{index}"
        for role, url in (("primary", primary_url), ("mirror", mirror_url(digest))):
            data = {**r, "role": role, "url": url, "sha256": digest}
            observations.append(lock_liveness.Row(lock_liveness.Pin(r["scope"], r["name"], url, digest, role), r["status"], r["detail"]))
        from pm.artifact_mirror import historical_url
        historical = historical_url(primary_url)
        if historical:
            observations.append(lock_liveness.Row(lock_liveness.Pin(r["scope"], r["name"], historical, digest, "historical"), r["status"], r["detail"]))
    return lock_liveness.report(lock_liveness.Census(observations))


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
    assert "Unrecoverable pins" in change["body"]


def test_dead_mirror_opens_with_the_archive_path_not_the_termux_path():
    change = pin_liveness_tracker.plan(report([DEAD_MIRROR, ALIVE]), None)
    assert change["action"] == "open"
    assert "Unrecoverable pins" in change["body"]
    assert "archive-inputs.yml" in change["body"]


def test_unknown_rows_never_open_but_hold_the_issue():
    assert pin_liveness_tracker.plan(report([UNKNOWN, ALIVE]), None)["action"] == "none"
    assert pin_liveness_tracker.plan(report([UNKNOWN, ALIVE]), {"number": 3})["action"] == "none"


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


def test_target_missing_does_not_probe_unrelated_architecture(tmp_path):
    write_pins(tmp_path, {"tool": {"artifacts": {"linux-x64": row("https://upstream.test/linux", "a" * 64)}}})
    assert lock_liveness.lock_pins(tmp_path, target="win32-x64") == []


def test_dead_primary_with_live_mirror_does_not_fail_cli(tmp_path, monkeypatch):
    write_pins(tmp_path, {"tool": {"artifacts": {"any": row("https://upstream.test/gone", "a" * 64)}}})
    monkeypatch.setattr(lock_liveness, "probe", lambda url: ("alive", "HTTP 200") if "sha256" in url else ("dead", "HTTP 404"))
    assert lock_liveness.main(["--repo", str(tmp_path), "--format", "json"]) == 0


def test_inconclusive_census_keeps_existing_incident_open():
    assert pin_liveness_tracker.plan(report([UNKNOWN, ALIVE]), {"number": 3})["action"] == "none"


def test_empty_census_cannot_close_existing_incident():
    with pytest.raises(ValueError):
        pin_liveness_tracker.plan(report([]), {"number": 3})


@pytest.mark.parametrize("statuses,expected", [
    (("dead", "alive", "dead"), "alive"),
    (("dead", "dead", "alive"), "alive"),
    (("dead", "dead", "unknown"), "unknown"),
    (("dead", "dead", "dead"), "dead"),
    (("unknown", "alive", "unknown"), "alive"),
])
def test_complete_ladder_recoverability(statuses, expected):
    payload = report([DEAD_PRIMARY])
    for observation, status in zip(payload["rows"], statuses):
        observation["status"] = status
    for status in ("alive", "dead", "unknown"):
        payload[status] = sum(r["status"] == status for r in payload["rows"])
    assert lock_liveness.validate_report(payload)[0]["status"] == expected
    action = pin_liveness_tracker.plan(payload, {"number": 1})["action"]
    assert action == {"alive": "close", "unknown": "none", "dead": "update"}[expected]


def test_distinct_artifacts_share_owner_but_not_recovery(tmp_path, monkeypatch):
    write_pins(tmp_path, {"tool": {"artifacts": {"any": [
        row("https://upstream.test/a", "a" * 64), row("https://upstream.test/b", "b" * 64)]}}})
    monkeypatch.setattr(lock_liveness, "probe", lambda url: ("alive", "HTTP 200") if url.endswith("a") else ("dead", "HTTP 404"))
    groups = lock_liveness.validate_report(lock_liveness.report(lock_liveness.census(tmp_path)))
    assert len(groups) == 2
    assert {pin["status"] for pin in groups} == {"alive", "dead"}


@pytest.mark.parametrize("damage", ["missing-row", "duplicate-row", "count", "status", "partial", "missing-mirror", "missing-history"])
def test_invalid_or_partial_receipt_cannot_close(damage):
    payload = report([DEAD_PRIMARY])
    for r in payload["rows"]:
        r["status"] = "alive"
    payload.update(alive=3, dead=0, unknown=0)
    if damage == "missing-row": payload["rows"].pop()
    elif damage == "duplicate-row": payload["rows"].append(payload["rows"][0])
    elif damage == "count": payload["alive"] = 4
    elif damage == "status": payload["rows"][0]["status"] = "ok"
    elif damage == "partial": payload["complete"] = False
    else:
        role = "mirror" if damage == "missing-mirror" else "historical"
        payload["rows"] = [r for r in payload["rows"] if r["role"] != role]
        payload["inventory"] = [r for r in payload["inventory"] if r["role"] != role]
        payload.update(total=2, alive=2)
    with pytest.raises(ValueError):
        pin_liveness_tracker.plan(payload, {"number": 1})


def test_census_deduplicates_network_requests_without_losing_owners(tmp_path, monkeypatch):
    write_pins(tmp_path, {name: {"artifacts": {"any": row("https://upstream.test/shared", "a" * 64)}} for name in ("a", "b")})
    seen = []
    def probe(url):
        seen.append(url)
        return "alive", "HTTP 200"
    monkeypatch.setattr(lock_liveness, "probe", probe)
    result = lock_liveness.census(tmp_path)
    assert len(result.rows) == 4 and len(seen) == 2
    assert len(lock_liveness.validate_report(lock_liveness.report(result))) == 2


def test_mirror_layout_belongs_to_selected_repository(tmp_path):
    write_pins(tmp_path, {"tool": {"artifacts": {"any": row("https://upstream.test/tool", "a" * 64)}}})
    (tmp_path / "pm/artifact-mirror.json").write_text(json.dumps({"origin": "https://archive.test", "prefix": "inputs/"}), encoding="utf-8")
    assert [p.url for p in lock_liveness.pinned_inputs(tmp_path) if p.role == "mirror"] == ["https://archive.test/inputs/" + "a" * 64]


@pytest.mark.parametrize("layout", [
    {"origin": "https:///no-host", "prefix": "inputs/"},
    {"origin": "https://archive.test?wrong", "prefix": "inputs/"},
    {"origin": "https://user@archive.test", "prefix": "inputs/"},
    {"origin": "https://archive.test/path", "prefix": "inputs/"},
    {"origin": "https://archive.test", "prefix": "inputs"},
    {"origin": "https://archive.test", "prefix": "../inputs/"},
])
def test_malformed_mirror_layout_cannot_produce_a_complete_census(tmp_path, layout):
    write_pins(tmp_path, {"tool": {"artifacts": {"any": row("https://upstream.test/tool", "a" * 64)}}})
    (tmp_path / "pm/artifact-mirror.json").write_text(json.dumps(layout), encoding="utf-8")
    with pytest.raises(ValueError):
        lock_liveness.pinned_inputs(tmp_path)


def test_mirror_layout_normalization_matches_the_consumer(tmp_path):
    from pm.artifact_mirror import public_prefix

    layout = {"origin": "https://archive.test/", "prefix": "inputs/"}
    write_pins(tmp_path, {"tool": {"artifacts": {"any": row("https://upstream.test/tool", "a" * 64)}}})
    (tmp_path / "pm/artifact-mirror.json").write_text(json.dumps(layout), encoding="utf-8")
    mirror = next(pin for pin in lock_liveness.pinned_inputs(tmp_path) if pin.role == "mirror")
    assert mirror.url == public_prefix(layout) + mirror.sha256 == "https://archive.test/inputs/" + "a" * 64


def test_tracker_dry_run_is_offline(tmp_path, monkeypatch, capsys):
    write_pins(tmp_path, {"tool": {"artifacts": {"any": row("https://upstream.test/tool", "a" * 64)}}})
    monkeypatch.setattr(lock_liveness, "probe", lambda url: ("dead", "HTTP 404"))
    path = tmp_path / "report.json"
    path.write_text(json.dumps(lock_liveness.report(lock_liveness.census(tmp_path))), encoding="utf-8")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    def forbidden(*args):
        raise AssertionError("dry-run invoked GitHub")
    monkeypatch.setattr(pin_liveness_tracker, "_gh", forbidden)
    assert pin_liveness_tracker.main(["--report", str(path), "--repo", str(tmp_path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "open"


def test_invalid_table_or_digest_cannot_omit_inventory(tmp_path):
    write_pins(tmp_path, {"tool": {"artifacts": {"any": row("https://upstream.test/tool", "bad")}}})
    with pytest.raises(ValueError): lock_liveness.pinned_inputs(tmp_path)
    write_pins(tmp_path, {}, {"libs": {"bad": {"sha256": "a" * 64}}})
    with pytest.raises(ValueError): lock_liveness.pinned_inputs(tmp_path)


@pytest.mark.parametrize("artifacts", [{}, {"any": None}, {"any": []}])
def test_empty_package_inventory_cannot_produce_recovery(tmp_path, artifacts):
    write_pins(tmp_path, {
        "good": {"artifacts": {"any": row("https://upstream.test/good", "a" * 64)}},
        "broken": {"artifacts": artifacts},
    })
    with pytest.raises(ValueError):
        lock_liveness.pinned_inputs(tmp_path)


@pytest.mark.parametrize("scope", ["lock", "table"])
def test_inventory_digest_must_be_a_string(tmp_path, scope):
    bad = row("https://upstream.test/bad", int("1" * 64))
    write_pins(tmp_path, {"tool": {"artifacts": {"any": bad}}} if scope == "lock" else {},
               {"libs": {"bad": bad}} if scope == "table" else None)
    with pytest.raises(ValueError):
        lock_liveness.pinned_inputs(tmp_path)


@pytest.mark.parametrize("damage", ["omit-owner", "retarget-mirror"])
def test_tracker_checks_the_checkout_before_any_github_call(tmp_path, monkeypatch, damage):
    write_pins(tmp_path, {name: {"artifacts": {"any": row(f"https://upstream.test/{name}", digest * 64)}}
                          for name, digest in (("first", "a"), ("second", "b"))})
    monkeypatch.setattr(lock_liveness, "probe", lambda url: ("alive", "HTTP 200"))
    payload = lock_liveness.report(lock_liveness.census(tmp_path))
    if damage == "omit-owner":
        for key in ("rows", "inventory"):
            payload[key] = [r for r in payload[key] if r["name"] == "first@any"]
        payload.update(alive=2, total=2)
    else:
        for key in ("rows", "inventory"):
            for r in payload[key]:
                if r["role"] == "mirror":
                    r["url"] = "https://unrelated.test/always-200"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GITHUB_REPOSITORY", "NousResearch/hermes-agent")
    def forbidden(*args):
        raise AssertionError("invalid report reached GitHub")
    monkeypatch.setattr(pin_liveness_tracker, "_gh", forbidden)
    with pytest.raises(ValueError):
        pin_liveness_tracker.main(["--report", str(path), "--repo", str(tmp_path)])


def test_labeled_pull_requests_cannot_hide_an_older_incident(monkeypatch):
    calls = []
    def github(args):
        calls.append(args)
        if args[-1].endswith("&page=1"):
            return json.dumps([{"number": i, "pull_request": {}} for i in range(100)])
        return json.dumps([{"number": 42}])
    monkeypatch.setattr(pin_liveness_tracker, "_gh", github)
    assert pin_liveness_tracker._open_issue("NousResearch/hermes-agent") == {"number": 42}
    assert len(calls) == 2 and "page=2" in calls[-1][-1]
    # Unknown data still holds that discovered incident rather than opening another.
    assert pin_liveness_tracker.plan(report([UNKNOWN]), {"number": 42})["action"] == "none"
