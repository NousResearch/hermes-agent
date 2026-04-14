#!/usr/bin/env python3
"""Focused failure-simulation harness for GTC publish transaction semantics.

Coverage target:
- Crash window before publish-promotion should not mutate live latest artifacts.
- Publish-lock contention should fail-close without live latest mutation.
- Recovery run after a crash-window interruption should still promote successfully.
- Mid-promotion crash windows should be journaled and deterministically recovered to terminal semantics.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from strict_required_check_contracts import (
    required_check_provenance,
    strict_required_check_contract,
)

ROOT = Path(__file__).resolve().parents[3]
GTC_SYNC = ROOT / "ops" / "openclaw" / "continuity" / "gtc_v2_sync.sh"
INIT_DB = ROOT / "ops" / "openclaw" / "continuity" / "init_db.sh"
INGRESS_GUARD = ROOT / "ops" / "openclaw" / "continuity" / "mutator_ingress_guard.sh"

CONTRACT = strict_required_check_contract("gtc_publish_transaction_regressions")
CHECK_ID = CONTRACT.check_id
HARNESS_ID = CONTRACT.harness
SUMMARY_SOURCE = CONTRACT.summary_source
SUMMARY_SCHEMA_VERSION = CONTRACT.summary_schema_version
LEGACY_SCHEMA_VERSION = "gtc.publish.transaction.regressions.v1"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def copy_exec(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR)


def make_root(prefix: str) -> tuple[Path, Path]:
    td = Path(tempfile.mkdtemp(prefix=prefix))
    root = td / "root"
    copy_exec(INIT_DB, root / "ops" / "openclaw" / "continuity" / "init_db.sh")
    copy_exec(INGRESS_GUARD, root / "ops" / "openclaw" / "continuity" / "mutator_ingress_guard.sh")
    return td, root


def env_for(root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OPENCLAW_ROOT": str(root),
            "OPENCLAW_CONTINUITY_DB_PATH": str(root / "state" / "continuity" / "continuity_os.sqlite"),
            "OPENCLAW_GTC_ROOT": str(root / "state" / "gtc-v2"),
            "OPENCLAW_INTERNAL_MUTATION": "1",
            "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "harness:check_gtc_publish_transaction_regressions",
            "OPENCLAW_GTC_SYNC_MAX_ROWS": "100",
        }
    )
    if extra:
        env.update(extra)
    return env


def run_sync(root: Path, *args: str, extra_env: dict[str, str] | None = None, timeout: int = 60) -> tuple[int, dict[str, Any], str]:
    cmd = ["bash", str(GTC_SYNC), *args, "--json"]
    cp = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        env=env_for(root, extra_env),
        timeout=timeout,
    )
    stdout = (cp.stdout or "").strip()
    stderr = (cp.stderr or "").strip()
    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        payload = {"_parse_error": "stdout_not_json", "stdout": stdout[:1200], "stderr": stderr[:600]}
    if not isinstance(payload, dict):
        payload = {"_parse_error": "payload_not_object", "payload_type": type(payload).__name__}
    return cp.returncode, payload, stderr


def seed_live_latest_sentinels(root: Path, tag: str) -> dict[Path, str]:
    latest = root / "state" / "gtc-v2" / "latest"
    connectors = latest / "connectors"
    connectors.mkdir(parents=True, exist_ok=True)

    sentinels = {
        latest / "publish_manifest.json": json.dumps({"sentinel": f"manifest_{tag}"}, sort_keys=True) + "\n",
        latest / "continuity_current.json": json.dumps({"sentinel": f"continuity_{tag}"}, sort_keys=True) + "\n",
        connectors / "runtime.gateway__gateway-main.json": json.dumps({"sentinel": f"connector_{tag}"}, sort_keys=True) + "\n",
    }
    for path, content in sentinels.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return sentinels


def seed_existing_live_release(root: Path, tag: str) -> dict[str, Any]:
    sentinels = seed_live_latest_sentinels(root, tag)
    latest = root / "state" / "gtc-v2" / "latest"
    surfaces = root / "state" / "gtc-v2" / "surfaces"
    old_generation_id = f"gen_old_{tag}"

    (latest / "publish_anchor.json").write_text(
        json.dumps(
            {
                "schema_version": "gtc.publish_anchor.v1",
                "generated_at": "2026-01-01T00:00:00Z",
                "build_generation_id": old_generation_id,
                "valid_until": "2026-01-01T00:05:00Z",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (latest / "publish_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "gtc.publish_manifest.v1",
                "generated_at": "2026-01-01T00:00:00Z",
                "build_generation_id": old_generation_id,
                "base_generation_id": None,
                "base_coherence_guard": {},
                "valid_until": "2026-01-01T00:05:00Z",
                "latest_paths": {},
                "latest_sha256": {},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    surfaces.mkdir(parents=True, exist_ok=True)
    sentinels[surfaces / "gtc_status.md"] = "old surface\n"
    (surfaces / "gtc_status.md").write_text("old surface\n", encoding="utf-8")

    return {
        "old_generation_id": old_generation_id,
        "sentinels": sentinels,
    }


def assert_sentinels_unchanged(sentinels: dict[Path, str], *, case: str) -> None:
    for path, expected in sentinels.items():
        got = path.read_text(encoding="utf-8")
        assert_true(got == expected, f"{case}: live latest mutated unexpectedly at {path}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _required_check_provenance() -> dict[str, object]:
    return required_check_provenance(CHECK_ID)


def _summary_payload(*, results: list[dict[str, Any]], failed: int, ok: bool | None = None, error: str | None = None) -> dict[str, Any]:
    summary_ok = failed == 0 if ok is None else bool(ok)
    payload = {
        "ok": summary_ok,
        "check_id": CHECK_ID,
        "harness": HARNESS_ID,
        "source": SUMMARY_SOURCE,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "schema_version": LEGACY_SCHEMA_VERSION,
        "required_check_provenance": _required_check_provenance(),
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "scenario_count": len(results),
        "failed_count": failed,
        "results": results,
    }
    if error:
        payload["error"] = error
    return payload


def scenario_lock_busy_failclose_preserves_live_latest() -> dict[str, Any]:
    td, root = make_root("gtc_publish_lock_busy_")
    try:
        sentinels = seed_live_latest_sentinels(root, "before_lock_busy")
        lock_path = root / "state" / "gtc-v2" / "locks" / "gtc_latest_publish.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            rc, payload, stderr = run_sync(
                root,
                "--skip-schema-gate",
                extra_env={"OPENCLAW_GTC_PUBLISH_LOCK_WAIT_SEC": "0.2"},
            )
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        assert_true(rc == 5, f"lock_busy: expected rc=5, got {rc}; stderr={stderr}")
        assert_true(payload.get("ok") is False, f"lock_busy: expected ok=false, got {payload.get('ok')}")
        promo = payload.get("publish_promotion") or {}
        assert_true(promo.get("error_class") == "lock_busy", f"lock_busy: expected error_class=lock_busy, got {promo.get('error_class')}")
        assert_true(promo.get("promoted") is False, "lock_busy: expected promoted=false")
        assert_sentinels_unchanged(sentinels, case="lock_busy")

        return {
            "name": "lock_busy_failclose_preserves_live_latest",
            "ok": True,
            "returncode": rc,
            "error_class": promo.get("error_class"),
            "promoted": promo.get("promoted"),
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _wait_for_staging_manifest(root: Path, timeout_sec: float = 8.0) -> Path | None:
    staging_root = root / "state" / "gtc-v2" / ".staging"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        matches = sorted(staging_root.glob("gtc_sync_*/latest/publish_manifest.json"))
        if matches:
            return matches[0]
        time.sleep(0.05)
    return None


def scenario_crash_window_then_recovery() -> dict[str, Any]:
    td, root = make_root("gtc_publish_crash_window_")
    try:
        sentinels = seed_live_latest_sentinels(root, "before_crash")
        lock_path = root / "state" / "gtc-v2" / "locks" / "gtc_latest_publish.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        proc: subprocess.Popen[str] | None = None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            proc = subprocess.Popen(
                ["bash", str(GTC_SYNC), "--skip-schema-gate", "--json"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_for(root, {"OPENCLAW_GTC_PUBLISH_LOCK_WAIT_SEC": "120"}),
                start_new_session=True,
            )

            manifest = _wait_for_staging_manifest(root)
            assert_true(manifest is not None, "crash_window: timed out waiting for staged publish_manifest")

            os.killpg(proc.pid, signal.SIGKILL)
            _stdout, stderr = proc.communicate(timeout=10)
            assert_true(proc.returncode is not None and proc.returncode < 0, f"crash_window: expected signal-terminated process, got rc={proc.returncode}")
            assert_true("Traceback" not in (stderr or ""), "crash_window: unexpected traceback in stderr")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        assert_sentinels_unchanged(sentinels, case="crash_window")

        orphan_dirs = sorted((root / "state" / "gtc-v2" / ".staging").glob("gtc_sync_*"))
        assert_true(bool(orphan_dirs), "crash_window: expected orphan staging directory after forced kill")

        rc_recover, payload_recover, stderr_recover = run_sync(root, "--skip-schema-gate")
        assert_true(rc_recover == 0, f"recovery: expected rc=0, got {rc_recover}; stderr={stderr_recover}")
        assert_true(payload_recover.get("ok") is True, f"recovery: expected ok=true, got {payload_recover.get('ok')}")
        promo = payload_recover.get("publish_promotion") or {}
        assert_true(promo.get("promoted") is True, f"recovery: expected promoted=true, got {promo.get('promoted')}")

        live_manifest_path = root / "state" / "gtc-v2" / "latest" / "publish_manifest.json"
        live_manifest = json.loads(live_manifest_path.read_text(encoding="utf-8"))
        assert_true(live_manifest.get("schema_version") == "gtc.publish_manifest.v1", "recovery: live publish_manifest not promoted")
        assert_true(
            str(live_manifest.get("build_generation_id") or "").startswith("gtcgen_"),
            "recovery: expected generated build_generation_id",
        )

        return {
            "name": "crash_window_then_recovery",
            "ok": True,
            "killed_returncode": proc.returncode if proc is not None else None,
            "orphan_staging_dirs": len(orphan_dirs),
            "recovery_returncode": rc_recover,
            "recovery_promoted": promo.get("promoted"),
            "recovery_generation_id": live_manifest.get("build_generation_id"),
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


def scenario_mid_promotion_crash_then_recovery_semantics() -> dict[str, Any]:
    td, root = make_root("gtc_publish_mid_promotion_crash_")
    try:
        rc_crash, payload_crash, stderr_crash = run_sync(
            root,
            "--skip-schema-gate",
            extra_env={"OPENCLAW_GTC_TEST_CRASH_STEP": "latest_promoted"},
        )
        assert_true(rc_crash == 91, f"mid_promotion_crash: expected rc=91, got {rc_crash}; stderr={stderr_crash}")

        journal_latest_path = root / "state" / "gtc-v2" / "publish_journal" / "latest_transaction.json"
        journal_events_path = root / "state" / "gtc-v2" / "publish_journal" / "transactions.jsonl"
        latest_crash = read_json(journal_latest_path)
        crash_tx_id = str(latest_crash.get("tx_id") or "")
        assert_true(bool(crash_tx_id), "mid_promotion_crash: missing tx_id in publish journal")
        assert_true(
            str(latest_crash.get("state") or "") == "promoting",
            f"mid_promotion_crash: expected non-terminal promoting state, got {latest_crash.get('state')}",
        )

        rc_recover, payload_recover, stderr_recover = run_sync(root, "--skip-schema-gate")
        assert_true(rc_recover == 0, f"mid_promotion_recover: expected rc=0, got {rc_recover}; stderr={stderr_recover}")
        assert_true(payload_recover.get("ok") is True, f"mid_promotion_recover: expected ok=true, got {payload_recover.get('ok')}")

        recovery = ((payload_recover.get("publish_promotion") or {}).get("recovery") or {})
        rec_status = str(recovery.get("status") or "")
        assert_true(
            rec_status in {"recovered_rolled_back", "recovered_committed"},
            f"mid_promotion_recover: unexpected recovery status={rec_status}",
        )

        latest_final = read_json(journal_latest_path)
        assert_true(str(latest_final.get("state") or "") == "committed", "mid_promotion_recover: latest tx not committed")
        assert_true(str(latest_final.get("step") or "") == "verified", "mid_promotion_recover: latest tx not verified")

        events = read_jsonl(journal_events_path)
        crash_events = [row for row in events if str(row.get("tx_id") or "") == crash_tx_id]
        crash_event_names = {str(row.get("event") or "") for row in crash_events}
        assert_true(
            bool({"recovered_rolled_back", "recovered_committed"} & crash_event_names),
            f"mid_promotion_recover: crash tx missing recovery terminal event; events={sorted(crash_event_names)}",
        )

        orphan_dirs = sorted((root / "state" / "gtc-v2" / ".staging").glob("gtc_sync_*"))
        assert_true(not orphan_dirs, f"mid_promotion_recover: orphan staging dirs remain: {orphan_dirs}")

        return {
            "name": "mid_promotion_crash_then_recovery_semantics",
            "ok": True,
            "crash_returncode": rc_crash,
            "recovery_returncode": rc_recover,
            "recovery_status": rec_status,
            "final_state": latest_final.get("state"),
            "final_step": latest_final.get("step"),
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


def scenario_fully_promoted_crash_recovery_discards_backups() -> dict[str, Any]:
    td, root = make_root("gtc_publish_fully_promoted_crash_")
    try:
        seed_existing_live_release(root, "before_surfaces_crash")

        rc_crash, payload_crash, stderr_crash = run_sync(
            root,
            "--skip-schema-gate",
            extra_env={"OPENCLAW_GTC_TEST_CRASH_STEP": "surfaces_promoted"},
        )
        assert_true(
            rc_crash == 91,
            f"fully_promoted_crash: expected rc=91, got {rc_crash}; stderr={stderr_crash}",
        )

        journal_latest_path = root / "state" / "gtc-v2" / "publish_journal" / "latest_transaction.json"
        journal_events_path = root / "state" / "gtc-v2" / "publish_journal" / "transactions.jsonl"

        latest_crash = read_json(journal_latest_path)
        crash_tx_id = str(latest_crash.get("tx_id") or "")
        crash_generation_id = str(latest_crash.get("build_generation_id") or "")
        assert_true(bool(crash_tx_id), "fully_promoted_crash: missing tx_id in publish journal")
        assert_true(bool(crash_generation_id), "fully_promoted_crash: missing build_generation_id in publish journal")
        assert_true(
            str(latest_crash.get("state") or "") == "promoting",
            f"fully_promoted_crash: expected promoting state, got {latest_crash.get('state')}",
        )
        assert_true(
            str(latest_crash.get("step") or "") == "surfaces_promoted",
            f"fully_promoted_crash: expected step=surfaces_promoted, got {latest_crash.get('step')}",
        )

        rc_recover, payload_recover, stderr_recover = run_sync(root, "--skip-schema-gate")
        assert_true(
            rc_recover == 0,
            f"fully_promoted_recover: expected rc=0, got {rc_recover}; stderr={stderr_recover}",
        )
        assert_true(
            payload_recover.get("ok") is True,
            f"fully_promoted_recover: expected ok=true, got {payload_recover.get('ok')}",
        )

        recovery = ((payload_recover.get("publish_promotion") or {}).get("recovery") or {})
        assert_true(
            str(recovery.get("status") or "") == "recovered_committed",
            f"fully_promoted_recover: expected recovered_committed, got {recovery.get('status')}",
        )
        assert_true(
            str(recovery.get("tx_id") or "") == crash_tx_id,
            "fully_promoted_recover: recovery tx_id does not match crash tx_id",
        )

        events = read_jsonl(journal_events_path)
        crash_events = [row for row in events if str(row.get("tx_id") or "") == crash_tx_id]
        crash_event_names = {str(row.get("event") or "") for row in crash_events}
        assert_true(
            "recovered_rolled_back" not in crash_event_names,
            f"fully_promoted_recover: unexpected recovered_rolled_back event; events={sorted(crash_event_names)}",
        )

        recovered_committed = None
        for row in crash_events:
            if str(row.get("event") or "") == "recovered_committed":
                recovered_committed = row
                break
        assert_true(recovered_committed is not None, "fully_promoted_recover: missing recovered_committed event")
        details = (recovered_committed or {}).get("details") or {}
        assert_true(
            str(details.get("live_generation_id") or "") == crash_generation_id,
            "fully_promoted_recover: recovered_committed details live_generation_id mismatch",
        )
        assert_true(details.get("backup_latest_discarded") is True, "fully_promoted_recover: backup_latest_discarded should be true")
        assert_true(details.get("backup_surfaces_discarded") is True, "fully_promoted_recover: backup_surfaces_discarded should be true")

        orphan_dirs = sorted((root / "state" / "gtc-v2" / ".staging").glob("gtc_sync_*"))
        assert_true(not orphan_dirs, f"fully_promoted_recover: orphan staging dirs remain: {orphan_dirs}")

        return {
            "name": "fully_promoted_crash_recovery_discards_backups",
            "ok": True,
            "crash_returncode": rc_crash,
            "recovery_returncode": rc_recover,
            "recovery_status": recovery.get("status"),
            "backup_latest_discarded": details.get("backup_latest_discarded"),
            "backup_surfaces_discarded": details.get("backup_surfaces_discarded"),
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


SCENARIOS: list[tuple[str, Any]] = [
    ("lock_busy_failclose_preserves_live_latest", scenario_lock_busy_failclose_preserves_live_latest),
    ("crash_window_then_recovery", scenario_crash_window_then_recovery),
    ("mid_promotion_crash_then_recovery_semantics", scenario_mid_promotion_crash_then_recovery_semantics),
    ("fully_promoted_crash_recovery_discards_backups", scenario_fully_promoted_crash_recovery_discards_backups),
]

_IMPLEMENTED_SCENARIO_NAMES = [name for name, _ in SCENARIOS]
if _IMPLEMENTED_SCENARIO_NAMES != list(CONTRACT.scenario_names):
    raise RuntimeError(
        "required-check scenario contract mismatch for "
        f"{CHECK_ID}: implemented={_IMPLEMENTED_SCENARIO_NAMES} expected={list(CONTRACT.scenario_names)}"
    )


def main() -> int:
    if not GTC_SYNC.exists():
        summary = _summary_payload(
            results=[],
            failed=1,
            ok=False,
            error="gtc_sync_missing",
        )
        summary["path"] = str(GTC_SYNC)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    results: list[dict[str, Any]] = []
    failed = 0
    for name, fn in SCENARIOS:
        try:
            row = fn()
            results.append(row)
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            results.append({"name": name, "ok": False, "error": str(exc)})
            print(f"FAIL {name}: {exc}")

    summary = _summary_payload(results=results, failed=failed)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
