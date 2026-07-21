from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import tarfile
from pathlib import Path

from jsonschema import Draft202012Validator


def _load_commands_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "plugins" / "truth-ledger" / "commands.py"
    spec = importlib.util.spec_from_file_location("truth_ledger_commands_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _append(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
        fh.write("\n")


def _seed_active_fact(root: Path, *, fact_id: str = "fact_1") -> None:
    _append(
        root / "ledger" / "2026-07.jsonl",
        {
            "schema_version": 1,
            "event_id": "evt_assert_1",
            "operation": "assert",
            "fact_id": fact_id,
            "scope": "user",
            "subject": "platform-user:cli:u1",
            "key": "profile.reply_style",
            "value": "contains-secret-token-12345",
            "fact": {
                "scope": "user",
                "kind": "preference",
                "subject": "platform-user:cli:u1",
                "key": "profile.reply_style",
                "value": "contains-secret-token-12345",
            },
            "occurred_at": "2026-07-17T20:00:00Z",
        },
    )


def test_status_report_is_redacted_and_disabled_when_missing(tmp_path):
    mod = _load_commands_module()

    missing = mod.status_report(tmp_path / "does-not-exist")
    assert missing["enabled"] is False

    root = tmp_path / "truth-ledger"
    _seed_active_fact(root)
    status = mod.status_report(root)

    assert status["enabled"] is True
    assert status["ledger_events"] == 1
    dumped = json.dumps(status)
    assert "contains-secret-token-12345" not in dumped


def test_retract_appends_event_and_never_rewrites_history(tmp_path):
    mod = _load_commands_module()
    root = tmp_path / "truth-ledger"
    _seed_active_fact(root, fact_id="fact_abc")

    dry = mod.retract_fact(root=root, fact_id="fact_abc", apply=False)
    assert dry["ok"] is True
    assert dry["dry_run"] is True

    before = (root / "ledger" / "2026-07.jsonl").read_text(encoding="utf-8")
    applied = mod.retract_fact(root=root, fact_id="fact_abc", apply=True)
    assert applied["ok"] is True
    assert applied["appended"] is True

    after = (root / "ledger" / "2026-07.jsonl").read_text(encoding="utf-8")
    assert before in after
    assert after.count("\n") == before.count("\n") + 1

    appended = json.loads(after.strip().splitlines()[-1])
    schema_path = Path(__file__).resolve().parents[3] / "plugins" / "truth-ledger" / "schemas" / "ledger-event-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(appended)) == []


def test_retract_rejects_bad_fact_ids(tmp_path):
    mod = _load_commands_module()
    root = tmp_path / "truth-ledger"
    _seed_active_fact(root)

    out = mod.retract_fact(root=root, fact_id="bad id!!", apply=True)
    assert out["ok"] is False
    assert out["reason"] == "invalid_fact_id"


def test_rebuild_reports_first_then_backs_up_and_replaces_current(tmp_path):
    mod = _load_commands_module()
    root = tmp_path / "truth-ledger"
    _seed_active_fact(root)
    views = root / "views"
    views.mkdir(parents=True, exist_ok=True)
    current = views / "current.jsonl"
    current.write_text('{"old":true}\n', encoding="utf-8")

    report = mod.rebuild_views(root=root, apply=False)
    assert report["ok"] is True
    assert report["dry_run"] is True

    out = mod.rebuild_views(root=root, apply=True)
    assert out["ok"] is True
    assert out["backup_path"]
    assert Path(out["backup_path"]).exists()
    assert out["before_sha256"] != out["after_sha256"]

    new_current = current.read_text(encoding="utf-8")
    assert '"old":true' not in new_current


def test_export_defaults_to_protected_local_tarball(tmp_path):
    mod = _load_commands_module()
    root = tmp_path / "truth-ledger"
    _seed_active_fact(root)

    report = mod.export_snapshot(root=root, apply=False)
    assert report["ok"] is True
    assert report["dry_run"] is True

    out = mod.export_snapshot(root=root, apply=True)
    export_path = Path(out["path"])
    assert out["ok"] is True
    assert export_path.exists()
    assert tarfile.is_tarfile(export_path)
    assert (export_path.stat().st_mode & 0o777) == 0o600

    digest = hashlib.sha256(export_path.read_bytes()).hexdigest()
    assert digest == out["sha256"]


def test_headless_dispatch_supports_json_output(tmp_path):
    mod = _load_commands_module()
    root = tmp_path / "truth-ledger"
    _seed_active_fact(root)

    payload = mod.dispatch_headless(action="status", root=root)
    assert payload["ok"] is True

    rendered = mod.handle_truth_ledger_command("status --json", root=root)
    loaded = json.loads(rendered)
    assert loaded["ok"] is True
    assert loaded["action"] == "status"


def test_review_report_reads_nested_dead_letter_reason_from_flow(tmp_path):
    mod = _load_commands_module()
    root = tmp_path / "truth-ledger"
    dead_dir = root / "spool" / "dead-letter"
    dead_dir.mkdir(parents=True, exist_ok=True)
    (dead_dir / "dead-1.json").write_text(
        json.dumps(
            {
                "schema_name": "truth-ledger.spool-record.v1",
                "schema_version": 1,
                "state": "dead_lettered",
                "flow": {"dead_letter_reason": "schema_mismatch"},
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    out = mod.review_report(root=root, limit=5)
    assert out["ok"] is True
    assert out["dead_letter_entries"] == 1
    assert out["dead_letter_preview"][0]["reason"] == "schema_mismatch"


def test_process_command_is_bounded_dry_run_by_default_and_receives_runtime_context(tmp_path, monkeypatch):
    mod = _load_commands_module()
    runtime_ctx = object()
    observed: dict = {}

    async def fake_process_pending(*, root, ctx, limit, apply):
        observed.update({"root": root, "ctx": ctx, "limit": limit, "apply": apply})
        return {
            "ok": True,
            "action": "process",
            "dry_run": not apply,
            "limit": limit,
            "would_process": 0,
        }

    monkeypatch.setattr(mod, "_process_pending", fake_process_pending)
    rendered = asyncio.run(
        mod.handle_truth_ledger_command(
            "process --json",
            root=tmp_path / "truth-ledger",
            runtime_ctx=runtime_ctx,
        )
    )

    payload = json.loads(rendered)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert observed["ctx"] is runtime_ctx
    assert observed["limit"] == 1
    assert observed["apply"] is False


def test_process_command_requires_explicit_apply_and_rejects_limit_above_hard_cap(tmp_path, monkeypatch):
    mod = _load_commands_module()
    calls: list[tuple[int, bool]] = []

    async def fake_process_pending(*, root, ctx, limit, apply):
        calls.append((limit, apply))
        return {"ok": True, "action": "process", "dry_run": not apply, "limit": limit}

    monkeypatch.setattr(mod, "_process_pending", fake_process_pending)
    rendered = asyncio.run(
        mod.handle_truth_ledger_command(
            "process --limit 3 --apply --json",
            root=tmp_path / "truth-ledger",
            runtime_ctx=object(),
        )
    )
    assert json.loads(rendered)["dry_run"] is False
    assert calls == [(3, True)]

    rejected = mod.handle_truth_ledger_command(
        "process --limit 4 --json",
        root=tmp_path / "truth-ledger",
        runtime_ctx=object(),
    )
    assert json.loads(rejected)["ok"] is False
    assert calls == [(3, True)]
