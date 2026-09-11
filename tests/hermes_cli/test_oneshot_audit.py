from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _write_audit(home: str, prompt: str, index: int) -> str:
    import os

    os.environ["HERMES_HOME"] = home
    from hermes_cli.oneshot_audit import OneShotAudit

    audit = OneShotAudit(prompt, "query")
    audit.bind_session(f"session-{index}")
    audit.finish(0)
    return audit.audit_id


def _records(home: Path) -> list[dict]:
    path = home / "logs" / "oneshot-audit.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_audit_correlates_lifecycle_without_copying_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.oneshot_audit import OneShotAudit

    prompt = "private prompt token: swordfish"
    audit = OneShotAudit.start(prompt, "query-file")
    assert audit is not None
    audit.bind_session("session-123")
    audit.finish(2, "validation_error")

    raw = (tmp_path / "logs" / "oneshot-audit.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert prompt not in raw
    assert [row["event"] for row in rows] == ["started", "finished"]
    assert {row["audit_id"] for row in rows} == {audit.audit_id}
    assert rows[0]["prompt_sha256"] == rows[1]["prompt_sha256"]
    assert rows[0]["prompt_chars"] == len(prompt)
    assert rows[1]["outcome"] == "validation_error"
    assert rows[1]["exit_code"] == 2
    assert rows[1]["session_id"] == "session-123"
    assert "argv" not in rows[0]


def test_concurrent_invocations_append_complete_correlated_records(tmp_path):
    prompts = [f"secret-{index}" for index in range(12)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        audit_ids = list(pool.map(_write_audit, [str(tmp_path)] * len(prompts), prompts, range(len(prompts))))

    rows = _records(tmp_path)
    assert len(rows) == len(prompts) * 2
    assert {row["audit_id"] for row in rows} == set(audit_ids)
    for audit_id in audit_ids:
        lifecycle = [row for row in rows if row["audit_id"] == audit_id]
        assert [row["event"] for row in lifecycle] == ["started", "finished"]
    raw = (tmp_path / "logs" / "oneshot-audit.jsonl").read_text(encoding="utf-8")
    assert not any(prompt in raw for prompt in prompts)