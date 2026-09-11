"""Provenance records for CLI one-shot invocations."""

import json

from hermes_cli.oneshot import _write_oneshot_audit


def test_oneshot_audit_links_lifecycle_without_copying_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.oneshot.os.getpid", lambda: 101)
    monkeypatch.setattr("hermes_cli.oneshot.os.getppid", lambda: 99)
    monkeypatch.setattr("hermes_cli.oneshot._oneshot_parent_comm", lambda _ppid: "bash")
    monkeypatch.setattr("hermes_cli.oneshot.os.getcwd", lambda: "/work")

    _write_oneshot_audit("started", "audit-1", "read secret.txt")
    _write_oneshot_audit("finished", "audit-1", "read secret.txt", exit_code=0, session_id="session-1")

    path = tmp_path / "logs" / "oneshot-audit.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["started", "finished"]
    assert {row["audit_id"] for row in rows} == {"audit-1"}
    assert rows[0]["pid"] == 101
    assert rows[0]["ppid"] == 99
    assert rows[0]["parent_comm"] == "bash"
    assert rows[0]["cwd"] == "/work"
    assert rows[1]["session_id"] == "session-1"
    assert rows[0]["prompt_sha256"] == rows[1]["prompt_sha256"]
    assert "read secret.txt" not in path.read_text()
