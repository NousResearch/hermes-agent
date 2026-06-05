from __future__ import annotations

import json

from scripts.self_improvement import audit_memory_context as audit


FIXTURE = """
## Mnemosyne Context
  [2026-06-05T14:35] (importance 0.86) Context-hygiene rule: standalone user command fragments such as “proceed”, “commit”, “what next”, “make it happen”, or one-off task prompts should not be stored or injected as durable memory.
  [2026-06-05T14:35] (importance 0.30) [USER] You decide how you want to handle it and proceed
  [2026-06-04T07:23] (importance 0.30) [USER] proceed
  [2026-06-04T10:13] (importance 0.30) [USER] [IMPORTANT: Background process proc_edc70544f2c2 matched watch pattern "Startup done".
Command: cd /opt/fish-speech/repo && source .venv/bin/activate && python tools/api_server.py --listen 127.0.0.1:18080
Matched output:
Startup done]
  [2026-06-05T18:41] (importance 0.30) [USER] Review now I ran another task
  [2026-06-05T14:00] (importance 0.90) User prefers concise phase summaries and native file attachments when requested.
"""


def test_audit_text_flags_raw_user_fragments_but_keeps_durable_rule():
    report = audit.audit_text(FIXTURE)

    flagged = {item.content for item in report.candidates}
    assert "[USER] You decide how you want to handle it and proceed" in flagged
    assert "[USER] proceed" in flagged
    assert any("Startup done" in content for content in flagged)
    assert "[USER] Review now I ran another task" in flagged
    assert not any("Context-hygiene rule" in content for content in flagged)
    assert not any("concise phase summaries" in content for content in flagged)


def test_audit_text_records_reason_codes():
    report = audit.audit_text(FIXTURE)

    reasons_by_content = {item.content: set(item.reasons) for item in report.candidates}
    assert {"raw_user_fragment", "standalone_command_fragment"}.issubset(
        reasons_by_content["[USER] proceed"]
    )
    startup = next(item for item in report.candidates if "Startup done" in item.content)
    assert "background_process_fragment" in startup.reasons


def test_main_writes_jsonl_report_without_applying(tmp_path, capsys):
    input_path = tmp_path / "memory-context.txt"
    output_path = tmp_path / "memory_context_audit.jsonl"
    input_path.write_text(FIXTURE, encoding="utf-8")

    rc = audit.main(["--input", str(input_path), "--output", str(output_path)])

    assert rc == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["candidate_count"] == 4
    assert stdout["applied"] is False
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["kind"] == "memory_context_audit"
    assert rows[0]["candidate_count"] == 4
    assert rows[0]["candidates"][0]["content"]


def test_main_can_emit_suggested_invalidation_commands(tmp_path, capsys):
    input_path = tmp_path / "memory-context.txt"
    input_path.write_text(
        "[2026-06-04T07:23] (importance 0.30) id=abc123 [USER] proceed\n",
        encoding="utf-8",
    )

    rc = audit.main(["--input", str(input_path), "--commands"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "mnemosyne_validate" in output
    assert "abc123" in output


def test_parser_extracts_memory_id_when_present():
    entries = audit.parse_memory_context("[time] (importance 0.30) id=abc123 [USER] proceed")

    assert len(entries) == 1
    assert entries[0].memory_id == "abc123"
    assert entries[0].importance == 0.30
    assert entries[0].content == "[USER] proceed"
