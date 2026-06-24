"""Unit tests for scripts/identity_ledger.py.

Covers the mechanical rollup (kanban completion receipts -> LEDGER.md, with an
advancing, dedup-safe watermark) and the gated manual append (a receipt is
mandatory; prose-only entries are refused).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.identity_ledger as ledger
from hermes_cli import kanban_db as kb


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _complete_a_task(*, summary, metadata):
    conn = kb.connect()
    try:
        # Default initial_status ("running") + no parents resolves to "ready",
        # which claim_task can then transition to "running" (creating a run).
        tid = kb.create_task(conn, title="seed task", assignee="builder")
        kb.claim_task(conn, tid, claimer="builder")
        kb.complete_task(conn, tid, summary=summary, metadata=metadata)
        return tid
    finally:
        conn.close()


def test_rollup_appends_entry_with_receipts_and_advances_watermark(hermes_home):
    tid = _complete_a_task(
        summary="implemented the widget",
        metadata={"changed_files": ["a.py", "b.py"], "tests_run": ["pytest tests/x"]},
    )

    rc = ledger.main(["rollup"])
    assert rc == 0

    ledger_text = (hermes_home / "identity" / "LEDGER.md").read_text(encoding="utf-8")
    assert tid in ledger_text
    assert "implemented the widget" in ledger_text
    assert "changed_files: a.py, b.py" in ledger_text
    assert "tests_run: pytest tests/x" in ledger_text

    watermark = int((hermes_home / "identity" / ".ledger_watermark").read_text())
    assert watermark > 0


def test_bare_invocation_defaults_to_rollup(hermes_home):
    # The cron scheduler runs `python identity_ledger.py` with no args.
    tid = _complete_a_task(summary="bare run", metadata={"changed_files": ["z.py"]})
    assert ledger.main([]) == 0
    text = (hermes_home / "identity" / "LEDGER.md").read_text(encoding="utf-8")
    assert tid in text


def test_deployed_script_finds_repo_root_from_hermes_home(tmp_path):
    home = tmp_path / ".hermes"
    scripts_dir = home / "scripts"
    agent_root = home / "hermes-agent"
    scripts_dir.mkdir(parents=True)
    (agent_root / "hermes_cli").mkdir(parents=True)

    shutil.copyfile(Path(ledger.__file__), scripts_dir / "identity_ledger.py")
    (agent_root / "hermes_constants.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def get_hermes_home():\n"
        "    return Path(os.environ['HERMES_HOME'])\n",
        encoding="utf-8",
    )
    (agent_root / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    (agent_root / "hermes_cli" / "kanban_db.py").write_text(
        "class _Rows:\n"
        "    def fetchall(self):\n"
        "        return []\n"
        "class _Conn:\n"
        "    def execute(self, *args, **kwargs):\n"
        "        return _Rows()\n"
        "    def close(self):\n"
        "        pass\n"
        "def connect():\n"
        "    return _Conn()\n"
        "def get_run(conn, run_id):\n"
        "    return None\n",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "HERMES_HOME": str(home),
        "PYTHONPATH": "",
    }
    proc = subprocess.run(
        [sys.executable, str(scripts_dir / "identity_ledger.py"), "rollup"],
        text=True,
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr


def test_import_roots_include_deployed_agent_root(tmp_path):
    home = tmp_path / ".hermes"
    script = home / "scripts" / "identity_ledger.py"
    assert home / "hermes-agent" in ledger._candidate_import_roots(script, home)


def test_rollup_is_idempotent(hermes_home):
    _complete_a_task(summary="one", metadata={"changed_files": ["x.py"]})
    assert ledger.main(["rollup"]) == 0
    first = (hermes_home / "identity" / "LEDGER.md").read_text(encoding="utf-8")

    # No new completions -> no-op, no duplicate entries.
    assert ledger.main(["rollup"]) == 0
    second = (hermes_home / "identity" / "LEDGER.md").read_text(encoding="utf-8")
    assert first == second


def test_append_without_receipt_is_refused(hermes_home):
    rc = ledger.main(["append", "--summary", "I helped with stuff"])
    assert rc == 2
    assert not (hermes_home / "identity" / "LEDGER.md").exists()


def test_append_with_file_receipt_is_recorded(hermes_home, tmp_path):
    receipt = tmp_path / "proof.txt"
    receipt.write_text("evidence", encoding="utf-8")

    rc = ledger.main([
        "append", "--summary", "wrote a proof file", "--file", str(receipt),
    ])
    assert rc == 0
    text = (hermes_home / "identity" / "LEDGER.md").read_text(encoding="utf-8")
    assert "wrote a proof file" in text
    assert str(receipt) in text


def test_append_with_missing_file_receipt_is_refused(hermes_home, tmp_path):
    rc = ledger.main([
        "append", "--summary", "claimed a file", "--file", str(tmp_path / "nope.txt"),
    ])
    assert rc == 2
    assert not (hermes_home / "identity" / "LEDGER.md").exists()


def test_append_with_command_receipt_records_exit_code(hermes_home):
    rc = ledger.main([
        "append", "--summary", "ran a check", "--command", "echo hello-ledger",
    ])
    assert rc == 0
    text = (hermes_home / "identity" / "LEDGER.md").read_text(encoding="utf-8")
    assert "ran a check" in text
    assert "exit 0" in text
    assert "hello-ledger" in text
