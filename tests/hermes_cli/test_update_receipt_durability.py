"""Receipt recovery must preserve terminal and user-action facts across processes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, body: str) -> dict:
    home = tmp_path / "account"
    state = home / ".hermes"
    state.mkdir(parents=True)
    script = tmp_path / "receipt_probe.py"
    script.write_text(textwrap.dedent("""
        import json, os, subprocess, sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from hermes_cli import update_receipt as receipts
        from hermes_cli.process_identity import _process_create_time
    """) + textwrap.dedent(body), encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HERMES_", "PYTEST_"))}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state),
               HERMES_RUNTIME_DIR=str(tmp_path / "tools"),
               XDG_CONFIG_HOME=str(home / ".config"), XDG_CACHE_HOME=str(home / ".cache"),
               PYTHONPATH=os.pathsep.join([str(ROOT), *filter(None, sys.path)]),
               PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, str(script), str(ROOT)], env=env, cwd=tmp_path,
                          text=True, capture_output=True, timeout=60, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))


def test_nested_begin_does_not_reconcile_live_outer_receipt(tmp_path):
    result = _run(tmp_path, """
        receipts.begin_update_receipt()
        outer = dict(receipts._current.get().data)
        receipts.begin_update_receipt()
        archive = receipts._run_file(receipts._receipt_dir(), outer)
        before = json.loads(archive.read_text())
        receipts.finalize_pending_update_receipt(0)
        restored = receipts.current_correlation_id()
        receipts.finalize_pending_update_receipt(0)
        Path('result.json').write_text(json.dumps({
            'outcome': before['outcome'], 'restored': restored, 'outer': outer['update_id']}))
    """)
    assert result["outcome"] == "running"
    assert result["restored"] == result["outer"]


def test_writer_identity_cannot_resurrect_a_mismatched_owner(tmp_path):
    result = _run(tmp_path, """
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            ct = _process_create_time(child.pid)
            assert ct is not None
            same = {'pid': child.pid, 'writer_pid': child.pid, 'pid_create_time': ct - 60}
            writer = {'pid': 0, 'writer_pid': child.pid, 'writer_create_time': ct - 60}
            control = {'pid': child.pid, 'writer_pid': child.pid, 'pid_create_time': ct}
            Path('result.json').write_text(json.dumps({
                'same': receipts._owner_alive(same), 'writer': receipts._owner_alive(writer),
                'control': receipts._owner_alive(control)}))
        finally:
            child.terminate()
            child.wait(timeout=10)
    """)
    assert result == {"same": False, "writer": False, "control": True}


def test_user_action_survives_interrupt_before_next_stage(tmp_path):
    result = _run(tmp_path, """
        receipts.begin_update_receipt()
        receipts.record_user_action('local_changes', 'Reapply stash@{0} by hand')
        running = receipts.read_latest_receipt()
        receipts.finalize_interrupted_update_receipt('operator interrupted')
        terminal = receipts.read_latest_receipt()
        Path('result.json').write_text(json.dumps({'running': running, 'terminal': terminal}))
    """)
    expected = {"step": "local_changes", "reason": "Reapply stash@{0} by hand"}
    assert result["running"].get("user_action") == expected
    assert result["terminal"].get("user_action") == expected
    assert result["terminal"]["outcome"] == "interrupted"


def test_parent_followup_does_not_replace_child_terminal_with_running(tmp_path):
    result = _run(tmp_path, """
        receipts.begin_update_receipt()
        frozen = dict(receipts._current.get().data)
        Path('request.json').write_text(json.dumps(frozen))
        child = subprocess.run([sys.executable, '-c',
            'import json,sys; from pathlib import Path; '
            'from hermes_cli.update_completion import _resume_receipt; '
            'from hermes_cli.update_receipt import finalize_pending_update_receipt; '
            '_resume_receipt(json.loads(Path("request.json").read_text())); '
            'finalize_pending_update_receipt(0)'], check=True)
        before = receipts.read_latest_receipt()
        receipts.record_followup('windows_resume', 'recovery refused')
        receipts.amend_terminal_followup(frozen['update_id'], 'windows_resume', 'recovery refused')
        after = receipts.read_latest_receipt()
        archive = json.loads(receipts._run_file(receipts._receipt_dir(), frozen).read_text())
        Path('result.json').write_text(json.dumps({'before': before, 'after': after, 'archive': archive}))
    """)
    assert result["before"]["outcome"] == "success"
    for terminal in (result["after"], result["archive"]):
        assert terminal["outcome"] == "success"
        assert terminal["finished_at"] == result["before"]["finished_at"]
        assert [row["step"] for row in terminal["followups"]] == ["windows_resume"]
