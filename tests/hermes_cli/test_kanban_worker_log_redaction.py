from __future__ import annotations

import io
import os
import stat
import subprocess
import sys

from hermes_cli.worker_log import RedactingTextStream


def test_redacting_stream_handles_split_tokens_and_multiline_private_keys():
    output = io.StringIO()
    stream = RedactingTextStream(output)
    secret = "ghp_" + "A" * 40
    private_key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "sensitive-key-material\n"
        "-----END RSA PRIVATE KEY-----\n"
    )

    stream.write("ordinary output\ncredential: ghp_")
    stream.write("A" * 40 + "\n")
    for line in private_key.splitlines(keepends=True):
        stream.write(line)
    stream.finalize()

    stored = output.getvalue()
    assert "ordinary output" in stored
    assert secret not in stored
    assert "sensitive-key-material" not in stored
    assert "[REDACTED PRIVATE KEY]" in stored


def test_redacting_stream_drops_unterminated_private_key_material():
    output = io.StringIO()
    stream = RedactingTextStream(output)

    stream.write("-----BEGIN OPENSSH PRIVATE KEY-----\nraw-key-data")
    stream.finalize()

    assert output.getvalue() == "[REDACTED INCOMPLETE PRIVATE KEY]\n"


def test_main_startup_installs_redaction_for_kanban_worker(tmp_path):
    secret = "ghp_" + "B" * 40
    env = dict(os.environ)
    env["HERMES_HOME"] = str(tmp_path / ".hermes")
    env["HERMES_KANBAN_TASK"] = "t_worker"

    result = subprocess.run(  # noqa: S603 -- interpreter and inline code are fixed
        [sys.executable, "-c", f'import hermes_cli.main; print("{secret}")'],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert secret not in result.stdout


def _make_task(kb):
    return kb.Task(
        id="t_private_log",
        title="private log",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=1,
    )


def test_default_spawn_creates_owner_only_worker_log(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import profiles

    log_dir = tmp_path / "logs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: log_dir)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda home: None)
    monkeypatch.setattr(profiles, "resolve_profile_env", lambda profile: str(tmp_path))

    captured = {}

    class FakeProc:
        pid = 4321

    def fake_popen(cmd, *args, **kwargs):
        log_file = kwargs["stdout"]
        captured["mode"] = stat.S_IMODE(os.fstat(log_file.fileno()).st_mode)
        log_file.close()
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert kb._default_spawn(_make_task(kb), str(workspace)) == 4321
    assert captured["mode"] == 0o600


def test_rotation_hardens_existing_worker_log_and_backups(tmp_path):
    from hermes_cli import kanban_db as kb

    log_path = tmp_path / "worker.log"
    log_path.write_text("secret")
    os.chmod(log_path, 0o644)

    kb._rotate_worker_log(log_path, max_bytes=1, backup_count=1)

    rotated = tmp_path / "worker.log.1"
    assert stat.S_IMODE(rotated.stat().st_mode) == 0o600


def test_rotation_hardens_backup_without_rotating_current_log(tmp_path):
    from hermes_cli import kanban_db as kb

    log_path = tmp_path / "worker.log"
    backup_path = tmp_path / "worker.log.1"
    log_path.write_text("current")
    backup_path.write_text("historic secret")
    os.chmod(log_path, 0o644)
    os.chmod(backup_path, 0o644)

    kb._rotate_worker_log(log_path, max_bytes=1024, backup_count=1)

    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup_path.stat().st_mode) == 0o600
