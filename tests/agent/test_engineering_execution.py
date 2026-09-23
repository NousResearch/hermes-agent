"""Real child processes must not inherit parent credentials or forge check receipts."""

import os
import sys

import pytest

from agent.engineering_execution import (
    CheckSpec,
    ExecutionUnavailable,
    execute_checks,
    run_host_command,
)
from agent.engineering_workflow import (
    ReceiptError,
    VerificationContext,
    verify_receipts,
)


def context():
    return VerificationContext(
        run_id="run",
        workspace_id="workspace",
        attempt_id="attempt",
        revision=1,
        snapshot_digest="digest",
        check_ids=("unit",),
    )


def test_native_child_and_grandchild_receive_no_parent_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTHETIC_PROVIDER_SECRET", "top-secret")
    script = (
        "import os,subprocess,sys;"
        "assert 'SYNTHETIC_PROVIDER_SECRET' not in os.environ;"
        "subprocess.run([sys.executable,'-c',"
        "\"import os; assert 'SYNTHETIC_PROVIDER_SECRET' not in os.environ\""
        "],check=True)"
    )
    result = run_host_command(
        (sys.executable, "-c", script), tmp_path, backend="native", timeout=20
    )
    assert result.exit_code == 0
    assert result.complete is True
    assert result.timed_out is False


def test_host_checks_issue_bound_receipts_from_real_processes(tmp_path):
    ctx = context()
    checks = (CheckSpec("unit", (sys.executable, "-c", "raise SystemExit(0)"), 20),)
    receipts = execute_checks(
        ctx,
        checks,
        tmp_path,
        backend="native",
        snapshot_digest=lambda: "digest",
    )
    assert verify_receipts(ctx, receipts) is True
    failed = execute_checks(
        ctx,
        (CheckSpec("unit", (sys.executable, "-c", "raise SystemExit(1)"), 20),),
        tmp_path,
        backend="native",
        snapshot_digest=lambda: "digest",
    )
    assert verify_receipts(ctx, failed) is False


def test_verifier_refuses_workspace_change_during_check(tmp_path):
    ctx = context()
    digests = iter(["digest", "changed"])
    with pytest.raises(ReceiptError):
        execute_checks(
            ctx,
            (CheckSpec("unit", (sys.executable, "-c", "pass"), 20),),
            tmp_path,
            backend="native",
            snapshot_digest=lambda: next(digests),
        )


def test_timeout_is_never_a_success_receipt(tmp_path):
    ctx = context()
    receipts = execute_checks(
        ctx,
        (
            CheckSpec(
                "unit", (sys.executable, "-c", "import time; time.sleep(3)"), 0.01
            ),
        ),
        tmp_path,
        backend="native",
        snapshot_digest=lambda: "digest",
    )
    with pytest.raises(ReceiptError):
        verify_receipts(ctx, receipts)


def test_unavailable_docker_stops_without_native_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.engineering_execution.shutil.which", lambda _: None)
    marker = tmp_path / "ran"
    with pytest.raises(ExecutionUnavailable):
        run_host_command(
            (sys.executable, "-c", f"open({str(marker)!r},'w').close()"),
            tmp_path,
            backend="docker",
            timeout=20,
        )
    assert not marker.exists()


def test_workspace_digest_tracks_content_changes(tmp_path):
    from agent.engineering_execution import workspace_digest

    target = tmp_path / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    first = workspace_digest(tmp_path)
    target.write_text("value = 2\n", encoding="utf-8")
    assert workspace_digest(tmp_path) != first


def test_docker_refuses_credential_file_mount_before_start(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.engineering_execution.shutil.which", lambda _: "docker")
    (tmp_path / ".env").write_text("SYNTHETIC_KEY=not-a-real-secret", encoding="utf-8")
    with pytest.raises(ExecutionUnavailable, match="credential file"):
        run_host_command(
            ("python", "-V"),
            tmp_path,
            backend="docker",
            image="local-test:1",
            timeout=20,
        )


def test_worker_output_is_bounded_and_known_secret_redacted(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTHETIC_PROVIDER_TOKEN", "synthetic-secret-123")
    output = run_host_command(
        (sys.executable, "-c", "print('synthetic-secret-123')"),
        tmp_path,
        backend="native",
        timeout=20,
    )
    assert "synthetic-secret-123" not in output.output
    assert "[redacted]" in output.output


def test_stop_request_terminates_active_native_child(tmp_path):
    import time

    started = time.monotonic()
    result = run_host_command(
        (sys.executable, "-c", "import time; time.sleep(10)"),
        tmp_path,
        backend="native",
        timeout=20,
        stop_requested=lambda: time.monotonic() - started > 0.1,
    )
    assert result.complete is False
    assert result.timed_out is False
    assert result.output == "command interrupted"
    assert time.monotonic() - started < 3


def test_docker_refuses_local_dependency_storage_mount(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.engineering_execution.shutil.which", lambda _: "docker")
    (tmp_path / ".venv").mkdir()
    with pytest.raises(ExecutionUnavailable, match="dependency storage"):
        run_host_command(
            ("python", "-V"),
            tmp_path,
            backend="docker",
            image="local-test:1",
            timeout=20,
        )


def test_interrupted_docker_attempt_removes_named_container(monkeypatch, tmp_path):
    from types import SimpleNamespace

    commands = []

    class FakeProcess:
        returncode = None
        pid = -1

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    def start(command, **kwargs):
        commands.append(command)
        return FakeProcess()

    def cleanup(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("agent.engineering_execution.shutil.which", lambda _: "docker")
    monkeypatch.setattr("agent.engineering_execution.subprocess.Popen", start)
    monkeypatch.setattr("agent.engineering_execution.subprocess.run", cleanup)
    monkeypatch.setattr(
        "agent.engineering_execution._stop_process_tree",
        lambda process: setattr(process, "returncode", -1),
    )
    result = run_host_command(
        ("python", "-V"),
        tmp_path,
        backend="docker",
        image="local-test:1",
        timeout=20,
        stop_requested=lambda: True,
    )
    assert result.complete is False
    name = commands[0][commands[0].index("--name") + 1]
    assert name.startswith("hermes-engineering-")
    assert commands[1] == ["docker", "rm", "-f", name]
