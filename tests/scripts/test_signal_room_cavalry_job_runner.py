from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_cavalry_job_runner.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_cavalry_job_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_submit_job_writes_review_only_queued_manifest(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"

    job = module.submit_job(
        queue_root=queue_root,
        job_id="fee-machine-motion-pass",
        command=[r"C:\Program Files\Cavalry\Cavalry.exe", "--run", "motion-pass.cav"],
        cwd=tmp_path,
        inputs=[tmp_path / "motion-pass.cav"],
        outputs=[tmp_path / "render.mov"],
    )

    job_path = queue_root / "queued" / "fee-machine-motion-pass.json"
    written = json.loads(job_path.read_text())
    assert job["id"] == "fee-machine-motion-pass"
    assert written["status"] == "queued"
    assert written["job_type"] == "cavalry"
    assert written["public_release"] is False
    assert written["attempts"] == 0
    assert written["command"][0].endswith("Cavalry.exe")


def test_submit_job_rejects_unsafe_job_id(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"

    try:
        module.submit_job(
            queue_root=queue_root,
            job_id="../escape",
            command=["Cavalry.exe", "--run", "scene.cav"],
            cwd=tmp_path,
        )
    except ValueError as exc:
        assert "job_id contains unsafe characters" in str(exc)
    else:
        raise AssertionError("unsafe job_id should be rejected")

    assert not (queue_root / "escape.json").exists()


def test_cli_submit_rejects_unsafe_job_id_with_json_error(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "signal_room_cavalry_job_runner.py",
            "submit",
            "--queue-root",
            str(tmp_path / "jobs"),
            "--job-id",
            "../escape",
            "--",
            "Cavalry.exe",
        ],
    )

    assert module.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["error"] == "job_id contains unsafe characters; use letters, numbers, dot, underscore, or hyphen"


def test_run_once_claims_job_and_records_success(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    module.submit_job(
        queue_root=queue_root,
        job_id="success-job",
        command=["Cavalry.exe", "--run", "scene.cav"],
        cwd=tmp_path,
    )
    calls = []

    def fake_runner(command, *, cwd, timeout_seconds):
        calls.append({"command": command, "cwd": cwd, "timeout_seconds": timeout_seconds})
        return module.RunResult(returncode=0, stdout="rendered", stderr="")

    result = module.run_once(queue_root=queue_root, runner=fake_runner)

    assert result["processed"] is True
    assert result["status"] == "done"
    assert calls == [
        {
            "command": ["Cavalry.exe", "--run", "scene.cav"],
            "cwd": str(tmp_path),
            "timeout_seconds": module.DEFAULT_TIMEOUT_SECONDS,
        }
    ]
    done_path = queue_root / "done" / "success-job.json"
    assert done_path.exists()
    done = json.loads(done_path.read_text())
    assert done["status"] == "done"
    assert done["returncode"] == 0
    assert done["stdout_tail"] == "rendered"
    assert not (queue_root / "running" / "success-job.json").exists()


def test_run_once_records_failed_job(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    module.submit_job(
        queue_root=queue_root,
        job_id="failed-job",
        command=["Cavalry.exe", "--run", "broken.cav"],
        cwd=tmp_path,
    )

    def fake_runner(command, *, cwd, timeout_seconds):
        return module.RunResult(returncode=9, stdout="", stderr="missing asset")

    result = module.run_once(queue_root=queue_root, runner=fake_runner)

    assert result["processed"] is True
    assert result["status"] == "failed"
    failed_path = queue_root / "failed" / "failed-job.json"
    failed = json.loads(failed_path.read_text())
    assert failed["status"] == "failed"
    assert failed["attempts"] == 1
    assert failed["returncode"] == 9
    assert failed["stderr_tail"] == "missing asset"


def test_run_once_records_runner_exception_as_failed_job(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    module.submit_job(
        queue_root=queue_root,
        job_id="exception-job",
        command=["Cavalry.exe", "--run", "broken.cav"],
        cwd=tmp_path,
    )

    def failing_runner(command, *, cwd, timeout_seconds):
        raise RuntimeError("cavalry crashed")

    result = module.run_once(queue_root=queue_root, runner=failing_runner)

    assert result["processed"] is True
    assert result["status"] == "failed"
    failed_path = queue_root / "failed" / "exception-job.json"
    failed = json.loads(failed_path.read_text())
    assert failed["status"] == "failed"
    assert failed["returncode"] == -1
    assert "cavalry crashed" in failed["stderr_tail"]
    assert not (queue_root / "running" / "exception-job.json").exists()


def test_run_once_fails_successful_job_with_missing_required_output(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    missing_output = tmp_path / "missing-render.mov"
    module.submit_job(
        queue_root=queue_root,
        job_id="missing-output-job",
        command=["Cavalry.exe", "--render", "scene.cav"],
        cwd=tmp_path,
        outputs=[missing_output],
        require_outputs=True,
    )

    def fake_runner(command, *, cwd, timeout_seconds):
        return module.RunResult(returncode=0, stdout="render complete", stderr="")

    result = module.run_once(queue_root=queue_root, runner=fake_runner)

    assert result["processed"] is True
    assert result["status"] == "failed"
    failed = json.loads((queue_root / "failed" / "missing-output-job.json").read_text())
    assert failed["returncode"] == 2
    assert "missing expected output" in failed["stderr_tail"]
    assert str(missing_output) in failed["stderr_tail"]


def test_run_once_fails_job_with_missing_required_input_before_launch(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    missing_input = tmp_path / "missing-scene.cav"
    module.submit_job(
        queue_root=queue_root,
        job_id="missing-input-job",
        command=["Cavalry.exe", "--render", str(missing_input)],
        cwd=tmp_path,
        inputs=[missing_input],
        require_inputs=True,
    )
    calls = []

    def fake_runner(command, *, cwd, timeout_seconds):
        calls.append(command)
        return module.RunResult(returncode=0, stdout="render complete", stderr="")

    result = module.run_once(queue_root=queue_root, runner=fake_runner)

    assert result["processed"] is True
    assert result["status"] == "failed"
    assert calls == []
    failed = json.loads((queue_root / "failed" / "missing-input-job.json").read_text())
    assert failed["returncode"] == 4
    assert "missing required input" in failed["stderr_tail"]
    assert str(missing_input) in failed["stderr_tail"]


def test_recover_stale_running_jobs_marks_old_claim_failed(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    module.submit_job(
        queue_root=queue_root,
        job_id="stale-running-job",
        command=["Cavalry.exe", "--render", "scene.cav"],
        cwd=tmp_path,
    )

    claimed = module.claim_next_job(queue_root)
    assert claimed is not None
    running_path, job = claimed
    job["claimed_at"] = "2026-05-29T00:00:00Z"
    running_path.write_text(json.dumps(job, indent=2) + "\n")

    result = module.recover_stale_running_jobs(
        queue_root=queue_root,
        max_age_seconds=60,
        now="2026-05-29T00:05:00Z",
    )

    assert result["recovered"] == 1
    failed_path = queue_root / "failed" / "stale-running-job.json"
    assert failed_path.exists()
    failed = json.loads(failed_path.read_text())
    assert failed["status"] == "failed"
    assert failed["returncode"] == -2
    assert "stale running job recovered" in failed["stderr_tail"]


def test_recover_stale_running_jobs_handles_invalid_running_manifest(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    for state in ("queued", "running", "done", "failed"):
        (queue_root / state).mkdir(parents=True)
    (queue_root / "running" / "invalid-running.json").write_text("{not json\n")

    result = module.recover_stale_running_jobs(
        queue_root=queue_root,
        max_age_seconds=60,
        now="2026-05-29T00:05:00Z",
    )

    assert result["recovered"] == 1
    failed_path = queue_root / "failed" / "invalid-running.json"
    assert failed_path.exists()
    failed = json.loads(failed_path.read_text())
    assert failed["id"] == "invalid-running"
    assert failed["status"] == "failed"
    assert failed["returncode"] == -1
    assert "invalid running job manifest" in failed["stderr_tail"]


def test_recover_stale_running_jobs_handles_non_object_running_manifest(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    for state in ("queued", "running", "done", "failed"):
        (queue_root / state).mkdir(parents=True)
    (queue_root / "running" / "array-running.json").write_text("[]\n")

    result = module.recover_stale_running_jobs(
        queue_root=queue_root,
        max_age_seconds=60,
        now="2026-05-29T00:05:00Z",
    )

    assert result["recovered"] == 1
    failed_path = queue_root / "failed" / "array-running.json"
    assert failed_path.exists()
    failed = json.loads(failed_path.read_text())
    assert failed["id"] == "array-running"
    assert failed["status"] == "failed"
    assert failed["returncode"] == -1
    assert "running job manifest must be a JSON object" in failed["stderr_tail"]


def test_run_once_records_empty_command_manifest_as_failed_job(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    queued_dir = queue_root / "queued"
    queued_dir.mkdir(parents=True)
    (queue_root / "running").mkdir()
    (queue_root / "done").mkdir()
    (queue_root / "failed").mkdir()
    (queued_dir / "empty-command.json").write_text(
        json.dumps(
            {
                "id": "empty-command",
                "job_type": "cavalry",
                "status": "queued",
                "public_release": False,
                "attempts": 0,
                "command": [],
            }
        )
        + "\n"
    )

    result = module.run_once(queue_root=queue_root)

    assert result["processed"] is True
    assert result["status"] == "failed"
    failed = json.loads((queue_root / "failed" / "empty-command.json").read_text())
    assert failed["status"] == "failed"
    assert failed["returncode"] == -1
    assert "command must contain at least one argument" in failed["stderr_tail"]
    assert not (queue_root / "running" / "empty-command.json").exists()


def test_run_once_records_invalid_json_manifest_as_failed_job(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    queued_dir = queue_root / "queued"
    queued_dir.mkdir(parents=True)
    (queue_root / "running").mkdir()
    (queue_root / "done").mkdir()
    (queue_root / "failed").mkdir()
    (queued_dir / "invalid-json.json").write_text("{not json\n")

    result = module.run_once(queue_root=queue_root)

    assert result["processed"] is True
    assert result["status"] == "failed"
    failed = json.loads((queue_root / "failed" / "invalid-json.json").read_text())
    assert failed["id"] == "invalid-json"
    assert failed["status"] == "failed"
    assert failed["returncode"] == -1
    assert "invalid queued job manifest" in failed["stderr_tail"]
    assert not (queue_root / "running" / "invalid-json.json").exists()


def test_run_once_records_non_object_manifest_as_failed_job(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    queued_dir = queue_root / "queued"
    queued_dir.mkdir(parents=True)
    (queue_root / "running").mkdir()
    (queue_root / "done").mkdir()
    (queue_root / "failed").mkdir()
    (queued_dir / "array-manifest.json").write_text("[]\n")

    result = module.run_once(queue_root=queue_root)

    assert result["processed"] is True
    assert result["status"] == "failed"
    failed = json.loads((queue_root / "failed" / "array-manifest.json").read_text())
    assert failed["id"] == "array-manifest"
    assert failed["status"] == "failed"
    assert failed["returncode"] == -1
    assert "queued job manifest must be a JSON object" in failed["stderr_tail"]


def test_write_windows_worker_bundle_points_to_queue_root(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "cavalry" / "jobs"
    out_dir = tmp_path / "windows" / "cavalry"

    result = module.write_windows_worker_bundle(queue_root=queue_root, out_dir=out_dir)

    ps1 = out_dir / "run-cavalry-worker.ps1"
    readme = out_dir / "README.md"
    assert result["powershell"] == str(ps1)
    assert ps1.exists()
    assert readme.exists()
    text = ps1.read_text()
    assert str(queue_root) in text
    assert "signal_room_cavalry_job_runner.py" in text
