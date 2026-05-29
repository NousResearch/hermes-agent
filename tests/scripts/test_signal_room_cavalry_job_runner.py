from __future__ import annotations

import importlib.util
import json
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
