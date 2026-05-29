from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "signal_room_moho_job_runner.py"


def load_module():
    spec = importlib.util.spec_from_file_location("signal_room_moho_job_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_submit_pose_export_job_writes_moho_review_metadata(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "moho" / "jobs"
    project = tmp_path / "fee-machine.moho"
    output_dir = tmp_path / "pose-exports" / "Suit_Male"

    job = module.submit_pose_export_job(
        queue_root=queue_root,
        job_id="suit-male-pose-export",
        command=[r"C:\Program Files\Moho 14\Moho.exe", "--export", str(project)],
        cwd=tmp_path,
        project=project,
        output_dir=output_dir,
        candidate_name="Suit_Male",
        expected_frame_count=8,
        license_status="licensed internal review",
    )

    job_path = queue_root / "queued" / "suit-male-pose-export.json"
    written = json.loads(job_path.read_text())
    assert job["id"] == "suit-male-pose-export"
    assert written["job_type"] == "moho"
    assert written["status"] == "queued"
    assert written["public_release"] is False
    assert written["render_tool"] == "moho"
    assert written["candidate_name"] == "Suit_Male"
    assert written["expected_frame_count"] == 8
    assert written["license_status"] == "licensed internal review"
    assert written["project"] == str(project)
    assert written["output_dir"] == str(output_dir)


def test_submit_pose_export_job_rejects_unsafe_job_id(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "moho" / "jobs"

    try:
        module.submit_pose_export_job(
            queue_root=queue_root,
            job_id=r"..\escape",
            command=["Moho.exe", "--export", "scene.moho"],
            cwd=tmp_path,
            project=tmp_path / "scene.moho",
            output_dir=tmp_path / "frames",
            candidate_name="Suit_Male",
            expected_frame_count=8,
            license_status="licensed internal review",
        )
    except ValueError as exc:
        assert "job_id contains unsafe characters" in str(exc)
    else:
        raise AssertionError("unsafe job_id should be rejected")

    assert not (queue_root / "escape.json").exists()


def test_cli_submit_pose_export_rejects_unsafe_job_id_with_json_error(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "signal_room_moho_job_runner.py",
            "submit-pose-export",
            "--queue-root",
            str(tmp_path / "jobs"),
            "--job-id",
            "../escape",
            "--project",
            str(tmp_path / "scene.moho"),
            "--output-dir",
            str(tmp_path / "frames"),
            "--candidate-name",
            "Suit_Male",
            "--expected-frame-count",
            "8",
            "--license-status",
            "licensed internal review",
            "--",
            "Moho.exe",
        ],
    )

    assert module.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["error"] == "job_id contains unsafe characters; use letters, numbers, dot, underscore, or hyphen"


def test_moho_run_once_uses_shared_durable_queue(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "moho" / "jobs"
    module.submit_pose_export_job(
        queue_root=queue_root,
        job_id="moho-success",
        command=["Moho.exe", "--export", "scene.moho"],
        cwd=tmp_path,
        project=tmp_path / "scene.moho",
        output_dir=tmp_path / "frames",
        candidate_name="Suit_Male",
        expected_frame_count=8,
        license_status="licensed internal review",
    )
    calls = []

    def fake_runner(command, *, cwd, timeout_seconds):
        calls.append({"command": command, "cwd": cwd, "timeout_seconds": timeout_seconds})
        frames = tmp_path / "frames"
        frames.mkdir()
        for index in range(8):
            (frames / f"pose_{index:02d}.png").write_text("frame")
        return module.RunResult(returncode=0, stdout="pose export complete", stderr="")

    result = module.run_once(queue_root=queue_root, runner=fake_runner)

    assert result["processed"] is True
    assert result["status"] == "done"
    assert calls[0]["command"] == ["Moho.exe", "--export", "scene.moho"]
    done = json.loads((queue_root / "done" / "moho-success.json").read_text())
    assert done["status"] == "done"
    assert done["candidate_name"] == "Suit_Male"
    assert done["stdout_tail"] == "pose export complete"


def test_moho_success_requires_expected_output_directory(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "moho" / "jobs"
    output_dir = tmp_path / "missing-frames"
    module.submit_pose_export_job(
        queue_root=queue_root,
        job_id="moho-missing-output",
        command=["Moho.exe", "--export", "scene.moho"],
        cwd=tmp_path,
        project=tmp_path / "scene.moho",
        output_dir=output_dir,
        candidate_name="Suit_Male",
        expected_frame_count=8,
        license_status="licensed internal review",
    )

    def fake_runner(command, *, cwd, timeout_seconds):
        return module.RunResult(returncode=0, stdout="reported success", stderr="")

    result = module.run_once(queue_root=queue_root, runner=fake_runner)

    assert result["processed"] is True
    assert result["status"] == "failed"
    failed = json.loads((queue_root / "failed" / "moho-missing-output.json").read_text())
    assert failed["returncode"] == 2
    assert "missing expected output" in failed["stderr_tail"]
    assert str(output_dir) in failed["stderr_tail"]


def test_moho_success_requires_expected_frame_count(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "moho" / "jobs"
    output_dir = tmp_path / "short-frames"
    module.submit_pose_export_job(
        queue_root=queue_root,
        job_id="moho-short-frame-count",
        command=["Moho.exe", "--export", "scene.moho"],
        cwd=tmp_path,
        project=tmp_path / "scene.moho",
        output_dir=output_dir,
        candidate_name="Suit_Male",
        expected_frame_count=8,
        license_status="licensed internal review",
    )

    def fake_runner(command, *, cwd, timeout_seconds):
        output_dir.mkdir()
        for index in range(2):
            (output_dir / f"pose_{index:02d}.png").write_text("frame")
        return module.RunResult(returncode=0, stdout="reported success", stderr="")

    result = module.run_once(queue_root=queue_root, runner=fake_runner)

    assert result["processed"] is True
    assert result["status"] == "failed"
    failed = json.loads((queue_root / "failed" / "moho-short-frame-count.json").read_text())
    assert failed["returncode"] == 3
    assert "expected at least 8 pose frames" in failed["stderr_tail"]
    assert "found 2" in failed["stderr_tail"]


def test_write_moho_worker_bundle_points_to_moho_runner(tmp_path: Path) -> None:
    module = load_module()
    queue_root = tmp_path / "windows" / "moho" / "jobs"
    out_dir = tmp_path / "windows" / "moho"

    result = module.write_windows_worker_bundle(queue_root=queue_root, out_dir=out_dir)

    ps1 = out_dir / "run-moho-worker.ps1"
    readme = out_dir / "README.md"
    assert result["powershell"] == str(ps1)
    assert ps1.exists()
    assert readme.exists()
    text = ps1.read_text()
    assert str(queue_root) in text
    assert "signal_room_moho_job_runner.py" in text
