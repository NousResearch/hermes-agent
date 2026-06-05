import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "scripts" / "runtime" / "codex_review_packet.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _metadata(packet: str) -> dict:
    metadata_text = packet.split("```json\n", 1)[1].split("\n```", 1)[0]
    return json.loads(metadata_text)


def test_review_packet_bounds_diff_and_marks_truncation(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    target = tmp_path / "demo.py"
    target.write_text("print('old')\n", encoding="utf-8")
    _git(tmp_path, "add", "demo.py")
    _git(tmp_path, "commit", "-m", "init")
    target.write_text("\n".join(f"print('line {i}')" for i in range(200)) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(PACKET),
            "--workdir",
            str(tmp_path),
            "--file",
            "demo.py",
            "--max-diff-chars",
            "500",
            "--max-total-chars",
            "1400",
        ],
        cwd=str(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    assert "# Bounded Codex review packet" in proc.stdout
    assert "demo.py" in proc.stdout
    assert "[truncated" in proc.stdout
    assert len(proc.stdout) <= 1500


def test_review_packet_includes_bounded_untracked_scope_file(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    untracked = tmp_path / "new_guard.py"
    untracked.write_text("UNTRACKED_SENTINEL = 'included'\n" + "x = 1\n" * 200, encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(PACKET),
            "--workdir",
            str(tmp_path),
            "--file",
            "new_guard.py",
            "--max-diff-chars",
            "500",
            "--max-total-chars",
            "2600",
        ],
        cwd=str(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    assert "## bounded untracked file previews" in proc.stdout
    assert "new_guard.py" in proc.stdout
    assert "UNTRACKED_SENTINEL" in proc.stdout
    assert "[truncated" in proc.stdout
    assert len(proc.stdout) <= 2700


def test_review_packet_v2_metadata_header_includes_candidate_and_scope_fields(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    target = tmp_path / "demo.py"
    target.write_text("print('old')\n", encoding="utf-8")
    _git(tmp_path, "add", "demo.py")
    _git(tmp_path, "commit", "-m", "init")
    target.write_text("print('new')\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(PACKET),
            "--workdir",
            str(tmp_path),
            "--file",
            "demo.py",
            "--allowed-file",
            "demo.py",
            "--allowed-glob",
            "tests/**/*.py",
            "--dirty-baseline",
            "M preexisting.py",
            "--test-run",
            "python -m pytest tests/scripts/test_codex_review_packet.py -q -o addopts='' => passed",
            "--candidate-id",
            "candidate-123",
            "--candidate-disposition",
            "pending_review",
            "--completion-trusted",
            "false",
            "--max-total-chars",
            "5000",
        ],
        cwd=str(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    assert "## Packet metadata header" in proc.stdout
    metadata = _metadata(proc.stdout)
    assert metadata == {
        "schema_version": "review_packet.v2",
        "touched_files": ["demo.py"],
        "allowed_files": ["demo.py"],
        "allowed_globs": ["tests/**/*.py"],
        "dirty_baseline": ["M preexisting.py"],
        "tests_run": ["python -m pytest tests/scripts/test_codex_review_packet.py -q -o addopts='' => passed"],
        "candidate_id": "candidate-123",
        "candidate_disposition": "pending_review",
        "completion_trusted": False,
    }


def test_review_packet_metadata_uses_untruncated_touched_files(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    for name in ("alpha_long_name.py", "beta_long_name.py"):
        target = tmp_path / name
        target.write_text("print('old')\n", encoding="utf-8")
    _git(tmp_path, "add", "alpha_long_name.py", "beta_long_name.py")
    _git(tmp_path, "commit", "-m", "init")
    for name in ("alpha_long_name.py", "beta_long_name.py"):
        (tmp_path / name).write_text("print('new')\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(PACKET),
            "--workdir",
            str(tmp_path),
            "--file",
            "alpha_long_name.py",
            "--file",
            "beta_long_name.py",
            "--max-name-chars",
            "20",
            "--max-total-chars",
            "5000",
        ],
        cwd=str(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    metadata = _metadata(proc.stdout)
    assert metadata["touched_files"] == ["alpha_long_name.py", "beta_long_name.py"]
    assert all("truncated" not in name for name in metadata["touched_files"])


def test_review_packet_defaults_allowed_files_to_unique_file_scope(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")

    proc = subprocess.run(
        [
            sys.executable,
            str(PACKET),
            "--workdir",
            str(tmp_path),
            "--file",
            "alpha.py",
            "--file",
            "beta.py",
            "--file",
            "alpha.py",
            "--max-total-chars",
            "5000",
        ],
        cwd=str(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    metadata = _metadata(proc.stdout)
    assert metadata["allowed_files"] == ["alpha.py", "beta.py"]
    assert metadata["allowed_globs"] == []


def test_review_packet_explicit_allowlist_wins_over_file_scope(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")

    proc = subprocess.run(
        [
            sys.executable,
            str(PACKET),
            "--workdir",
            str(tmp_path),
            "--file",
            "scope.py",
            "--allowed-file",
            "explicit.py",
            "--allowed-glob",
            "tests/**/*.py",
            "--max-total-chars",
            "5000",
        ],
        cwd=str(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    metadata = _metadata(proc.stdout)
    assert metadata["allowed_files"] == ["explicit.py"]
    assert metadata["allowed_globs"] == ["tests/**/*.py"]


def test_review_packet_metadata_touched_files_scopes_untracked_files(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    (tmp_path / "scoped_new.py").write_text("SCOPED = True\n", encoding="utf-8")
    (tmp_path / "outside_new.py").write_text("OUTSIDE = True\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(PACKET),
            "--workdir",
            str(tmp_path),
            "--file",
            "scoped_new.py",
            "--max-total-chars",
            "5000",
        ],
        cwd=str(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    metadata = _metadata(proc.stdout)
    assert metadata["touched_files"] == ["scoped_new.py"]


def test_review_packet_completion_trusted_unknown_is_null(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")

    proc = subprocess.run(
        [
            sys.executable,
            str(PACKET),
            "--workdir",
            str(tmp_path),
            "--candidate-id",
            "candidate-unknown",
            "--candidate-disposition",
            "pending_review",
            "--max-total-chars",
            "5000",
        ],
        cwd=str(tmp_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    metadata = _metadata(proc.stdout)
    assert metadata["candidate_id"] == "candidate-unknown"
    assert metadata["completion_trusted"] is None
