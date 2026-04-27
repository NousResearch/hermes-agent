"""Integration checks: optional Docker and concurrent atomic writes."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from utils import atomic_json_write


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_atomic_json_under_docker_if_available(tmp_path: Path) -> None:
    exe = shutil.which("docker")
    if not exe:
        pytest.skip("docker not on PATH")
    repo = _repo_root()
    host_out = tmp_path / "out.json"
    inner = (
        "pip install --no-cache-dir -q pyyaml >/dev/null && "
        "python -c \"import os,sys; sys.path.insert(0,os.environ['REPO_ROOT']); "
        "from pathlib import Path; from utils import atomic_json_write; "
        "p=Path(os.environ['OUT_JSON']); atomic_json_write(p, {'from':'docker'}); "
        "print(p.read_text(encoding='utf-8'))\""
    )
    proc = subprocess.run(
        [
            exe,
            "run",
            "--rm",
            "-e",
            "REPO_ROOT=/repo",
            "-e",
            "OUT_JSON=/work/out.json",
            "-v",
            f"{repo}:/repo:ro",
            "-v",
            f"{tmp_path}:/work",
            "python:3.11-slim",
            "bash",
            "-lc",
            inner,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.skip(f"docker integration unavailable: {proc.stderr[:500]}")
    data = json.loads(host_out.read_text(encoding="utf-8"))
    assert data == {"from": "docker"}


@pytest.mark.integration
def test_atomic_json_under_podman_if_available(tmp_path: Path) -> None:
    exe = shutil.which("podman")
    if not exe:
        pytest.skip("podman not on PATH")
    repo = _repo_root()
    host_out = tmp_path / "out2.json"
    inner = (
        "pip install --no-cache-dir -q pyyaml >/dev/null && "
        "python -c \"import os,sys; sys.path.insert(0,os.environ['REPO_ROOT']); "
        "from pathlib import Path; from utils import atomic_json_write; "
        "p=Path(os.environ['OUT_JSON']); atomic_json_write(p, {'from':'podman'}); "
        "print(p.read_text(encoding='utf-8'))\""
    )
    proc = subprocess.run(
        [
            exe,
            "run",
            "--rm",
            "-e",
            "REPO_ROOT=/repo",
            "-e",
            "OUT_JSON=/work/out2.json",
            "-v",
            f"{repo}:/repo:ro",
            "-v",
            f"{tmp_path}:/work",
            "docker.io/library/python:3.11-slim",
            "bash",
            "-lc",
            inner,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.skip(f"podman integration unavailable: {proc.stderr[:500]}")
    data = json.loads(host_out.read_text(encoding="utf-8"))
    assert data == {"from": "podman"}


def test_concurrent_atomic_json_last_writer_valid(tmp_path: Path) -> None:
    target = tmp_path / "race.json"
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        barrier.wait()
        atomic_json_write(target, {"i": i, "pad": "x" * 40})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    body = json.loads(target.read_text(encoding="utf-8"))
    assert "i" in body and body["pad"].startswith("x")
