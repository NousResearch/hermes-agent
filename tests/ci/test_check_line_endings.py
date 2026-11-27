"""scripts/ci/check_line_endings.py on real temp git repos: a CRLF blob that reached the index."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci" / "check_line_endings.py"
sys.path.insert(0, str(SCRIPT.parent))

import check_line_endings as cle


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    proc = subprocess.run(["git", *args], cwd=repo, env=env, input=stdin, capture_output=True,
                          timeout=60, check=False)
    assert proc.returncode == 0, (args, proc.stderr)
    return proc.stdout.decode().strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".gitattributes").write_text("* text=auto eol=lf\n*.ps1 text eol=crlf\n*.bin -text\n")
    (repo / "ok.yaml").write_bytes(b"a: 1\nb: 2\n")
    (repo / "run.ps1").write_bytes(b"Write-Host hi\r\n")  # stored LF, checked out CRLF: allowed
    (repo / "blob.bin").write_bytes(b"x\r\ny\r\n")  # binary: never judged
    _git(repo, "add", ".")
    return repo


def _stage_raw(repo: Path, rel: str, data: bytes) -> None:
    """Put a CRLF blob in the index the way an attribute-blind client (web upload, API) does."""
    sha = _git(repo, "hash-object", "-w", "--no-filters", "--stdin", stdin=data)
    _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{sha},{rel}")
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_bytes(data)


def test_a_crlf_blob_fails_and_only_it_is_named(tmp_path, capsys):
    repo = _repo(tmp_path)
    assert cle.main(["--root", str(repo)]) == 0
    _stage_raw(repo, "catalog/entry.yaml", b"name: x\r\nsha: y\r\n")
    assert cle.main(["--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "catalog/entry.yaml" in out
    assert "run.ps1" not in out and "blob.bin" not in out


def test_fix_stages_an_lf_copy_that_then_passes(tmp_path):
    repo = _repo(tmp_path)
    _stage_raw(repo, "contrib/email", b"someone\r\n# PR #1\r\n")
    assert cle.main(["--root", str(repo), "--fix"]) == 0
    assert cle.main(["--root", str(repo)]) == 0
    assert _git(repo, "show", ":contrib/email") == "someone\n# PR #1"
