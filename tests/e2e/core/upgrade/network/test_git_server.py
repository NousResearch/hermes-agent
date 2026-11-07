"""The fake forge's git server serves what a real partial-clone checkout asks for."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tests.e2e.core.upgrade.network import _netedge as N

pytestmark = [pytest.mark.skipif(shutil.which("git") is None, reason="git required")]


def _git(*args: str, cwd) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_blobless_checkout_fetches_its_blobs_in_one_gzipped_request(tmp_path):
    # A checkout of thousands of missing blobs makes git gzip the upload-pack request body.
    src = tmp_path / "src"
    src.mkdir()
    _git("init", "-q", "-b", "main", cwd=src)
    for i in range(3000):
        (src / f"f{i}.txt").write_text(f"{i}\n", encoding="utf-8")
    _git("add", ".", cwd=src)
    _git("-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "commit", "-qm", "files", cwd=src)
    bare = tmp_path / "forge" / "NousResearch" / "hermes-agent.git"
    bare.parent.mkdir(parents=True)
    _git("clone", "-q", "--bare", str(src), str(bare), cwd=tmp_path)
    _git("config", "uploadpack.allowFilter", "true", cwd=bare)
    git_app = N.git_app(tmp_path / "forge")
    encodings = []

    def recording_app(req):
        encodings.append(req.headers.get("content-encoding", ""))
        return git_app(req)

    site = N.HttpSite(recording_app)
    try:
        cp = subprocess.run(["git", "clone", "-q", "--filter=blob:none", f"{site.url}/NousResearch/hermes-agent.git",
                             str(tmp_path / "checkout")], capture_output=True, text=True, timeout=300, check=False)
    finally:
        site.close()
    failed = [str(h) for h in site.hits if h.status != 200]
    assert cp.returncode == 0 and not failed, f"{cp.stderr}\n{failed}"
    assert "gzip" in encodings, f"git sent no gzipped request, so this test does not cover that path: {encodings}"
    assert (tmp_path / "checkout" / "f2999.txt").read_text(encoding="utf-8") == "2999\n"
