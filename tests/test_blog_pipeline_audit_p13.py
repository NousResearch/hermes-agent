"""P13 isolation proof for scripts/blog-pipeline-audit.sh +
content_engine/tools/blog_pipeline_audit.py.

Verifies --engine-root / --blog-root overrides (CLI + env) let the audit
run against a disposable temp tree without touching the real repos. The
wrapper shell forwards the overrides. Fixture inputs are byte-identical
before/after (audit must be read-only).
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "blog-pipeline-audit.sh"
AUDIT = REPO_ROOT / "content_engine" / "tools" / "blog_pipeline_audit.py"


def _sha_tree(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root)
            out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _make_git_repo(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / ".gitkeep").write_text("")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)


def _make_fixture(tmp_path: Path):
    # ENGINE (per the script) = content_engine dir; its .parent is the git repo.
    repo = tmp_path / "repo"
    engine = repo / "content_engine"
    engine.mkdir(parents=True)
    blog = tmp_path / "blog"
    (engine / "tools").mkdir(parents=True)
    # copy the audit script into the fixture so ENGINE resolves correctly
    shutil.copy(AUDIT, engine / "tools" / "blog_pipeline_audit.py")
    bt = engine / "blog_topics"
    bt.mkdir()
    (bt / "pending_approvals.jsonl").write_text(
        '{"slug":"alpha","status":"pending","mdx_path":"'
        + str(blog / "src/content/blog/alpha.mdx")
        + '","preview_path":"' + str(blog / "preview/alpha.png") + '"}\n'
    )
    (bt / "failed_images.jsonl").write_text(
        '{"slug":"beta","attempts":1,"date":"2026-07-01","last_error":"boom"}\n'
    )
    (bt / "published_exempt.jsonl").write_text('{"slug":"exempt"}\n')
    (engine / "output" / "logs").mkdir(parents=True)
    # blog tree
    posts = blog / "src" / "content" / "blog"
    posts.mkdir(parents=True)
    (posts / "alpha.mdx").write_text('---\ntitle: "Alpha"\napproved: "false"\n---\nbody\n')
    (posts / "beta.mdx").write_text('---\ntitle: "Beta"\napproved: "true"\n---\nbody\n')
    (blog / "preview").mkdir()
    (blog / "preview" / "alpha.png").write_bytes(b"\x89PNG fake")
    _make_git_repo(repo)
    _make_git_repo(blog)
    return engine, blog


def _run_wrapper(engine: Path, blog: Path, via: str):
    env = dict(os.environ)
    env["BLOG_AUDIT_ENGINE_ROOT"] = str(engine)
    env["BLOG_AUDIT_BLOG_ROOT"] = str(blog)
    # The wrapper cds into REPO_ROOT; we run the python directly with overrides
    # to mirror what the wrapper does, but ALSO assert the wrapper forwards.
    if via == "env":
        return subprocess.run(
            [sys.executable, str(AUDIT)],
            capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
        )
    # CLI
    env.pop("BLOG_AUDIT_ENGINE_ROOT", None)
    env.pop("BLOG_AUDIT_BLOG_ROOT", None)
    return subprocess.run(
        [sys.executable, str(AUDIT), "--engine-root", str(engine), "--blog-root", str(blog)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


@pytest.mark.parametrize("via", ["env", "cli"])
def test_audit_uses_overrides_and_preserves_fixture(tmp_path, via):
    engine, blog = _make_fixture(tmp_path)
    before_engine = _sha_tree(engine)
    before_blog = _sha_tree(blog)

    r = _run_wrapper(engine, blog, via)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    # The audit should report problems found in the disposable tree only.
    # alpha is approved:false with a tracker entry (pending) -> no "no entry" issue;
    # beta approved:true -> not flagged. failed_images beta -> flagged.
    assert "beta" in out, f"expected beta flagged in stdout: {out!r}"
    # Must not touch the real repos — fixture byte-identical.
    assert _sha_tree(engine) == before_engine, "engine tree mutated"
    assert _sha_tree(blog) == before_blog, "blog tree mutated"


def test_wrapper_shell_forwards_overrides(tmp_path):
    """The wrapper blog-pipeline-audit.sh must honour the env overrides."""
    engine, blog = _make_fixture(tmp_path)
    env = dict(os.environ)
    env["BLOG_AUDIT_ENGINE_ROOT"] = str(engine)
    env["BLOG_AUDIT_BLOG_ROOT"] = str(blog)
    r = subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )
    assert r.returncode == 0, r.stderr
    assert "beta" in r.stdout, f"wrapper did not forward overrides: {r.stdout!r}"
