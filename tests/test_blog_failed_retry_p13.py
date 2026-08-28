"""P13 isolation proof for scripts/blog-failed-retry.sh.

Verifies the rewritten wrapper:
- moves BLOG_RETRY_NOOP guard before mkdir (noop must not create dirs)
- runs synchronously (no backgrounding) and propagates the pipeline exit code
- captures the subprocess output without errexit aborting the script
- writes exactly one blog-failed-retry-status.json with parsed fields
- leaves the disposable engine tree byte-identical except for the status/log
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "blog-failed-retry.sh"


def _sha_tree(root: Path, ignore: set[str] | None = None) -> dict:
    ignore = ignore or set()
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            if any(rel.startswith(i) for i in ignore):
                continue
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _make_engine(tmp_path: Path, *, fake_rc=0, fake_out="") -> Path:
    engine = tmp_path / "engine" / "content_engine"
    engine.mkdir(parents=True)
    (engine / "output" / "logs").mkdir(parents=True)
    (engine / "blog").mkdir()
    # fake blog_pipeline executable that prints canned output and exits fake_rc
    fake = engine / "blog" / "fake_pipeline.py"
    lines = ["import sys", f"sys.stdout.write({fake_out!r})", f"sys.exit({fake_rc})"]
    fake.write_text("\n".join(lines) + "\n")
    return engine


def _run(engine: Path, *, noop=False, fake_cmd=None, env_extra=None):
    env = dict(os.environ)
    env["BLOG_RETRY_ENGINE_ROOT"] = str(engine)
    if fake_cmd is not None:
        env["BLOG_RETRY_PIPELINE_CMD"] = fake_cmd
    if noop:
        env["BLOG_RETRY_NOOP"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def test_noop_guard_before_mkdir(tmp_path):
    """BLOG_RETRY_NOOP must exit before creating any directories."""
    engine = tmp_path / "engine" / "content_engine"
    # Do NOT pre-create output/logs — the noop path must not create it either.
    engine.mkdir(parents=True)
    logs = engine / "output" / "logs"
    assert not logs.exists()
    r = _run(engine, noop=True)
    assert r.returncode == 0, r.stderr
    assert "noop" in r.stdout
    # No directories created by the noop path.
    assert not logs.exists(), "noop guard ran after mkdir and created dirs"


def test_synchronous_run_writes_status_and_propagates_rc(tmp_path):
    engine = _make_engine(
        tmp_path,
        fake_rc=0,
        fake_out="retry_all_pending_images: {recovered: [post-a], still_failed: [], no_draft: [], deferred: [], idle: []}",
    )
    fake_cmd = f"{sys.executable} {engine/'blog'/'fake_pipeline.py'}"
    before = _sha_tree(engine, ignore={"output/logs"})

    r = _run(engine, fake_cmd=fake_cmd)
    # Synchronous: exit code propagates from the pipeline (rc=0 here).
    assert r.returncode == 0, r.stderr

    status = engine / "output" / "logs" / "blog-failed-retry-status.json"
    assert status.exists(), "status JSON not written"
    payload = json.loads(status.read_text())
    assert payload["rc"] == 0
    assert payload["recovered"] == ["post-a"]
    assert payload["still_failed"] == []
    # Engine tree unchanged outside output/logs.
    after = _sha_tree(engine, ignore={"output/logs"})
    assert after == before, "engine tree mutated outside output/logs"


def test_nonzero_pipeline_exit_propagates_and_status_written(tmp_path):
    engine = _make_engine(
        tmp_path,
        fake_rc=3,
        fake_out="retry_all_pending_images: {recovered: [], still_failed: [post-b], no_draft: [], deferred: [cap], idle: []}",
    )
    fake_cmd = f"{sys.executable} {engine/'blog'/'fake_pipeline.py'}"
    r = _run(engine, fake_cmd=fake_cmd)
    assert r.returncode == 3, f"pipeline rc=3 must propagate; got {r.returncode}; stderr={r.stderr}"
    status = engine / "output" / "logs" / "blog-failed-retry-status.json"
    payload = json.loads(status.read_text())
    assert payload["rc"] == 3
    assert payload["still_failed"] == ["post-b"]
    assert payload["deferred"] == ["cap"]


def test_exactly_one_status_file(tmp_path):
    engine = _make_engine(tmp_path, fake_rc=0, fake_out="retry_all_pending_images: {idle: []}")
    fake_cmd = f"{sys.executable} {engine/'blog'/'fake_pipeline.py'}"
    _run(engine, fake_cmd=fake_cmd)
    status_files = list((engine / "output" / "logs").glob("blog-failed-retry-status.json"))
    assert len(status_files) == 1, f"expected exactly one status file, got {status_files}"


def test_secondary_refresh_is_persisted_and_primary_restored(tmp_path):
    """A rotated secondary refresh token must survive the temporary account swap."""
    home = tmp_path / "home"
    codex_dir = home / ".codex"
    codex_dir.mkdir(parents=True)
    primary = {"auth_mode": "chatgpt", "tokens": {"refresh_token": "primary-old"}}
    secondary = {"auth_mode": "chatgpt", "tokens": {"refresh_token": "secondary-old"}}
    auth = codex_dir / "auth.json"
    secondary_auth = codex_dir / "auth.json.secondary"
    auth.write_text(json.dumps(primary))
    secondary_auth.write_text(json.dumps(secondary))
    auth.chmod(0o600)
    secondary_auth.chmod(0o600)

    engine = _make_engine(tmp_path, fake_rc=0, fake_out="")
    state = tmp_path / "pipeline-call-count"
    fake = engine / "blog" / "fake_auth_pipeline.py"
    fake.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "state = Path(os.environ['FAKE_PIPELINE_STATE'])\n"
        "count = int(state.read_text()) if state.exists() else 0\n"
        "state.write_text(str(count + 1))\n"
        "if count == 0:\n"
        "    print('codex_capped')\n"
        "    raise SystemExit(3)\n"
        "auth = Path(os.environ['HOME']) / '.codex' / 'auth.json'\n"
        "auth.write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {'refresh_token': 'secondary-rotated'}}))\n"
        "auth.chmod(0o600)\n"
        "print('retry_all_pending_images: {recovered: [post-a], still_failed: [], no_draft: [], deferred: [], idle: []}')\n"
    )
    fake_cmd = f"{sys.executable} {fake}"

    r = _run(
        engine,
        fake_cmd=fake_cmd,
        env_extra={
            "HOME": str(home),
            "HERMES_HOME": str(tmp_path / "hermes-home"),
            "FAKE_PIPELINE_STATE": str(state),
        },
    )

    assert r.returncode == 0, r.stderr
    assert json.loads(auth.read_text()) == primary
    assert json.loads(secondary_auth.read_text())["tokens"]["refresh_token"] == "secondary-rotated"
    assert auth.stat().st_mode & 0o777 == 0o600
    assert secondary_auth.stat().st_mode & 0o777 == 0o600
    assert not (codex_dir / ".fallback_in_use").exists()
    assert not (codex_dir / "auth.json.primary-stash").exists()
