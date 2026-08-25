"""Regression: the TUI bundle must never be observable in a partial state.

Bug: launching several Hermes panes at once (an AgentGrid grid, a tmux layout,
or the dashboard Chat tab racing a terminal launch) made one pane rebuild
``ui-tui/dist/entry.js`` while another exec'd that same path. esbuild wrote
``outfile`` in place — truncating the existing ~3.7MB bundle to 0 bytes and
streaming it back — so the reading pane loaded a half-written module and died:

    SyntaxError: Unexpected end of input
        at compileSourceTextModule (node:internal/modules/esm/utils)

Two defences, one test each:
  1. ``ui-tui/scripts/build.mjs`` bundles to a temp file and ``rename``s it into
     place, so a concurrent reader sees old-or-new, never a prefix.
  2. ``hermes_cli.main._tui_build_lock`` serializes concurrent builds so N panes
     don't run N redundant esbuilds over the same output.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "ui-tui" / "scripts" / "build.mjs"


@pytest.fixture
def main_mod():
    import hermes_cli.main_tui_launch as m

    return m


# ---------------------------------------------------------------- build.mjs


def test_build_script_publishes_via_atomic_rename():
    """The bundler must not write dist/entry.js in place."""
    src = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "renameSync(tmp, out)" in src, (
        "build.mjs must publish the bundle with an atomic rename; writing "
        "esbuild's output directly to dist/entry.js lets a concurrently "
        "launching pane load a truncated module."
    )
    assert "outfile: tmp" in src, "esbuild must emit to the temp path, not the live bundle"
    assert "outfile: out" not in src, "esbuild must not write dist/entry.js in place"
    # A failed build must not strand temp bundles in dist/ (launcher hot path).
    assert "rmSync(tmp, { force: true })" in src


def test_build_script_strips_shebang_before_publishing():
    """The shebang rewrite must target the temp file, not the published bundle.

    Rewriting after the rename would reintroduce exactly the truncation window
    the rename exists to close.
    """
    src = BUILD_SCRIPT.read_text(encoding="utf-8")
    rewrite = src.index("writeFileSync(tmp,")
    publish = src.index("renameSync(tmp, out)")
    assert rewrite < publish, "shebang strip must happen before the atomic publish"
    assert "readFileSync(tmp," in src


def test_atomic_rename_keeps_concurrent_readers_valid(tmp_path):
    """A reader mid-rebuild sees a complete module under rename, not a prefix.

    Lockstep rather than timing-based: the writer pauses mid-write and the
    reader inspects exactly then, so the test can't flake on a fast disk.
    """
    node = _require_node()

    # Fixture must parse standalone (.mjs => ESM, unique bindings), otherwise
    # the probe would measure a broken fixture instead of write atomicity.
    body = "".join(f"const v{i} = {i};\n" for i in range(20000)) + "export default v0;\n"
    live = tmp_path / "entry.mjs"
    live.write_text(body, encoding="utf-8")
    assert _node_check(node, live) == 0, "fixture must be valid at rest"

    half = len(body) // 2

    # In-place truncate+stream (the OLD behaviour) => reader sees a prefix.
    fd = os.open(live, os.O_RDWR)
    try:
        os.ftruncate(fd, 0)
        os.write(fd, body[:half].encode())
        assert _node_check(node, live) != 0, (
            "sanity: an in-place partial write must be observably invalid, "
            "otherwise this test cannot detect the regression"
        )
    finally:
        os.close(fd)
    live.write_text(body, encoding="utf-8")

    # Temp + rename (the NEW behaviour) => reader still sees the old bundle.
    tmp = tmp_path / "entry.mjs.tmp-1"
    tmp.write_text(body[:half], encoding="utf-8")
    assert _node_check(node, live) == 0, "live bundle must stay valid mid-build"
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, live)
    assert _node_check(node, live) == 0, "published bundle must be valid"
    assert not tmp.exists()


def _require_node() -> str:
    from shutil import which

    node = which("node")
    if not node:
        pytest.skip("node not available")
    return node


def _node_check(node: str, path: Path) -> int:
    return subprocess.run(
        [node, "--check", str(path)], capture_output=True, text=True
    ).returncode


# ------------------------------------------------------------- build lock


_LOCK_HOLDER_SCRIPT = """
import sys, time
sys.path.insert(0, {repo!r})
import hermes_cli.main_tui_launch as m
from pathlib import Path

with m._tui_build_lock(Path(sys.argv[1])):
    print("ACQUIRED", flush=True)
    time.sleep(float(sys.argv[2]))
print("RELEASED", flush=True)
"""


def _spawn_lock_holder(tmp_path: Path, hold_seconds: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER_SCRIPT.format(repo=str(REPO_ROOT)),
         str(tmp_path), str(hold_seconds)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_tui_build_lock_is_exclusive_across_processes(main_mod, tmp_path):
    """Two processes must not hold the build lock simultaneously.

    Uses real subprocesses: flock is per-open-file-description, so a
    same-process check would not exercise the cross-pane behaviour that
    matters here.
    """
    _require_fcntl()

    holder = _spawn_lock_holder(tmp_path, hold_seconds=3.0)
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ACQUIRED"

        contender = _spawn_lock_holder(tmp_path, hold_seconds=0.0)
        try:
            # While the holder sleeps, the contender must be blocked.
            time.sleep(1.0)
            assert contender.poll() is None, (
                "second process acquired the build lock while the first held "
                "it — concurrent panes would run redundant esbuilds over one "
                "output file"
            )
            # Once the holder exits, the contender must proceed.
            holder.wait(timeout=15)
            out, err = contender.communicate(timeout=15)
            assert contender.returncode == 0, err
            assert "ACQUIRED" in out
        finally:
            if contender.poll() is None:
                contender.kill()
            contender.wait(timeout=10)
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait(timeout=10)


def _require_fcntl() -> None:
    try:
        import fcntl  # noqa: F401
    except ImportError:
        pytest.skip("flock unavailable (Windows); atomic rename still applies")


def test_tui_build_lock_releases_on_exception(main_mod, tmp_path):
    """A failed build must not leave the lock held (would wedge every pane)."""
    with pytest.raises(RuntimeError):
        with main_mod._tui_build_lock(tmp_path):
            raise RuntimeError("build blew up")

    # Re-acquire immediately; a leaked lock would block here.
    with main_mod._tui_build_lock(tmp_path):
        pass


def test_tui_build_lock_survives_unwritable_dir(main_mod, tmp_path):
    """Locking is best-effort: it must never prevent the TUI from launching."""
    target = tmp_path / "ro"
    target.mkdir()
    (target / "dist").mkdir()
    os.chmod(target / "dist", 0o500)
    try:
        with main_mod._tui_build_lock(target):
            pass  # must not raise
    finally:
        os.chmod(target / "dist", 0o700)


def test_build_lock_file_lives_in_ignored_dist(main_mod, tmp_path):
    """The lock artifact must not show up as untracked repo noise."""
    with main_mod._tui_build_lock(tmp_path):
        lock = tmp_path / "dist" / ".build.lock"
        assert lock.exists()
    # ui-tui/.gitignore ignores dist/ wholesale.
    assert "dist/" in (REPO_ROOT / "ui-tui" / ".gitignore").read_text()
