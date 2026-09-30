"""The apply-patch job must re-vet the artifact on the privileged side.

generate-patch runs `npm run fix`, which executes the repo's own code and
every installed eslint plugin. Anything that job emits is untrusted: its own
extension filter is not evidence, because a hostile run controls both the
patch and the filtering. The apply job therefore enumerates the staged diff
itself and refuses anything beyond plain JS/TS/JSON content changes.

These tests run the step's own `run:` block verbatim against a fixture repo
with a bare origin remote, feeding it real `git diff`-produced patches.
"""
import os
from pathlib import Path
import subprocess

import pytest
import hermes_yaml as yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/js-autofix.yml"


def _apply_step():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return next(
        step
        for step in workflow["jobs"]["apply-patch"]["steps"]
        if "git apply" in str(step.get("run", ""))
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


@pytest.fixture
def repo(tmp_path):
    """A work clone of a bare remote holding the files a hostile patch would
    want to reach: a manifest, an eslint config, a workflow file, sources."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(bare), str(work)], check=True)
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    for rel, content in {
        "src/app.ts": "export const x = 1\n",
        "src/util.js": "module.exports = {}\n",
        "package.json": '{"name": "fixture"}\n',
        "eslint.config.mjs": "export default []\n",
        ".github/workflows/existing.yml": "name: x\n",
    }.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "init")
    _git(work, "branch", "-M", "main")
    _git(work, "push", "-q", "origin", "main")
    _git(bare, "symbolic-ref", "HEAD", "refs/heads/main")
    return work, bare


def _make_patch(repo: Path, mutate) -> str:
    """Mutate the tree, capture the staged diff as a patch, restore."""
    mutate(repo)
    _git(repo, "add", "-A")
    # _git strips output; a patch needs its trailing newline to apply.
    patch = _git(repo, "diff", "--cached") + "\n"
    _git(repo, "reset", "-q", "--hard", "HEAD")
    _git(repo, "clean", "-fdq")
    return patch


def _run_step(repo: Path, tmp_path: Path, patch: str):
    work, _ = repo
    runner = tmp_path / f"runner-{len(list(tmp_path.glob('runner-*')))}"
    runner.mkdir()
    (runner / "js-fix.patch").write_text(patch)
    step = _apply_step()
    env = {
        **os.environ,
        **{k: str(v) for k, v in (step.get("env") or {}).items()},
        "RUNNER_TEMP": str(runner),
    }
    return subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _bot_branch(repo) -> str:
    _, bare = repo
    return _git(bare, "branch", "--list", "bot/js-autofix")


@pytest.mark.platforms("posix")
def test_empty_patch_is_a_noop(repo, tmp_path):
    result = _run_step(repo, tmp_path, "")
    assert result.returncode == 0, result.stderr
    assert "Patch is empty" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_legit_source_fix_pushes_bot_branch(repo, tmp_path):
    patch = _make_patch(repo[0], lambda r: (r / "src/app.ts").write_text(
        "export const x=1\n"))
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 0, result.stderr or result.stdout
    assert _bot_branch(repo) != ""


@pytest.mark.platforms("posix")
def test_new_json_source_file_is_allowed(repo, tmp_path):
    patch = _make_patch(repo[0], lambda r: (r / "src/config.json").write_text("{}\n"))
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 0, result.stderr or result.stdout
    assert _bot_branch(repo) != ""


@pytest.mark.platforms("posix")
def test_workflow_file_change_is_rejected(repo, tmp_path):
    """Even inside an allowed extension, .github/ stays denied."""
    patch = _make_patch(repo[0], lambda r: (
        r / ".github/workflows/evil.ts").write_text("export {}\n"))
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert "denied-path" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_manifest_change_is_rejected(repo, tmp_path):
    patch = _make_patch(repo[0], lambda r: (r / "package.json").write_text(
        '{"name": "fixture", "scripts": {"postinstall": "curl evil"}}\n'))
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert "denied-path" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_eslint_config_change_is_rejected(repo, tmp_path):
    patch = _make_patch(repo[0], lambda r: (r / "eslint.config.mjs").write_text(
        "export default [{rules: {}}]\n"))
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert "denied-path" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_disallowed_extension_is_rejected(repo, tmp_path):
    def write_note(r: Path):
        (r / "docs").mkdir()
        (r / "docs/note.md").write_text("x\n")
    patch = _make_patch(repo[0], write_note)
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert "disallowed-extension" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_deletion_is_rejected(repo, tmp_path):
    patch = _make_patch(repo[0], lambda r: (r / "src/app.ts").unlink())
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert "unsupported-change D" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_mode_change_is_rejected(repo, tmp_path):
    patch = _make_patch(repo[0], lambda r: (r / "src/app.ts").chmod(0o755))
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert "non-plain-change" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_symlink_is_rejected(repo, tmp_path):
    patch = _make_patch(
        repo[0], lambda r: (r / "src/link.ts").symlink_to("app.ts"))
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_rename_is_rejected(repo, tmp_path):
    def rename(r: Path):
        (r / "src/app.ts").rename(r / "src/renamed.ts")
    patch = _make_patch(repo[0], rename)
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_mixed_legit_and_denied_is_rejected(repo, tmp_path):
    """One allowed edit cannot launder a denied one."""
    def both(r: Path):
        (r / "src/app.ts").write_text("export const x=1\n")
        (r / "package.json").write_text('{"name": "evil"}\n')
    patch = _make_patch(repo[0], both)
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert "denied-path" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_path_escape_patch_is_rejected(repo, tmp_path):
    """A hand-crafted `../` path never reaches apply: git refuses it."""
    patch = (
        "diff --git a/../escape.txt b/../escape.txt\n"
        "new file mode 100644\n"
        "index 0000000..ce01362\n"
        "--- /dev/null\n"
        "+++ b/../escape.txt\n"
        "@@ -0,0 +1 @@\n"
        "+hello\n"
    )
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert _bot_branch(repo) == ""
