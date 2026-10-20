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


def _produce_step():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return next(
        step
        for step in workflow["jobs"]["generate-patch"]["steps"]
        if step.get("id") == "produce-patch"
    )


def _run_produce_step(work: Path, tmp_path: Path):
    """Run the produce step's own run block over the worktree's unstaged
    edits, the state `npm run fix` leaves behind."""
    output = tmp_path / f"out-{len(list(tmp_path.glob('out-*')))}"
    output.touch()
    env = {**os.environ, "GITHUB_OUTPUT": str(output)}
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _produce_step()["run"]],
        cwd=work, env=env, capture_output=True, text=True, timeout=60,
    )
    return result, output


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


@pytest.fixture
def repo(tmp_path):
    """A work clone of a bare remote holding the files a hostile patch would
    want to reach: manifests and lockfiles at root and nested, an eslint
    config, a tsconfig, a workflow file, sources."""
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
        "package-lock.json": "{}\n",
        "npm-shrinkwrap.json": "{}\n",
        "eslint.config.mjs": "export default []\n",
        "tsconfig.json": "{}\n",
        "apps/desktop/package.json": '{"name": "nested"}\n',
        ".github/workflows/existing.yml": "name: x\n",
        "docs/guide.md": "guide\n",
        ".gitignore": "build/\n",
        # The run block calls the checked-in gate script; the fixture repo
        # carries the real file so the apply step exercises genuine wiring.
        "scripts/ci/autofix_path_gate.py": (
            ROOT / "scripts/ci/autofix_path_gate.py"
        ).read_text(encoding="utf-8"),
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


def _make_patch(repo: Path, mutate, stage: bool = True) -> str:
    """Mutate the tree, capture the staged diff as a patch, restore.

    `-f` on the add mirrors a hostile produce job, which can stage ignored
    files too; `-x` on the clean removes them again. stage=False is for
    mutators that manipulate the index directly."""
    mutate(repo)
    if stage:
        _git(repo, "add", "-Af")
    # _git strips output; a patch needs its trailing newline to apply.
    patch = _git(repo, "diff", "--cached") + "\n"
    _git(repo, "reset", "-q", "--hard", "HEAD")
    _git(repo, "clean", "-fdxq")
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
        ["bash", "-eo", "pipefail", "-c", step["run"]],
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
        (r / "docs").mkdir(exist_ok=True)
        (r / "docs/note.md").write_text("x\n")
    patch = _make_patch(repo[0], write_note)
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 1
    assert "disallowed-extension" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_lockfiles_are_rejected(repo, tmp_path):
    """package-lock.json and npm-shrinkwrap.json land dependencies just as
    surely as the manifest does."""
    for rel in ("package-lock.json", "npm-shrinkwrap.json"):
        patch = _make_patch(repo[0], lambda r, rel=rel: (r / rel).write_text(
            '{"evil": true}\n'))
        result = _run_step(repo, tmp_path, patch)
        assert result.returncode == 1, rel
        assert "denied-path" in result.stdout
        assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_nested_and_case_variant_denied_paths(repo, tmp_path):
    """The deny list is not root-anchored and a hostile artifact controls
    casing."""
    for rel, content in (
        ("apps/desktop/package.json", '{"name": "evil"}\n'),
        ("apps/foo/.github/hook.ts", "export {}\n"),
        ("Package.json", '{"name": "evil"}\n'),
        ("ESLint.config.mjs", "export default [evil]\n"),
    ):
        patch = _make_patch(
            repo[0],
            lambda r, rel=rel, content=content: (
                (r / rel).parent.mkdir(parents=True, exist_ok=True),
                (r / rel).write_text(content),
            ),
        )
        result = _run_step(repo, tmp_path, patch)
        assert result.returncode == 1, rel
        assert "denied-path" in result.stdout
        assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_tool_configs_are_rejected(repo, tmp_path):
    """tsconfig and vitest/vite-style configs are executed by tooling; an
    artifact must not rewrite them."""
    for rel, content in (
        ("tsconfig.json", '{"compilerOptions": {"outDir": "/tmp"}}\n'),
        ("tsconfig.base.json", "{}\n"),
        ("vitest.config.ts", "export default {}\n"),
    ):
        patch = _make_patch(
            repo[0], lambda r, rel=rel, content=content: (r / rel).write_text(content))
        result = _run_step(repo, tmp_path, patch)
        assert result.returncode == 1, rel
        assert "denied-path" in result.stdout
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
def test_gitlink_patch_lands_nothing(repo, tmp_path):
    """`git apply` cannot materialize a gitlink into the worktree, so a
    160000 create in the artifact stages nothing and no-ops. The gate still
    lists the shape as refused in case a future apply gains --index."""
    def add_gitlink(r: Path):
        head = _git(r, "rev-parse", "HEAD")
        _git(r, "update-index", "--add", "--cacheinfo",
             f"160000,{head},modules/evil")
    patch = _make_patch(repo[0], add_gitlink, stage=False)
    assert "160000" in patch
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 0, result.stderr
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_staged_gate_rejects_a_gitlink(repo, tmp_path):
    """The --staged check itself refuses a gitlink if one ever gets staged."""
    work, _ = repo
    head = _git(work, "rev-parse", "HEAD")
    _git(work, "update-index", "--add", "--cacheinfo",
         f"160000,{head},modules/evil")
    result = subprocess.run(
        ["python3", "scripts/ci/autofix_path_gate.py", "--staged"],
        cwd=work, capture_output=True, text=True)
    assert result.returncode == 1
    assert "non-plain-change" in result.stderr
    _git(work, "reset", "-q", "--hard", "HEAD")


@pytest.mark.platforms("posix")
def test_unusual_filenames_reach_the_gate(repo, tmp_path):
    """Non-ASCII and whitespace-bearing names are allowed when they are
    ordinary sources; the -z enumeration must not mangle them into a
    rejection."""
    def write(r: Path):
        (r / "src/café.ts").write_text("export {}\n")
        (r / "src/has space.ts").write_text("export {}\n")
    patch = _make_patch(repo[0], write)
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 0, result.stdout or result.stderr
    assert _bot_branch(repo) != ""


@pytest.mark.platforms("posix")
def test_ignored_only_patch_is_a_noop(repo, tmp_path):
    """A patch whose every path is gitignored applies cleanly but stages
    nothing; that is a no-op, not a commit failure."""
    def ignored_only(r: Path):
        (r / "build").mkdir()
        (r / "build/out.ts").write_text("export {}\n")
    patch = _make_patch(repo[0], ignored_only)
    result = _run_step(repo, tmp_path, patch)
    assert result.returncode == 0, result.stderr
    assert "no tracked changes" in result.stdout
    assert _bot_branch(repo) == ""


@pytest.mark.platforms("posix")
def test_produce_step_drops_denied_paths(repo, tmp_path):
    """The artifact a hostile run uploads is filtered by the same policy the
    apply side enforces: denied edits are excluded from the patch entirely."""
    work, _ = repo
    (work / "package.json").write_text('{"name": "evil"}\n')
    (work / "src/app.ts").write_text("export const x=1\n")
    result, output = _run_produce_step(work, tmp_path)
    assert result.returncode == 0, result.stderr or result.stdout
    patch = (work / "js-fix.patch").read_text()
    assert "package.json" not in patch
    assert "src/app.ts" in patch
    assert "has-fixes=true" in output.read_text()


@pytest.mark.platforms("posix")
def test_produce_step_denied_only_is_noop(repo, tmp_path):
    work, _ = repo
    (work / "package.json").write_text('{"name": "evil"}\n')
    result, output = _run_produce_step(work, tmp_path)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "has-fixes=false" in output.read_text()
    assert (work / "js-fix.patch").read_text() == ""


@pytest.mark.platforms("posix")
def test_produce_step_rejects_disallowed_extension(repo, tmp_path):
    work, _ = repo
    (work / "docs/guide.md").write_text("tampered\n")
    result, _ = _run_produce_step(work, tmp_path)
    assert result.returncode == 1
    assert "docs/guide.md" in result.stdout


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
