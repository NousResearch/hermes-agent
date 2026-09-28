"""The catalog admission gate must validate the pinned clone's own files only.

plugin-catalog-ci.yml joins a PR-controlled ``subdir`` onto the pinned clone before
checking the manifest and running ``hermes plugins validate``. These tests execute the
real ``run:`` block of that step against a fixture git repo (served via a git
``url.insteadOf`` rewrite): a ``..`` traversal, a ``file://`` repo, a non-hex ``sha``
that would resolve to a branch tip instead of the pin, or a symlinked directory inside
the clone must fail the gate, not silently validate files outside the pinned commit.
"""

import os
import subprocess
import sys
from pathlib import Path

import hermes_yaml as yaml
import pytest

ROOT = Path(__file__).resolve().parents[2]
# CATALOG_CI_WORKFLOW lets a caller point the harness at an older workflow copy
# (e.g. `git show HEAD:...`) to prove a case is a load-bearing regression test.
WORKFLOW = Path(os.environ.get(
    "CATALOG_CI_WORKFLOW", ROOT / ".github/workflows/plugin-catalog-ci.yml"))


def _run_block() -> str:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = data["jobs"]["pinned-source-validate"]["steps"]
    step = next(s for s in steps if s.get("name", "").startswith("Clone each entry"))
    return step["run"]


def _changed_files_block() -> str:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = data["jobs"]["pinned-source-validate"]["steps"]
    step = next(s for s in steps if s.get("name", "").startswith("Find changed catalog"))
    # The workflow expands ${{ github.base_ref }} before bash ever sees it.
    return step["run"].replace("${{ github.base_ref }}", "main")


def _origin_repo(tmp_path: Path) -> tuple[Path, str]:
    origin = tmp_path / "origin"
    (origin / "plugin").mkdir(parents=True)
    (origin / "plugin" / "plugin.yaml").write_text("name: fixture\n", encoding="utf-8")
    (origin / "plugin.yaml").write_text("name: root-fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=origin, check=True)
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=origin, check=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
    return origin, sha


def _write_entry(tmp_path: Path, filename: str, *, repo: str, sha: str, subdir="") -> Path:
    path = tmp_path / filename
    path.write_text(
        yaml.safe_dump({"repo": repo, "sha": sha, "subdir": subdir}), encoding="utf-8")
    return path


@pytest.mark.platforms("linux")
def test_non_hex_sha_fails_before_clone(tmp_path, fixture):
    """A ref name like HEAD resolves to whatever the branch tip is at CI time, not the
    declared pin; option-looking values would become git checkout flags."""
    origin, sha, tmpdir = fixture
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo="https://fixture.invalid/repo", sha="HEAD", subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "sha must be exactly 40 lowercase hex" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_whitespace_in_repo_fails_before_clone(tmp_path, fixture):
    """The repo value is echoed into the log; a newline would smuggle workflow
    commands like ::error:: through that echo."""
    origin, sha, tmpdir = fixture
    entry = _write_entry(
        tmp_path, "evil.yaml",
        repo="https://fixture.invalid/repo\n::error::fake", sha=sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "repo must be an https:// git URL" in res.stdout + res.stderr
    assert "::error::fake" not in res.stdout  # the smuggled command must never reach the log
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_non_string_subdir_fails_before_clone(tmp_path, fixture):
    """Lockstep with the structural gate: a falsy non-string must not coerce to '' and
    silently point the gate at the clone root."""
    origin, sha, tmpdir = fixture
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir=0)
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "subdir must be a string" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_unparseable_entry_fails_without_killing_the_loop(tmp_path, fixture):
    """A malformed yaml entry fails itself but must not crash the step and skip the
    remaining entries."""
    origin, sha, tmpdir = fixture
    bad = tmp_path / "broken.yaml"
    bad.write_text("repo: [unclosed\n", encoding="utf-8")
    ok = _write_entry(tmp_path, "ok.yaml",
                      repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    res = _run_gate(_env_multi(tmp_path, origin, [bad, ok], tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "entry yaml is empty or unparseable" in res.stdout + res.stderr
    assert f"PASS: {ok}" in res.stdout


def _env(tmp_path: Path, origin: Path, entry: Path, tmpdir: Path,
         hermes_stub: str = "#!/bin/sh\nexit 0\n") -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "hermes"
    stub.write_text(hermes_stub, encoding="utf-8")
    stub.chmod(0o755)
    return {
        **os.environ,
        "CHANGED_FILES": str(entry),
        "TMPDIR": str(tmpdir),
        "HERMES_STUB_LOG": str(tmp_path / "hermes-calls.log"),
        # the interpreter running the tests first: the step's `python3` heredoc
        # needs ruamel.yaml, which the workflow python provides via site-packages
        "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "GIT_TERMINAL_PROMPT": "0",
        # Serve the fixture repo under an https URL so `git clone` exercises the real path.
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.{origin.as_uri()}.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://fixture.invalid/repo",
    }


def _env_multi(tmp_path: Path, origin: Path, entries: list[Path], tmpdir: Path,
               hermes_stub: str = "#!/bin/sh\nexit 0\n") -> dict:
    return _env(tmp_path, origin, entries[0], tmpdir, hermes_stub=hermes_stub) | {
        "CHANGED_FILES": "\n".join(str(e) for e in entries)}


def _run_gate(env: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    # bash <file>, not bash -c: the run text mentions "update", which trips the
    # live-system guard's hermes-update heuristic when it sits inside the argv.
    script = tmp_path / "gate-step.sh"
    script.write_text(_run_block(), encoding="utf-8")
    return subprocess.run(
        ["bash", str(script)], env=env, cwd=tmp_path,
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)


@pytest.fixture
def fixture(tmp_path):
    origin, sha = _origin_repo(tmp_path)
    tmpdir = tmp_path / "mktmp"  # mktemp lands here so the escape path is predictable
    tmpdir.mkdir()
    return origin, sha, tmpdir


@pytest.mark.platforms("linux")
def test_traversal_subdir_fails_the_gate(tmp_path, fixture):
    origin, sha, tmpdir = fixture
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "plugin.yaml").write_text("name: planted\n", encoding="utf-8")
    # mktemp gives TMPDIR/tmp.XXX; one '..' reaches TMPDIR, then walk to planted/.
    subdir = "../" + os.path.relpath(planted, tmpdir)
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir=subdir)
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "must be a relative path" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_symlinked_subdir_inside_the_clone_fails_the_gate(tmp_path, fixture):
    origin, sha, tmpdir = fixture
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "plugin.yaml").write_text("name: planted\n", encoding="utf-8")
    (origin / "link").symlink_to(planted)  # committed symlink pointing outside the clone
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "link"],
        cwd=origin, check=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir="link")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "resolves outside the pinned clone" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_file_scheme_repo_is_rejected_before_any_clone(tmp_path, fixture):
    origin, sha, tmpdir = fixture
    # The clone must be remote-only: file:// would let an entry 'pin' content that is
    # really the PR checkout (or any path on the runner).
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo=origin.as_uri(), sha=sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "repo must be an https:// git URL" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_legit_entry_still_passes_the_gate(tmp_path, fixture):
    origin, sha, tmpdir = fixture
    entry = _write_entry(tmp_path, "ok.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"PASS: {entry}" in res.stdout


@pytest.mark.platforms("linux")
def test_empty_subdir_uses_the_clone_root(tmp_path, fixture):
    """An empty subdir means the plugin lives at the repo root; PLUGIN_DIR must equal
    the resolved CLONE_DIR, which the confinement check accepts only via its
    equality branch."""
    origin, sha, tmpdir = fixture
    entry = _write_entry(tmp_path, "ok.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir="")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"PASS: {entry}" in res.stdout


@pytest.mark.platforms("linux")
def test_non_mapping_entry_yaml_fails(tmp_path, fixture):
    origin, sha, tmpdir = fixture
    entry = tmp_path / "list.yaml"
    entry.write_text("- not\n- a mapping\n", encoding="utf-8")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "entry yaml must be a mapping" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_tag_object_sha_is_not_a_commit_pin(tmp_path, fixture):
    """An annotated tag object's sha is 40-hex, but checkout would peel it to the
    tagged commit; the pin must name a commit object itself."""
    origin, sha, tmpdir = fixture
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "tag", "-a", "v1", "-m", "x"],
        cwd=origin, check=True)
    tag_sha = subprocess.check_output(
        ["git", "rev-parse", "v1"], cwd=origin, text=True).strip()
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo="https://fixture.invalid/repo", sha=tag_sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "is not a commit object" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_symlinked_entry_file_is_rejected(tmp_path, fixture):
    """A typechanged entry (file -> symlink) must not be followed into whatever it
    points at; it is not a catalog entry."""
    origin, sha, tmpdir = fixture
    real = _write_entry(tmp_path, "real.yaml",
                        repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    link = tmp_path / "link.yaml"
    link.symlink_to(real)
    res = _run_gate(_env(tmp_path, origin, link, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "must be a regular file" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_symlink_inside_plugin_dir_escaping_clone_fails(tmp_path, fixture):
    """PLUGIN_DIR itself can resolve inside the clone while a committed symlink
    under it still points out; every file the gate reads must stay confined."""
    origin, sha, tmpdir = fixture
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "evil.js").write_text("releases/latest writeFile(", encoding="utf-8")
    (origin / "plugin" / "escaped").symlink_to(planted)
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "link"],
        cwd=origin, check=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "symlink inside the plugin dir resolves outside" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_plugin_yaml_as_symlink_to_outside_file_fails(tmp_path, fixture):
    """The manifest check `-f` follows symlinks; plugin.yaml as a link to an
    outside file must be caught by the per-symlink scan before it is trusted."""
    origin, sha, tmpdir = fixture
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "plugin.yaml").write_text("name: planted\n", encoding="utf-8")
    (origin / "plugin" / "plugin.yaml").unlink()
    (origin / "plugin" / "plugin.yaml").symlink_to(planted / "plugin.yaml")
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "link"],
        cwd=origin, check=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "symlink inside the plugin dir resolves outside" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_symlink_resolving_inside_the_clone_is_allowed(tmp_path, fixture):
    """The confinement scan is about escape, not symlinks per se: a link whose
    target stays inside the pinned clone must not over-reject."""
    origin, sha, tmpdir = fixture
    (origin / "plugin" / "alias.yaml").symlink_to("plugin.yaml")
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "alias"],
        cwd=origin, check=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
    entry = _write_entry(tmp_path, "ok.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"PASS: {entry}" in res.stdout


@pytest.mark.platforms("linux")
def test_self_updater_in_spaced_dirname_is_caught(tmp_path, fixture):
    """A filename containing spaces must not split through the grep|xargs stage and
    let a self-updating file slip past."""
    origin, sha, tmpdir = fixture
    d = origin / "plugin" / "evil dir"
    d.mkdir()
    (d / "inject.js").write_text("releases/latest writeFile(", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "evil"],
        cwd=origin, check=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "self-updating code" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_newline_filename_cannot_forge_workflow_commands(tmp_path, fixture):
    """The matched filenames are echoed inside an ::error line; a newline in a
    committed filename must not smuggle an extra workflow command into the log."""
    origin, sha, tmpdir = fixture
    (origin / "plugin" / "evil\n::error::forged.js").write_text(
        "releases/latest writeFile(", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=origin, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "evil"],
        cwd=origin, check=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=origin, text=True).strip()
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "self-updating code" in res.stdout + res.stderr
    assert not any(
        line.lstrip().startswith("::error::forged") for line in res.stdout.splitlines())


@pytest.mark.platforms("linux")
def test_parser_crash_fails_that_entry_not_the_loop(tmp_path, fixture):
    """If the parser process dies without output (OOM-kill, signal), the entry must
    fail via the PARSED fallback; it must never inherit the previous iteration's
    repo/sha and pass on that content."""
    origin, sha, tmpdir = fixture
    ok = _write_entry(tmp_path, "ok.yaml",
                      repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    crash = _write_entry(tmp_path, "crash.yaml",
                         repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    ok2 = _write_entry(tmp_path, "ok2.yaml",
                       repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    env = _env_multi(tmp_path, origin, [ok, crash, ok2], tmpdir)
    stub = tmp_path / "bin" / "python3"  # parser dies only on the one entry file
    stub.write_text(
        '#!/bin/sh\ncase "$*" in */crash.yaml) exit 1;; esac\n'
        # Unresolved path on purpose: .venv/bin/python resolves to the base
        # interpreter, which would drop the venv's site-packages under -I.
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8")
    stub.chmod(0o755)
    res = _run_gate(env, tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "catalog entry parser failed" in res.stdout + res.stderr
    assert f"PASS: {ok}" in res.stdout
    assert f"PASS: {ok2}" in res.stdout
    assert f"PASS: {crash}" not in res.stdout


@pytest.mark.platforms("linux")
def test_parser_imports_cannot_be_shadowed_by_checkout_files(tmp_path, fixture):
    """The parser runs with cwd = the PR checkout; without interpreter isolation a
    PR-authored shlex.py shadows the stdlib import and can swallow the verdict.

    shlex, not re: `re` is already imported during interpreter startup, so a cwd
    re.py never shadows it and the guard would pass even without `-I`. shlex is
    first imported by the parser itself, so it is the load-bearing vector."""
    origin, sha, tmpdir = fixture
    (tmp_path / "shlex.py").write_text(
        "def quote(s):\n"
        "    if s.startswith('repo must'):\n"
        "        s = ''  # swallow the rejection so BAD evals empty\n"
        "    return \"'\" + s.replace(\"'\", \"'\\\\''\") + \"'\"\n",
        encoding="utf-8")
    entry = _write_entry(tmp_path, "evil.yaml",
                         repo=origin.as_uri(), sha=sha, subdir="plugin")
    res = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "repo must be an https:// git URL" in res.stdout + res.stderr
    assert "PASS" not in res.stdout


@pytest.mark.platforms("linux")
def test_validate_process_cannot_drain_the_remaining_entries(tmp_path, fixture):
    """The entry list travels on fd 3: plugin validate code reading stdin cannot
    consume the remaining entry names and skip their validation."""
    origin, sha, tmpdir = fixture
    ok = _write_entry(tmp_path, "ok.yaml",
                      repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    ok2 = _write_entry(tmp_path, "ok2.yaml",
                       repo="https://fixture.invalid/repo", sha=sha, subdir="plugin")
    stub = "#!/bin/sh\ncat >/dev/null\nexit 0\n"  # drains whatever stdin it is handed
    res = _run_gate(
        _env_multi(tmp_path, origin, [ok, ok2], tmpdir, hermes_stub=stub), tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"PASS: {ok}" in res.stdout
    assert f"PASS: {ok2}" in res.stdout
    assert "all changed catalog entries" in res.stdout


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("mut", [
    {},
    {"subdir": "../x"},
    {"subdir": "/abs/path"},
    {"subdir": 5},
    {"repo": "file:///etc/passwd"},
    {"repo": "https://x/y\n::error::x"},
    {"sha": "HEAD"},
    {"sha": "a" * 40 + "\n"},
])
def test_workflow_and_structural_gate_agree(tmp_path, fixture, mut):
    """The field rules are duplicated between the parse step and the structural
    validator by necessity (different environments); the same entry must produce
    the same verdict at both gates or the parity has drifted."""
    origin, sha, tmpdir = fixture
    data = {"name": "parity-plugin", "repo": "https://fixture.invalid/repo",
            "sha": sha, "subdir": "plugin", "description": "d",
            "maintainer": "o", **mut}
    entry = tmp_path / "parity.yaml"
    entry.write_text(yaml.safe_dump(data), encoding="utf-8")
    structural = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_plugin_catalog.py"), str(entry)],
        capture_output=True, text=True)
    gate = _run_gate(_env(tmp_path, origin, entry, tmpdir), tmp_path)
    assert (structural.returncode == 0) == (gate.returncode == 0), (
        f"structural rc={structural.returncode} gate rc={gate.returncode}\n"
        f"{structural.stdout}{structural.stderr}\n{gate.stdout}{gate.stderr}")


@pytest.mark.platforms("linux")
def test_changed_files_step_reports_renamed_and_typechanged_entries(tmp_path):
    """AM alone misses renames (R) and file->symlink typechanges (T); either would
    drop the entry from CHANGED_FILES and skip the supply-chain gate entirely."""
    work = tmp_path / "work"
    (work / "plugin-catalog").mkdir(parents=True)
    (work / "plugin-catalog" / "a.yaml").write_text("name: a\n", encoding="utf-8")
    (work / "plugin-catalog" / "b.yaml").write_text("name: b\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
        cwd=work, check=True)
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=work, text=True).strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", base], cwd=work, check=True)
    subprocess.run(["git", "checkout", "-qb", "pr"], cwd=work, check=True)
    subprocess.run(
        ["git", "mv", "plugin-catalog/a.yaml", "plugin-catalog/renamed.yaml"],
        cwd=work, check=True)
    (work / "plugin-catalog" / "b.yaml").unlink()
    (work / "plugin-catalog" / "b.yaml").symlink_to("/etc/hostname")
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "pr"],
        cwd=work, check=True)
    out = tmp_path / "gh-output.txt"
    script = tmp_path / "changed-step.sh"
    script.write_text(_changed_files_block(), encoding="utf-8")
    res = subprocess.run(
        ["bash", str(script)], env={**os.environ, "GITHUB_OUTPUT": str(out)},
        cwd=work, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stdout + res.stderr
    files = out.read_text(encoding="utf-8")
    assert "plugin-catalog/renamed.yaml" in files
    assert "plugin-catalog/b.yaml" in files
