import os
from pathlib import Path
import subprocess

import pytest
import yaml


_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github/workflows/install-e2e-run.yml"
_SANDBOX = _REPO / "scripts/dev-sandbox.sh"
_GIT_REPOSITORY_ENV = {
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
}


def _steps() -> list[dict]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["e2e"]["steps"]


def _one_step(steps: list[dict], predicate, description: str) -> dict:
    matches = [step for step in steps if predicate(step)]
    assert len(matches) == 1, f"expected one {description}, found {len(matches)}"
    return matches[0]


def _isolated_git_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in _GIT_REPOSITORY_ENV:
        env.pop(name, None)
    env.update({
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    })
    return env


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgSign=false",
            "-c",
            "core.hooksPath=",
            "-C",
            str(repo),
            *args,
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_isolated_git_env(),
    ).stdout.strip()


def _init_repo(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Install E2E test")
    _git(path, "config", "user.email", "install-e2e-test@invalid")


def test_fixture_git_ignores_ambient_repository_selection(tmp_path: Path, monkeypatch):
    decoy = tmp_path / "decoy"
    _init_repo(decoy)
    _git(decoy, "config", "user.name", "Decoy")

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))

    _git(fixture, "init", "-q", "-b", "main")

    assert (fixture / ".git").is_dir()
    assert _git(decoy, "config", "user.name") == "Decoy"


def test_fixture_git_allows_local_remotes_when_ambient_protocol_blocks_file(
    tmp_path: Path,
    monkeypatch,
):
    origin = tmp_path / "origin"
    _init_repo(origin)
    _commit_installer(origin, "origin", "origin")

    checkout = tmp_path / "checkout"
    _init_repo(checkout)
    _git(checkout, "remote", "add", "origin", str(origin))
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "https")

    _git(checkout, "fetch", "-q", "origin", "main")

    assert _git(checkout, "rev-parse", "FETCH_HEAD") == _git(
        origin, "rev-parse", "HEAD"
    )


@pytest.mark.parametrize(
    ("scope", "environment_name"),
    [("--global", "GIT_CONFIG_GLOBAL"), ("--system", "GIT_CONFIG_SYSTEM")],
)
def test_fixture_git_ignores_ambient_config_sources(
    tmp_path: Path,
    monkeypatch,
    scope: str,
    environment_name: str,
):
    repo = tmp_path / "repo"
    _init_repo(repo)
    ambient_config = tmp_path / "ambient.gitconfig"
    ambient_config.write_text("[fixture]\n\tmarker = ambient\n", encoding="utf-8")
    monkeypatch.setenv(environment_name, str(ambient_config))

    with pytest.raises(subprocess.CalledProcessError):
        _git(repo, "config", scope, "fixture.marker")


def _commit_installer(repo: Path, body: str, message: str) -> str:
    installer = repo / "scripts/install.sh"
    installer.parent.mkdir(exist_ok=True)
    installer.write_text(f"#!/bin/sh\n# {body}\nexit 0\n", encoding="utf-8")
    _git(repo, "add", "scripts/install.sh")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD^{commit}")


def _source_state(repo: Path) -> tuple[str, str, str, str, bool]:
    return (
        _git(repo, "rev-parse", "HEAD"),
        _git(repo, "for-each-ref", "--format=%(refname) %(objectname)"),
        _git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        _git(repo, "diff", "--cached"),
        (repo / ".git/FETCH_HEAD").exists(),
    )


def test_install_e2e_uses_upstream_checkout_without_leaking_credential_or_source_state():
    """Matrix legs consume an upstream checkout outside the source and token scope."""
    steps = _steps()
    source_checkout = _one_step(
        steps,
        lambda step: (
            str(step.get("uses", "")).startswith("actions/checkout@")
            and "repository" not in step.get("with", {})
        ),
        "source checkout step",
    )
    upstream_checkout = _one_step(
        steps,
        lambda step: (
            str(step.get("uses", "")).startswith("actions/checkout@")
            and step.get("with", {}).get("repository") == "NousResearch/hermes-agent"
        ),
        "upstream checkout step",
    )
    move_step = _one_step(
        steps,
        lambda step: step.get("name") == "Move upstream checkout outside source",
        "upstream move step",
    )
    run_step = _one_step(
        steps,
        lambda step: step.get("name") == "Run install + update E2E",
        "install E2E step",
    )

    assert source_checkout["with"]["fetch-depth"] == 0
    assert steps.index(source_checkout) < steps.index(upstream_checkout)
    assert steps.index(upstream_checkout) < steps.index(move_step)
    assert steps.index(move_step) < steps.index(run_step)
    upstream_with = upstream_checkout["with"]
    assert upstream_with["repository"] == "NousResearch/hermes-agent"
    assert upstream_with["ref"] == "${{ inputs.install-ref }}"
    assert upstream_with["fetch-depth"] == 0
    assert upstream_with["path"] == ".install-upstream"
    assert upstream_with["persist-credentials"] is False
    assert move_step["env"]["UPSTREAM_CHECKOUT"] == (
        "${{ runner.temp }}/install-upstream"
    )
    assert move_step["run"] == 'mv .install-upstream "$UPSTREAM_CHECKOUT"'
    assert run_step["env"]["HERMES_DEV_SANDBOX_UPSTREAM"] == (
        "${{ runner.temp }}/install-upstream"
    )
    assert "--install-ref '${{ inputs.install-ref }}'" in run_step["run"]


@pytest.mark.parametrize("ref_kind", ["tag", "branch", "sha"])
def test_dev_sandbox_resolves_distinct_upstream_ref_without_mutating_source(
    tmp_path: Path,
    monkeypatch,
    ref_kind: str,
):
    """The real resolver uses its upstream checkout, not same-named source refs."""
    source = tmp_path / "source"
    _init_repo(source)
    _commit_installer(source, "source tag decoy", "source tag decoy")
    _git(source, "tag", "-a", "v1", "-m", "source release")
    _commit_installer(source, "source branch decoy", "source branch decoy")

    upstream_origin = tmp_path / "upstream-origin"
    _init_repo(upstream_origin)
    upstream_tag = _commit_installer(upstream_origin, "upstream tag", "upstream tag")
    _git(upstream_origin, "tag", "-a", "v1", "-m", "upstream release")
    upstream_branch = _commit_installer(
        upstream_origin,
        "upstream branch",
        "upstream branch",
    )

    upstream_checkout = tmp_path / "upstream-checkout"
    upstream_checkout.mkdir()
    _git(upstream_checkout, "init", "-q")
    _git(upstream_checkout, "remote", "add", "origin", str(upstream_origin))
    _git(
        upstream_checkout,
        "fetch",
        "-q",
        "origin",
        "+refs/heads/*:refs/remotes/origin/*",
        "+refs/tags/*:refs/tags/*",
    )
    if ref_kind == "tag":
        install_ref = "v1"
        _git(upstream_checkout, "checkout", "-q", "--detach", "refs/tags/v1")
        expected_release = upstream_tag
    elif ref_kind == "branch":
        install_ref = "refs/heads/main"
        _git(
            upstream_checkout,
            "checkout",
            "-q",
            "-B",
            "main",
            "refs/remotes/origin/main",
        )
        expected_release = upstream_branch
    else:
        install_ref = upstream_tag
        expected_release = upstream_tag
        _git(upstream_checkout, "checkout", "-q", "--detach", install_ref)

    # A resolver must consume the checkout's local objects and refs. Its
    # persisted remote has no credential in CI and must not become a hidden
    # network dependency or a chance to re-resolve a moved ref.
    _git(
        upstream_checkout,
        "remote",
        "set-url",
        "origin",
        str(tmp_path / "unreachable-upstream"),
    )

    if ref_kind == "sha":
        with monkeypatch.context() as ambient_git:
            ambient_git.setenv("GIT_DIR", str(upstream_checkout / ".git"))
            source_has_ref = subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "cat-file",
                    "-e",
                    f"{install_ref}^{{commit}}",
                ],
                capture_output=True,
                check=False,
                env=_isolated_git_env(),
            )
        assert source_has_ref.returncode != 0
    else:
        assert (
            _git(source, "rev-parse", f"{install_ref}^{{commit}}") != expected_release
        )
    source_before = _source_state(source)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("bwrap", "slirp4netns"):
        fake_command = fake_bin / name
        fake_command.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        fake_command.chmod(0o755)
    fake_unshare = fake_bin / "unshare"
    fake_unshare.write_text("#!/bin/sh\nexit 73\n", encoding="utf-8")
    fake_unshare.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()

    completed = subprocess.run(
        [
            str(_SANDBOX),
            "install",
            "--root",
            "--install-ref",
            install_ref,
            "--",
            "--skip-setup",
        ],
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(home),
            "HERMES_SANDBOX_SOURCE_ROOT": str(source),
            "HERMES_DEV_SANDBOX_UPSTREAM": str(upstream_checkout),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 73, completed.stderr
    assert (
        f"[sandbox] fake main: upstream {install_ref} ({expected_release})"
        in completed.stderr
    )
    assert "could not resolve upstream ref" not in completed.stderr
    assert _source_state(source) == source_before
