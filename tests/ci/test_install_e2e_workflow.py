import os
from pathlib import Path
import subprocess

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


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_install_e2e_reuses_full_checkout_for_release_refs():
    """Matrix legs must not fan out unauthenticated GitHub fetches for tags."""
    steps = _steps()
    checkout = _one_step(
        steps,
        lambda step: str(step.get("uses", "")).startswith("actions/checkout@"),
        "checkout step",
    )
    run_step = _one_step(
        steps,
        lambda step: step.get("name") == "Run install + update E2E",
        "install E2E step",
    )

    checkout_with = checkout["with"]
    assert checkout_with["fetch-depth"] == 0
    assert "path" not in checkout_with
    assert "filter" not in checkout_with
    assert not checkout_with.get("sparse-checkout")
    assert "repository" not in checkout_with
    assert run_step["env"]["HERMES_DEV_SANDBOX_UPSTREAM"] == "${{ github.workspace }}"
    assert (
        'UPSTREAM_URL="${HERMES_DEV_SANDBOX_UPSTREAM:-https://github.com/'
        'NousResearch/hermes-agent.git}"' in _SANDBOX.read_text(encoding="utf-8")
    )


def test_dev_sandbox_resolves_release_from_local_checkout(tmp_path: Path, monkeypatch):
    """The workflow override reaches the real release-ref fetch without a network."""
    # Git runs tests from hooks and bisect with repository-selection variables
    # exported. Poison them here so fixture setup proves it cannot mutate or use
    # the invoking repository.
    for key in _GIT_REPOSITORY_ENV:
        monkeypatch.setenv(key, str(tmp_path / f"poison-{key.lower()}"))

    git_env = {
        key: value
        for key, value in os.environ.items()
        if key not in _GIT_REPOSITORY_ENV
    }
    git_env.update({
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    })

    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q", "-b", "main", env=git_env)
    _git(source, "config", "user.name", "Install E2E test", env=git_env)
    _git(source, "config", "user.email", "install-e2e-test@invalid", env=git_env)
    installer = source / "scripts/install.sh"
    installer.parent.mkdir()
    installer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _git(source, "add", "scripts/install.sh", env=git_env)
    _git(source, "commit", "-q", "-m", "release", env=git_env)
    _git(source, "tag", "v1", env=git_env)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_unshare = fake_bin / "unshare"
    fake_unshare.write_text("#!/bin/sh\nexit 73\n", encoding="utf-8")
    fake_unshare.chmod(0o755)

    env = git_env.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "HERMES_SANDBOX_SOURCE_ROOT": str(source),
        "HERMES_DEV_SANDBOX_UPSTREAM": str(source),
    })
    completed = subprocess.run(
        [
            str(_SANDBOX),
            "install",
            "--root",
            "--install-ref",
            "v1",
            "--",
            "--skip-setup",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 73, completed.stderr
    release = _git(source, "rev-parse", "v1^{commit}", env=git_env)
    assert f"[sandbox] fake main: upstream v1 ({release})" in completed.stderr
    assert "could not resolve upstream ref" not in completed.stderr
