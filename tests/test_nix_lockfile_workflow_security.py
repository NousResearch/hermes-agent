from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "nix-lockfile-fix.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _fix_job_steps() -> list[dict]:
    return _load_workflow()["jobs"]["fix"]["steps"]


def _step_named(name: str) -> dict:
    for step in _fix_job_steps():
        if step.get("name") == name:
            return step
    raise AssertionError(f"missing workflow step: {name}")


def test_pr_lockfile_fix_rejects_fork_prs_before_checkout() -> None:
    script = _step_named("Authorize & resolve PR")["with"]["script"]

    assert "pr.head.repo.full_name" in script
    assert "context.repo.owner" in script
    assert "context.repo.repo" in script
    assert "core.setFailed" in script
    assert "fork PR" in script


def test_pr_lockfile_fix_uses_trusted_setup_action_for_pr_code() -> None:
    setup_steps = [
        step
        for step in _fix_job_steps()
        if step.get("uses", "").endswith(
            "/nix-installer-action@ef8a148080ab6020fd15196c2084a2eea5ff2d25"
        )
        or step.get("uses", "").endswith(
            "/cachix-action@1eb2ef646ac0255473d23a5907ad7b04ce94065c"
        )
    ]

    assert len(setup_steps) == 2
    assert all(not step["uses"].startswith("./") for step in setup_steps)


def test_pr_lockfile_fix_does_not_expose_tokens_to_checked_out_pr_code() -> None:
    checkout = next(
        step
        for step in _fix_job_steps()
        if step.get("uses", "").startswith("actions/checkout@")
    )

    assert checkout["with"]["persist-credentials"] == "false"

    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    pr_fix_text = workflow_text.split("# \u2500\u2500 PR fix", 1)[1]
    assert "secrets.CACHIX_AUTH_TOKEN" not in pr_fix_text
