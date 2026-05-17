from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "docker-publish.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_release_publish_requires_main_reachability_check():
    workflow = _workflow_text()

    assert "authorize-publish:" in workflow
    assert "git merge-base --is-ancestor" in workflow
    assert '"${release_commit}" origin/main' in workflow
    assert "release_allowed=true" in workflow


def test_release_docker_publish_steps_are_gated_by_authorization():
    workflow = _workflow_text()
    release_publish_gate = (
        "github.event_name == 'release' && "
        "needs.authorize-publish.outputs.release_allowed == 'true'"
    )

    assert release_publish_gate in workflow
    assert "github.event_name == 'release')" not in workflow
