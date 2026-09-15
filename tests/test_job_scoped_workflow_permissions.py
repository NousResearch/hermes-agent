from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> dict:
    with (ROOT / ".github" / "workflows" / name).open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def test_ci_grants_write_permissions_only_to_the_jobs_that_need_them() -> None:
    workflow = _workflow("ci.yml")

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["supply-chain"]["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
    }
    assert workflow["jobs"]["review-labels"]["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
    }
    assert workflow["jobs"]["osv-scanner"]["permissions"] == {
        "actions": "read",
        "contents": "read",
        "security-events": "write",
    }


def test_deploy_grants_publish_permissions_only_to_publish_job() -> None:
    workflow = _workflow("deploy-compose.yml")

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["publish"]["permissions"] == {
        "attestations": "write",
        "checks": "read",
        "contents": "read",
        "id-token": "write",
        "packages": "write",
    }
    assert workflow["jobs"]["deploy-staging"]["permissions"] == {
        "contents": "read"
    }
