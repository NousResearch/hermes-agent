"""The release workflow's dependency graph enforces publication ordering."""
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return YAML(typ="base").load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def ancestors(jobs, name):
    seen = set()
    pending = [name]
    while pending:
        needs = jobs[pending.pop()].get("needs", [])
        for item in [needs] if isinstance(needs, str) else needs:
            if item not in seen:
                seen.add(item)
                pending.append(item)
    return seen


def test_release_reuses_whole_ci_and_docker_before_publication():
    jobs = workflow("stable-release.yml")["jobs"]
    assert jobs["ci"]["uses"] == "./.github/workflows/ci.yaml"
    assert jobs["ci"]["with"]["release"] == "true"
    assert "secrets" not in jobs["ci"]
    assert jobs["docker"]["uses"] == jobs["publish-docker"]["uses"] == jobs["promote-docker"]["uses"]
    assert jobs["docker"]["with"]["release-phase"] == "test"
    assert "ci" in ancestors(jobs, "docker")
    required = {"ci", "docker", "nix", "pm-bundle", "install-e2e", "windows-packaged", "macos-packaged", "termux-checks", "windows-live", "candidates"}
    assert required <= ancestors(jobs, "acceptance")
    for name in ("publish-docker", "publish-bundles"):
        assert required <= ancestors(jobs, name)
    for name in ("promote-docker", "promote-bundles"):
        assert {"publish-docker", "publish-bundles", "publication"} <= ancestors(jobs, name)
    assert {"promote-docker", "promote-bundles"} <= ancestors(jobs, "complete")
    for name in ("acceptance", "publication", "complete"):
        assert jobs[name]["if"] == "always()"


def test_all_applicable_ci_jobs_are_aggregated_and_desktop_e2e_stays_deferred():
    jobs = workflow("ci.yaml")["jobs"]
    checks = {name for name, job in jobs.items() if "uses" in job}
    assert checks <= set(jobs["all-checks-pass"]["needs"])
    assert jobs["e2e-desktop"]["if"] == "false"
    assert "workflow_call" in workflow("ci.yaml")["on"]
