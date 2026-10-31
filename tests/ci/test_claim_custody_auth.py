"""Every workflow claim-custody call can authenticate its maintainer lookup."""
from pathlib import Path
import re

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
CUSTODY_COMMAND = re.compile(
    r"\bpython(?:3)? -m scripts\.releases\.stable "
    r"(admit|verify|transitions|stage-receipt|candidate-manifest|complete)\b"
)


def custody_steps():
    yaml = YAML(typ="safe")
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        workflow = yaml.load(path)
        for name, job in workflow.get("jobs", {}).items():
            for step in job.get("steps", []):
                if CUSTODY_COMMAND.search(step.get("run", "")):
                    yield pytest.param(step, id=f"{path.name}/{name}/{step.get('name', step['run'])}")


@pytest.mark.parametrize("step", list(custody_steps()))
def test_claim_custody_explicitly_authenticates_github_cli(step):
    # Checkout's Git credential helper and tokens in other steps do not authenticate gh.
    # Scope the credential to the step rather than all candidate-code execution in the job.
    assert step.get("env", {}).get("GH_TOKEN") == "${{ github.token }}"
