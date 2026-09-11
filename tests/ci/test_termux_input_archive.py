"""CI runs archival before the exact tool and payload consumers."""
from pathlib import Path
import shlex

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
R2_ENV = {"CLOUDFLARE_R2_ACCOUNT_ID", "CLOUDFLARE_R2_ACCESS_KEY_ID", "CLOUDFLARE_R2_SECRET_ACCESS_KEY", "CLOUDFLARE_R2_BUCKET"}


def load(name):
    return YAML(typ="base").load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def test_termux_and_desktop_stagers_consume_archive_seeds():
    release = load("desktop-bundled-release.yml")
    for file, job_name in (("desktop-bundled-release.yml", "termux-deb"), ("termux-verify.yml", "native-runtime")):
        job = load(file)["jobs"][job_name]
        steps = job["steps"]
        index, step = next((i, s) for i, s in enumerate(steps) if "scripts.ci.archive_inputs" in s.get("run", ""))
        args = shlex.split(step["run"])
        assert args[args.index("--target") + 1] == "linux-arm64-bionic"
        assert args[args.index("--store") + 1] == "$HERMES_RUNTIME_DIR"
        assert args[args.index("--payload") + 1] == "termux-build/payload"
        assert "if" not in step and "continue-on-error" not in step
        assert R2_ENV <= (job.get("env", {}).keys() | step.get("env", {}).keys())
        first_stage = next(i for i, s in enumerate(steps) if "scripts/termux/build_cpython.sh" in s.get("run", ""))
        assert index < first_stage
    for name in ("build-win32", "build-darwin"):
        steps = release["jobs"][name]["steps"]
        index, step = next((i, s) for i, s in enumerate(steps) if "scripts.ci.archive_inputs" in s.get("run", ""))
        assert "--target '${{ matrix.target.label }}'" in step["run"]
        assert "--store apps/desktop/build/agent-payload/tools" in step["run"]
        assert index < next(i for i, s in enumerate(steps) if "scripts/bundles/desktop.py" in s.get("run", ""))
        setup = next(s for s in steps if s.get("uses") == "./.github/actions/setup-pm")
        assert setup["with"]["archive-inputs"] == "true"


def test_archive_gate_uses_bootstrap_python_and_trusted_exact_revision():
    workflow = load("archive-inputs.yml")
    assert not {"pull_request", "pull_request_target"} & workflow["on"].keys()
    assert "workflow_dispatch" in workflow["on"] and "push" in workflow["on"]
    job = workflow["jobs"]["archive-inputs"]
    assert job["environment"] == "release-signing"
    assert R2_ENV <= job["env"].keys()
    assert not any(s.get("uses") == "./.github/actions/setup-pm" for s in job["steps"])
    assert job["steps"][-1]["run"] == "python3 -m scripts.ci.archive_inputs"
    checkout = job["steps"][0]
    assert checkout["with"]["ref"] == "${{ inputs.sha || github.sha }}"
    release = load("desktop-bundled-release.yml")["jobs"]
    caller = release["archive-inputs"]
    assert caller["needs"] == ["validate"]
    assert caller["with"]["sha"] == "${{ needs.validate.outputs.sha }}"
    for name in ("build-win32", "build-darwin", "termux-deb"):
        assert "archive-inputs" in release[name]["needs"]

    action = YAML(typ="base").load((ROOT / ".github/actions/setup-pm/action.yml").read_text(encoding="utf-8"))
    assert action["inputs"]["archive-inputs"]["default"] == "false"
    steps = action["runs"]["steps"]
    archive_index, archive = next((i, s) for i, s in enumerate(steps) if "setup_toolchain.py\" archive-inputs" in s.get("run", ""))
    assert archive["if"] == "inputs.archive-inputs == 'true'"
    assert archive_index < next(i for i, s in enumerate(steps) if s.get("id") == "install")
