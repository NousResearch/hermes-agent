"""CI runs archival before the exact tool and payload consumers."""
import json
from pathlib import Path

import pytest

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
R2_ENV = {"CLOUDFLARE_R2_ACCOUNT_ID", "CLOUDFLARE_R2_ACCESS_KEY_ID", "CLOUDFLARE_R2_SECRET_ACCESS_KEY", "CLOUDFLARE_R2_BUCKET"}


def load(name):
    return YAML(typ="base").load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def test_archive_reader_accepts_bom_without_changing_pin_authority(tmp_path):
    from scripts.ci.archive_inputs import pinned_inputs

    (tmp_path / "pm").mkdir()
    (tmp_path / "pm/lock.json").write_bytes(b'\xef\xbb\xbf{"schema":1,"packages":{}}')
    table = tmp_path / "pm/termux_runtime_libs.json"
    row = {"url": "https://example.invalid/café.deb", "sha256": "a" * 64}
    table.write_bytes(b"\xef\xbb\xbf" + json.dumps({"libs": {"lib": row}}, ensure_ascii=False).encode("utf-8"))
    before = table.read_bytes()
    (pin,) = pinned_inputs(tmp_path, target="linux-arm64-bionic")
    assert (pin.name, pin.url, pin.sha256) == ("lib", row["url"], row["sha256"])
    assert table.read_bytes() == before
    row["sha256"] = " " + row["sha256"]
    table.write_bytes(b"\xef\xbb\xbf" + json.dumps({"libs": {"lib": row}}).encode("utf-8"))
    with pytest.raises(ValueError):
        pinned_inputs(tmp_path, target="linux-arm64-bionic")


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
    # The archive lives as a validate step with the job's own gate shape —
    # a skipped archive (disposable/termux runs) must not skip the builds.
    validate = release["validate"]
    (archive,) = [s for s in validate["steps"] if s.get("run") == "python3 -m scripts.ci.archive_inputs"]
    assert archive["if"] == (
        "inputs.disposable_run == '' && inputs.disposable_channel == ''"
        " && (inputs.release-phase == '' || inputs.release-phase == 'candidate')"
    )
    assert R2_ENV <= archive["env"].keys()
    assert validate["environment"] == "release-signing"
    checkout = validate["steps"][0]
    assert "actions/checkout" in checkout["uses"]
    for name in ("build-win32", "build-darwin", "termux-deb"):
        assert "archive-inputs" not in release[name].get("needs", [])

    action = YAML(typ="base").load((ROOT / ".github/actions/setup-pm/action.yml").read_text(encoding="utf-8"))
    assert action["inputs"]["archive-inputs"]["default"] == "false"
    steps = action["runs"]["steps"]
    archive_index, archive = next((i, s) for i, s in enumerate(steps) if "setup_toolchain.py\" archive-inputs" in s.get("run", ""))
    assert archive["if"] == "inputs.archive-inputs == 'true'"
    assert archive_index < next(i for i, s in enumerate(steps) if s.get("id") == "install")
