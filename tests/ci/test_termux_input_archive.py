"""CI runs archival before the exact tool and payload consumers."""
import json
from pathlib import Path

import pytest

from ruamel.yaml import YAML

from tests.ci.desktop_release_roles import DOWNLOADABLE_DISPATCHES, gate, needs_of

ROOT = Path(__file__).resolve().parents[2]
R2_ENV = {"CLOUDFLARE_R2_ACCOUNT_ID", "CLOUDFLARE_R2_ACCESS_KEY_ID", "CLOUDFLARE_R2_SECRET_ACCESS_KEY", "CLOUDFLARE_R2_BUCKET"}


def needs_r2_credentials(step):
    """The step runs with R2 credentials configured and skips without them."""
    configured = gate(step["if"], {}, {}, job_if=False, env={"CLOUDFLARE_R2_ACCOUNT_ID": "account"})
    return configured and not gate(step["if"], {}, {}, job_if=False, env={"CLOUDFLARE_R2_ACCOUNT_ID": ""})


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
    (archive,) = [s for s in job["steps"] if s.get("run") == "python3 -m scripts.ci.archive_inputs"]
    # Pushes to main without R2 credentials skip the archive rather than fail.
    assert needs_r2_credentials(archive)
    checkout = job["steps"][0]
    assert checkout["with"]["ref"] == "${{ inputs.sha || github.sha }}"
    release = load("desktop-bundled-release.yml")["jobs"]
    # The archive lives as a validate step with the job's own gate shape —
    # a skipped archive (disposable/termux runs) must not skip the builds.
    validate = release["validate"]
    (archive,) = [s for s in validate["steps"] if s.get("run") == "python3 -m scripts.ci.archive_inputs"]
    for dispatch, inputs in DOWNLOADABLE_DISPATCHES.items():
        assert gate(archive["if"], inputs, {}, job_if=False), dispatch
    for skipped in ({"disposable_run": "98765"}, {"disposable_channel": "native-preview"},
                    {"release-phase": "publish"}, {"release-phase": "promote"}):
        assert not gate(archive["if"], {**DOWNLOADABLE_DISPATCHES["tag"], **skipped}, {}, job_if=False), skipped
    assert R2_ENV <= archive["env"].keys()
    assert validate["environment"] == "release-signing"
    checkout = validate["steps"][0]
    assert "actions/checkout" in checkout["uses"]
    assert not [name for name, job in release.items() if "archive-inputs" in needs_of(job)]

    action = YAML(typ="base").load((ROOT / ".github/actions/setup-pm/action.yml").read_text(encoding="utf-8"))
    assert action["inputs"]["archive-inputs"]["default"] == "false"
    steps = action["runs"]["steps"]
    archive_index, archive = next((i, s) for i, s in enumerate(steps) if "setup_toolchain.py\" archive-inputs" in s.get("run", ""))
    assert gate(archive["if"], {"archive-inputs": "true"}, {}, job_if=False)
    assert not gate(archive["if"], {"archive-inputs": "false"}, {}, job_if=False)
    assert archive_index < next(i for i, s in enumerate(steps) if s.get("id") == "install")


def test_scheduled_and_main_r2_consumers_skip_without_credentials():
    """Only explicit release runs may fail on missing R2; main and the nightly stay green."""
    prune = load("canary-release.yml")["jobs"]["prune"]
    assert R2_ENV <= prune["env"].keys()
    r2_steps = [s for s in prune["steps"] if "scripts.releases.r2" in s.get("run", "")]
    assert r2_steps and all(needs_r2_credentials(s) for s in r2_steps)

    termux = load("termux-verify.yml")
    assert termux["on"]["push"]["branches"] == ["main"] and "pull_request" in termux["on"]
    native = termux["jobs"]["native-runtime"]
    probe = termux["jobs"][native["needs"]]
    assert probe["environment"] == "release-signing"
    for configured, release, runs in (("true", False, True), ("false", True, True), ("false", False, False),
                                      ("", False, False)):
        needs = {native["needs"]: {"result": "success", "outputs": {"configured": configured}}}
        assert gate(native["if"], {"release": release}, needs) is runs, (configured, release)

