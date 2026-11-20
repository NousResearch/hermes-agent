"""The native catalog gate pins a direct build or an exactly approved repack."""

import hashlib
import json
import runpy
import sys
import zipfile
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))

import catalog_validate_pinned_entries as gate

REPO = "Lokee86/example"
PIN = "a" * 40
PRODUCER = "b" * 40
WORKFLOW = ".github/workflows/native.yml"
APPROVED_RUN = f"https://github.com/{REPO}/actions/runs/100/attempts/1"
INPUT_RUN = f"https://github.com/{REPO}/actions/runs/99/attempts/1"


@pytest.mark.parametrize(
    ("approval_matches", "signer", "invocation", "admitted"),
    [
        (False, PIN, INPUT_RUN, True),
        (True, PRODUCER, APPROVED_RUN, True),
        (True, PRODUCER, f"{APPROVED_RUN[:-1]}2", False),
        (False, PRODUCER, APPROVED_RUN, False),
    ],
    ids=["pin", "approved-repack", "other-attempt", "unapproved-archive"],
)
def test_native_provenance_requires_reviewed_source_or_exact_repack(
    tmp_path, monkeypatch, approval_matches, signer, invocation, admitted
):
    tag = "v1.0.0"
    member = "example/bin/host"
    destination = "bin/host"
    target = "macos-arm64"
    filename = "example-macos-arm64.zip"
    published = tmp_path / filename
    with zipfile.ZipFile(published, "w") as archive:
        entry = zipfile.ZipInfo(member)
        entry.create_system = 3
        entry.external_attr = 0o100755 << 16
        archive.writestr(entry, bytes.fromhex("cffaedfe0c000001000000000200000015000000") + b"test host")
    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    declaration = {
        "tag": tag,
        "member": member,
        "destination": destination,
        "targets": {target: {
            "filename": filename,
            "sha256": digest,
            "min_macos": "15.0",
        }},
    }
    approval = {
        "artifact": {
            "repo": REPO, "pin": PIN, "tag": tag, "target": target,
            "member": member, "destination": destination, "filename": filename,
            "sha256": digest if approval_matches else "c" * 64,
        },
        "producer": {"sha": PRODUCER, "workflow": WORKFLOW, "invocation": APPROVED_RUN},
        "input": {"sha256": "d" * 64, "invocation": INPUT_RUN},
    }
    ledger = tmp_path / "approvals.json"
    ledger.write_text(json.dumps([approval]), encoding="utf-8")
    monkeypatch.setattr(gate, "_NATIVE_REPACK_ALLOWLIST", ledger)

    def download(_repo, _tag, _asset, output):
        output.write_bytes(published.read_bytes())

    def signed_result(argv, **_kwargs):
        if argv[0] == "git":
            return CompletedProcess(argv, 0, f"{PIN}\n".encode())
        assert argv[:3] == ["gh", "attestation", "verify"]
        assert argv[argv.index("--repo") + 1] == REPO
        expected = argv[argv.index("--source-digest") + 1]
        if expected != signer:
            return CompletedProcess(argv, 1, b"")
        if signer == PRODUCER:
            assert "--signer-repo" not in argv
            assert argv[argv.index("--signer-digest") + 1] == PRODUCER
            assert argv[argv.index("--signer-workflow") + 1] == f"{REPO}/{WORKFLOW}"
        else:
            assert argv[argv.index("--signer-repo") + 1] == REPO
        return CompletedProcess(argv, 0, f"{invocation}\n".encode())

    contract = runpy.run_path(str(gate._NATIVE_ASSETS_MODULE))
    monkeypatch.setattr(gate, "runpy", SimpleNamespace(
        run_path=lambda _path: {**contract, "download_native_archive": download}
    ))
    monkeypatch.setattr(gate, "_run", signed_result)
    fields = json.dumps([declaration, ["macos"]])
    if admitted:
        gate._check_native_assets(fields, str(tmp_path), f"https://github.com/{REPO}", PIN)
    else:
        with pytest.raises(gate.EntryRejected, match="provenance failed"):
            gate._check_native_assets(fields, str(tmp_path), f"https://github.com/{REPO}", PIN)
