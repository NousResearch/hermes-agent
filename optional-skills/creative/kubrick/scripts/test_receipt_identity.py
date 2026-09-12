#!/usr/bin/env python3
"""Contract tests for deterministic compile receipt identity metadata."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from runtime_identity import (  # noqa: E402
    compile_identity,
    corpus_files,
    corpus_identity,
    provider_adapter_identity,
    schema_bundle_identity,
)

PY = sys.executable
REQUIRED_FIELDS = {
    "kubrick_version",
    "corpus_version",
    "corpus_digest",
    "schema_bundle_version",
    "provider_adapter_version",
    "command",
    "command_digest",
    "normalized_options",
    "normalized_input_digest",
}


def copy_identity_surface(destination: Path) -> None:
    manifest = json.loads((ROOT / "kubrick.manifest.yaml").read_text(encoding="utf-8"))
    paths = [ROOT / "kubrick.manifest.yaml", *corpus_files(ROOT)]
    paths.extend(ROOT / relative for relative in manifest["schemas"])
    paths.extend(
        ROOT / "scripts" / name
        for name in (
            "build_model_adapter_packet.py",
            "adapt_provider.py",
            "adapt_grok_imagine.py",
            "adapt_flux.py",
            "adapt_sd3.py",
            "adapt_midjourney.py",
        )
    )
    for source in paths:
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_identity_is_stable_and_semantic() -> None:
    payload = {"brief": {"dramatic_problem": "authority transfer"}, "storyboard_plan": None}
    first = compile_identity(payload, mode="scene", provider="flux")
    second = compile_identity(payload, mode="scene", provider="flux")
    assert first == second
    assert REQUIRED_FIELDS <= set(first)
    assert first["command"] == "compile"
    assert all("/home/" not in str(value) for value in first.values())
    assert compile_identity(payload, mode="scene", provider="sd3")["command_digest"] != first["command_digest"]
    changed = {"brief": {"dramatic_problem": "authority residue"}, "storyboard_plan": None}
    assert compile_identity(changed, mode="scene", provider="flux")["normalized_input_digest"] != first["normalized_input_digest"]


def test_bundle_mutations_change_only_relevant_identities() -> None:
    with tempfile.TemporaryDirectory() as directory:
        clone = Path(directory)
        copy_identity_surface(clone)
        corpus_before = corpus_identity(clone)
        schema_before = schema_bundle_identity(clone)
        adapter_before = provider_adapter_identity("flux", clone)

        corpus_path = clone / "references/corpus-index.yaml"
        corpus_path.write_text(corpus_path.read_text(encoding="utf-8") + "\n# identity mutation\n", encoding="utf-8")
        assert corpus_identity(clone)[1] != corpus_before[1]
        assert schema_bundle_identity(clone) == schema_before
        assert provider_adapter_identity("flux", clone) == adapter_before

        schema_path = clone / "schemas/model-adapter-packet.schema.yaml"
        schema_path.write_text(schema_path.read_text(encoding="utf-8") + "\n# identity mutation\n", encoding="utf-8")
        assert schema_bundle_identity(clone) != schema_before

        adapter_path = clone / "scripts/adapt_flux.py"
        adapter_path.write_text(adapter_path.read_text(encoding="utf-8") + "\n# identity mutation\n", encoding="utf-8")
        assert provider_adapter_identity("flux", clone) != adapter_before


def test_canonical_compile_receipt_contains_identity() -> None:
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "compile"
        command = [
            PY,
            str(SCRIPTS / "kubrick.py"),
            "compile",
            "--brief",
            str(ROOT / "examples/authority-transfer-storyboard/brief.yaml"),
            "--ledger",
            str(ROOT / "examples/authority-transfer-storyboard/symbolic-ledger.yaml"),
            "--mode",
            "storyboard",
            "--storyboard-plan",
            str(ROOT / "examples/authority-transfer-storyboard/storyboard-plan.yaml"),
            "--provider",
            "flux",
            "--out",
            str(out),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        receipt = json.loads((out / "compile-receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "COMPILED"
        assert REQUIRED_FIELDS <= set(receipt)
        assert receipt["corpus_version"] == "0.11.5"
        assert receipt["provider_adapter_version"].startswith("sha256:")


def test_not_computable_receipt_contains_identity() -> None:
    import yaml

    source = yaml.safe_load(
        (ROOT / "examples/authority-transfer-storyboard/brief.yaml").read_text(encoding="utf-8")
    )
    source["observable_evidence"] = ["only one observed form"]
    source.pop("observed_forms", None)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        brief = root / "insufficient.yaml"
        out = root / "compile"
        brief.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
        result = subprocess.run(
            [
                PY,
                str(SCRIPTS / "kubrick.py"),
                "compile",
                "--brief",
                str(brief),
                "--out",
                str(out),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        receipt = json.loads((out / "compile-receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "NOT_COMPUTABLE"
        assert REQUIRED_FIELDS <= set(receipt)


def main() -> None:
    test_identity_is_stable_and_semantic()
    test_bundle_mutations_change_only_relevant_identities()
    test_canonical_compile_receipt_contains_identity()
    test_not_computable_receipt_contains_identity()
    print("receipt identity contract: PASS")


if __name__ == "__main__":
    main()
