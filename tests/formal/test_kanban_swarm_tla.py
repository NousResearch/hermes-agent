"""Execute the Kanban swarm TLA+ model when TLC tooling is supplied."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml


def test_formal_workflow_triggers_when_its_own_definition_changes():
    repo = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (repo / ".github/workflows/kanban-formal.yml").read_text(encoding="utf-8")
    )

    self_path = ".github/workflows/kanban-formal.yml"
    triggers = workflow.get("on", workflow.get(True))
    for event in ("pull_request", "push"):
        paths = triggers[event]["paths"]
        assert paths.count(self_path) == 1


def test_kanban_swarm_tlc_model_executes_fresh(tmp_path):
    java = os.environ.get("KANBAN_TLA_JAVA")
    jar = os.environ.get("TLA2TOOLS_JAR")
    if not java or not jar:
        pytest.skip("set KANBAN_TLA_JAVA and TLA2TOOLS_JAR to execute TLC")

    repo = Path(__file__).resolve().parents[2]
    semantics = tmp_path / "production-semantics.json"
    receipt = tmp_path / "kanban-swarm-tlc.json"
    export = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "export_kanban_swarm_semantics.py"),
            "--source-root",
            str(repo),
            "--output",
            str(semantics),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert export.returncode == 0, export.stdout

    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "verify_kanban_swarm_tla.py"),
            "--java",
            java,
            "--jar",
            jar,
            "--semantics",
            str(semantics),
            "--receipt",
            str(receipt),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    data = json.loads(receipt.read_text(encoding="utf-8"))
    original_receipt = copy.deepcopy(data)
    original_semantics = semantics.read_text(encoding="utf-8")
    assert data["schema"] == "hermes.kanban-swarm.tlc-receipt.v3"
    assert data["fresh_execution"] is True
    assert data["success"] is True
    assert data["result"]["states_generated"] > 0
    assert data["result"]["distinct_states"] > 0
    assert data["result"]["depth"] > 0
    assert data["result"]["states_left_on_queue"] == 0
    assert data["production_semantics"]["bound"] is True
    assert data["production_semantics"]["cases"] == 7
    assert data["production_semantics"]["board_cases"] == 7
    assert data["production_semantics"]["validation_failures"] == []
    semantics_module = Path(data["production_semantics"]["tla_module_path"])
    assert semantics_module.is_file()
    assert len(data["config"]["invariants"]) == 9
    log_text = Path(data["result"]["log_path"]).read_text(encoding="utf-8")
    assert "constant-level formula" not in log_text

    verification = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "verify_kanban_swarm_receipt.py"),
            "--receipt",
            str(receipt),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert verification.returncode == 0, verification.stdout

    semantics.write_text(semantics.read_text() + " ", encoding="utf-8")
    tampered = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "verify_kanban_swarm_receipt.py"),
            "--receipt",
            str(receipt),
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert tampered.returncode != 0
    assert "production semantics hash" in tampered.stdout

    def assert_receipt_rejected(mutated: dict, expected: str) -> None:
        receipt.write_text(json.dumps(mutated), encoding="utf-8")
        rejected = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "verify_kanban_swarm_receipt.py"),
                "--receipt",
                str(receipt),
            ],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert rejected.returncode != 0
        assert expected in rejected.stdout

    semantics.write_text(original_semantics, encoding="utf-8")
    duplicate_invariant = copy.deepcopy(original_receipt)
    duplicate_invariant["config"]["invariants"].append(
        duplicate_invariant["config"]["invariants"][0]
    )
    assert_receipt_rejected(duplicate_invariant, "invariant identities")

    missing_sources = copy.deepcopy(original_receipt)
    missing_sources["production_semantics"]["sources"] = {}
    assert_receipt_rejected(missing_sources, "production source identities")

    zero_states = copy.deepcopy(original_receipt)
    zero_states["result"]["states_generated"] = 0
    assert_receipt_rejected(zero_states, "states_generated")

    changed_semantics = json.loads(original_semantics)
    changed_semantics["board_cases"][0]["requested_board"] = "beta"
    changed_bytes = (
        json.dumps(changed_semantics, indent=2, sort_keys=True) + "\n"
    ).encode()
    semantics.write_bytes(changed_bytes)
    stale_module = copy.deepcopy(original_receipt)
    stale_module["production_semantics"]["sha256"] = hashlib.sha256(
        changed_bytes
    ).hexdigest()
    assert_receipt_rejected(
        stale_module, "semantics-to-generated-module correspondence"
    )

    for case_key, expected_failure in (
        ("cases", "semantics-to-receipt gate cases"),
        ("board_cases", "semantics-to-receipt board cases"),
    ):
        duplicated = json.loads(original_semantics)
        duplicated[case_key].append(copy.deepcopy(duplicated[case_key][0]))
        duplicated_bytes = (
            json.dumps(duplicated, indent=2, sort_keys=True) + "\n"
        ).encode()
        semantics.write_bytes(duplicated_bytes)
        forged = copy.deepcopy(original_receipt)
        forged["production_semantics"]["sha256"] = hashlib.sha256(
            duplicated_bytes
        ).hexdigest()
        assert_receipt_rejected(forged, expected_failure)

    for source_mutation in ("remove", "extra"):
        changed = json.loads(original_semantics)
        if source_mutation == "remove":
            changed["sources"] = {}
        else:
            changed["sources"]["extra.py"] = "0" * 64
        changed_bytes = (json.dumps(changed, indent=2, sort_keys=True) + "\n").encode()
        semantics.write_bytes(changed_bytes)
        forged = copy.deepcopy(original_receipt)
        forged["production_semantics"]["sha256"] = hashlib.sha256(
            changed_bytes
        ).hexdigest()
        assert_receipt_rejected(forged, "semantics source identities")

    semantics.write_text(original_semantics, encoding="utf-8")
    config_path = repo / original_receipt["config"]["path"]
    forged_config = tmp_path / "forged.cfg"
    config_text = config_path.read_text(encoding="utf-8")
    first_invariant = next(
        line for line in config_text.splitlines() if line.startswith("INVARIANT ")
    )
    forged_config.write_text(config_text + first_invariant + "\n", encoding="utf-8")
    forged = copy.deepcopy(original_receipt)
    forged["config"]["path"] = str(forged_config)
    forged["config"]["sha256"] = hashlib.sha256(forged_config.read_bytes()).hexdigest()
    assert_receipt_rejected(forged, "config-to-receipt invariant identities")

    original_log = Path(original_receipt["result"]["log_path"]).read_text(
        encoding="utf-8"
    )
    forged_log = tmp_path / "forged-zero-state.log"
    forged_log.write_text(
        re.sub(
            r"[0-9,]+ states generated, [0-9,]+ distinct states found",
            "0 states generated, 0 distinct states found",
            original_log,
        ).replace(
            "Model checking completed. No error has been found.",
            "Error: forged model failure",
        ),
        encoding="utf-8",
    )
    forged_log_receipt = copy.deepcopy(original_receipt)
    forged_log_receipt["result"]["log_path"] = str(forged_log)
    forged_log_receipt["result"]["stdout_sha256"] = hashlib.sha256(
        forged_log.read_bytes()
    ).hexdigest()
    assert_receipt_rejected(forged_log_receipt, "bound TLC success marker")

    forged_outdegree = tmp_path / "forged-outdegree.log"
    forged_outdegree.write_text(
        re.sub(
            r"The average outdegree of the complete state graph.*",
            "The average outdegree of the complete state graph is 999999999.",
            original_log,
        ),
        encoding="utf-8",
    )
    forged_outdegree_receipt = copy.deepcopy(original_receipt)
    forged_outdegree_receipt["result"]["log_path"] = str(forged_outdegree)
    forged_outdegree_receipt["result"]["stdout_sha256"] = hashlib.sha256(
        forged_outdegree.read_bytes()
    ).hexdigest()
    assert_receipt_rejected(
        forged_outdegree_receipt, "bound TLC normalized output correspondence"
    )
