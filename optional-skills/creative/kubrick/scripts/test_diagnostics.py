#!/usr/bin/env python3
"""Failure-matrix tests for Kubrick structured diagnostics and exit codes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCHEMA = json.loads((ROOT / "schemas/diagnostic.schema.json").read_text(encoding="utf-8"))
REQUIRED = set(SCHEMA["required"])


def assert_diagnostic(payload: dict, *, status: str, exit_code: int) -> None:
    assert REQUIRED <= set(payload), payload
    assert set(payload) <= set(SCHEMA["properties"]), payload
    assert payload["schema_version"] == "1.0.0"
    assert payload["status"] == status
    assert payload["exit_code"] == exit_code
    assert payload["reason_vector"]
    assert isinstance(payload["context"], dict)
    try:
        import jsonschema
    except ImportError:
        return
    jsonschema.validate(payload, SCHEMA)


def run_json(argv: list[str], *, python_s: bool = False) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, KUBRICK_DIAGNOSTICS="json")
    prefix = [PY, "-S"] if python_s else [PY]
    return subprocess.run(prefix + argv, cwd=ROOT, env=env, text=True, capture_output=True)


def test_router_failures_are_structured_exit_2() -> None:
    for args in (["do", "missing"], ["recipe"], ["do", "adapt", "--action", "missing"]):
        result = run_json([str(ROOT / "scripts/kubrick.py"), *args])
        assert result.returncode == 2, (args, result.stdout, result.stderr)
        assert_diagnostic(json.loads(result.stderr), status="INVALID_COMMAND", exit_code=2)


def test_human_router_error_remains_readable() -> None:
    result = subprocess.run(
        [PY, str(ROOT / "scripts/kubrick.py"), "do", "missing"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert result.stderr.startswith("unknown intent")
    assert not result.stderr.lstrip().startswith("{")


def test_missing_optional_dependency_is_exit_3() -> None:
    result = run_json([str(ROOT / "scripts/kubrick_compile.py")], python_s=True)
    assert result.returncode == 3, (result.stdout, result.stderr)
    payload = json.loads(result.stderr)
    assert_diagnostic(payload, status="DEPENDENCY_UNAVAILABLE", exit_code=3)
    assert payload["context"]["package"] == "PyYAML"


def test_not_computable_is_exit_4_and_embedded() -> bool:
    try:
        import yaml
    except ImportError:
        return False

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
            [PY, str(ROOT / "scripts/kubrick.py"), "compile", "--brief", str(brief), "--out", str(out)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 4, (result.stdout, result.stderr)
        receipt = json.loads((out / "compile-receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "NOT_COMPUTABLE"
        assert_diagnostic(receipt["diagnostic"], status="NOT_COMPUTABLE", exit_code=4)
    return True


def main() -> None:
    test_router_failures_are_structured_exit_2()
    test_human_router_error_remains_readable()
    test_missing_optional_dependency_is_exit_3()
    validation_case_ran = test_not_computable_is_exit_4_and_embedded()
    profile = "validation" if validation_case_ran else "stdlib"
    print(f"structured diagnostic failure matrix ({profile} profile): PASS")


if __name__ == "__main__":
    main()
