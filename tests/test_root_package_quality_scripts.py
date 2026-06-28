import json
from pathlib import Path


def test_root_package_exposes_quality_gate_scripts():
    package_json = json.loads(Path("package.json").read_text(encoding="utf-8"))
    scripts = package_json.get("scripts", {})

    assert scripts["test"] == "scripts/run_tests.sh"
    assert scripts["lint"] == "uv run ruff check ."
    assert "npm run lint" in scripts["check"]
    assert "npm test" in scripts["check"]
