from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "skill_frontmatter_doctor.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("skill_frontmatter_doctor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_skill(path: Path, content: str) -> Path:
    skill_dir = path / "demo"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


def test_scan_reports_valid_skill_without_issues(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "---\nname: demo\ndescription: Short useful description.\n---\n\n# Demo\n\nUseful steps.\n",
    )
    doctor = _load_module()

    results = doctor.scan_roots([tmp_path])

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].issues == []


def test_scan_flags_missing_required_fields_and_empty_body(tmp_path: Path) -> None:
    _write_skill(tmp_path, "---\nname: demo\n---\n\n   \n")
    doctor = _load_module()

    result = doctor.scan_roots([tmp_path])[0]

    assert result.status == "fail"
    assert {issue.code for issue in result.issues} == {"missing-description", "empty-body"}


def test_scan_flags_frontmatter_not_at_byte_zero(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "\n---\nname: demo\ndescription: Short useful description.\n---\n\n# Demo\n",
    )
    doctor = _load_module()

    result = doctor.scan_roots([tmp_path])[0]

    assert result.status == "fail"
    assert [issue.code for issue in result.issues] == ["frontmatter-not-byte-zero"]


def test_scan_flags_overlong_description(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "---\nname: demo\ndescription: " + ("x" * 1025) + "\n---\n\n# Demo\n",
    )
    doctor = _load_module()

    result = doctor.scan_roots([tmp_path])[0]

    assert result.status == "fail"
    assert [issue.code for issue in result.issues] == ["description-too-long"]


def test_cli_json_returns_nonzero_for_failures(tmp_path: Path) -> None:
    skill_file = _write_skill(tmp_path, "# No frontmatter\n")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["summary"] == {"ok": 0, "fail": 1}
    assert payload["results"][0]["path"] == str(skill_file)
    assert payload["results"][0]["issues"][0]["code"] == "missing-frontmatter"
