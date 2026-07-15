"""Hermetic tests for the live Migrate Web flow gate (MW-P6-I1 + I2)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FLOW_EVAL_PATH = REPO_ROOT / "scripts" / "mw" / "flow_eval.py"
FLOW_GATE_PATH = REPO_ROOT / "scripts" / "mw" / "flow_gate.py"
DEFAULT_RULES_PATH = REPO_ROOT / "scripts" / "mw" / "flow-rules.yaml"

_spec = importlib.util.spec_from_file_location("mw_flow_eval", FLOW_EVAL_PATH)
assert _spec and _spec.loader
flow_eval = importlib.util.module_from_spec(_spec)
sys.modules["mw_flow_eval"] = flow_eval
_spec.loader.exec_module(flow_eval)


@pytest.fixture
def rules() -> Dict[str, Any]:
    return flow_eval.load_rules(DEFAULT_RULES_PATH)


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    return subprocess.run(
        [sys.executable, str(FLOW_GATE_PATH), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )


def _run_json(*args: str, cwd: Path) -> Tuple[int, Dict[str, Any]]:
    proc = _run(*args, "--json", cwd=cwd)
    data = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, data


def _materialize_output(root: Path, menu: str, output: Dict[str, Any]) -> None:
    relative = output["path"].replace("{menu}", menu)
    tokens: Iterable[str] = list(output["must_contain"]) + list(
        output["must_contain_any"][:1]
    )
    content = "\n".join(tokens) + "\n"
    needed = max(0, output["min_bytes"] - len(content.encode("utf-8")))
    _write(root / relative, content + ("x" * needed))


def _seed_through(root: Path, menu: str, rules: Dict[str, Any], final_step: str) -> None:
    for step in rules["steps"]:
        for output in step["outputs"]:
            _materialize_output(root, menu, output)
        if step["id"] == final_step:
            return
    raise AssertionError(f"unknown fixture step {final_step}")


def _small_rules(*, duplicate: bool = False, outputs: bool = True) -> str:
    second_id = "M0" if duplicate else "M1"
    lines = ["version: 1", "steps:", "  - id: M0", '    title: "first"']
    if outputs:
        lines.extend(
            [
                "    outputs:",
                '      - path: ".work/{menu}/proof.md"',
                "        min_bytes: 1",
            ]
        )
    lines.extend(
        [
            f"  - id: {second_id}",
            '    title: "second"',
            "    outputs:",
            '      - path: ".work/{menu}/second.md"',
            "        min_bytes: 1",
        ]
    )
    return "\n".join(lines) + "\n"


def test_new_menu_is_at_m0_and_can_enter_m0(tmp_path: Path, rules: Dict[str, Any]) -> None:
    result = flow_eval.evaluate(tmp_path, "home", rules)
    assert result["current_step"] == "M0"
    assert result["done_count"] == 0
    code, data = _run_json("can-enter", "M0", "home", cwd=tmp_path)
    assert code == 0
    assert data["can_enter"] is True


def test_complete_through_m35_can_enter_m4(tmp_path: Path, rules: Dict[str, Any]) -> None:
    _seed_through(tmp_path, "home", rules, "M3.5")
    code, data = _run_json("can-enter", "M4", "home", cwd=tmp_path)
    assert code == 0
    assert data["can_enter"] is True
    assert data["missing"] == []


def test_skipping_m0_blocks_m4_and_names_m0(tmp_path: Path, rules: Dict[str, Any]) -> None:
    for step in rules["steps"][1:7]:
        for output in step["outputs"]:
            _materialize_output(tmp_path, "home", output)
    code, data = _run_json("can-enter", "M4", "home", cwd=tmp_path)
    assert code == 1
    assert data["can_enter"] is False
    assert any(reason.startswith("M0:") for reason in data["missing"])


@pytest.mark.parametrize("content", ["", "owner_quote:\ndate:\nmenu_url:\n"])
def test_empty_or_short_file_stays_pending(
    tmp_path: Path, rules: Dict[str, Any], content: str
) -> None:
    _write(tmp_path / ".project/menu-briefs/home.confirm.md", content)
    step = flow_eval.evaluate(tmp_path, "home", rules)["steps"][0]
    assert step["status"] == "pending"
    assert any("below minimum" in reason for reason in step["missing"])


def test_missing_required_text_is_reported(tmp_path: Path, rules: Dict[str, Any]) -> None:
    path = tmp_path / ".project/menu-briefs/home.confirm.md"
    _write(path, "owner_quote:\ndate:\n" + ("x" * 300))
    step = flow_eval.evaluate(tmp_path, "home", rules)["steps"][0]
    assert step["status"] == "pending"
    assert any("menu_url:" in reason for reason in step["missing"])


def test_must_contain_any_accepts_no_input(tmp_path: Path, rules: Dict[str, Any]) -> None:
    _materialize_output(tmp_path, "home", rules["steps"][0]["outputs"][0])
    _write(tmp_path / ".work/menus/home/benchmark.md", "NO_INPUT" + ("x" * 20))
    step = flow_eval.evaluate(tmp_path, "home", rules)["steps"][1]
    assert step["status"] == "done"


def test_unknown_step_exits_2(tmp_path: Path) -> None:
    code, data = _run_json("can-enter", "M99", "home", cwd=tmp_path)
    assert code == 2
    assert "unknown step id" in data["error"]


def test_missing_both_default_rules_locations_exits_2(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "isolated-tool"
    tool_dir.mkdir()
    shutil.copy2(FLOW_GATE_PATH, tool_dir / "flow_gate.py")
    shutil.copy2(FLOW_EVAL_PATH, tool_dir / "flow_eval.py")
    project = tmp_path / "project-without-rules"
    project.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(tool_dir / "flow_gate.py"),
            "status",
            "home",
            "--project-root",
            str(project),
            "--json",
        ],
        cwd=str(project),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "neither" in json.loads(proc.stdout)["error"]


@pytest.mark.parametrize(
    "rules_text, expected",
    [(_small_rules(duplicate=True), "duplicate step id"), (_small_rules(outputs=False), "outputs")],
)
def test_broken_rules_exit_2(tmp_path: Path, rules_text: str, expected: str) -> None:
    path = _write(tmp_path / "bad.yaml", rules_text)
    code, data = _run_json(
        "status", "home", "--rules", str(path), cwd=tmp_path
    )
    assert code == 2
    assert expected in data["error"]


@pytest.mark.parametrize("menu", ["../etc", "a/b", ""])
def test_dangerous_menu_slug_exits_2(tmp_path: Path, menu: str) -> None:
    code, data = _run_json("status", menu, cwd=tmp_path)
    assert code == 2
    assert "invalid menu slug" in data["error"]


def test_project_rules_override_default_completely(tmp_path: Path) -> None:
    override = textwrap.dedent(
        """\
        version: 1
        steps:
          - id: ONLY
            title: "project override"
            outputs:
              - path: ".work/{menu}/only.md"
                min_bytes: 1
        """
    )
    _write(tmp_path / ".work/flow-rules.yaml", override)
    code, data = _run_json("status", "home", cwd=tmp_path)
    assert code == 0
    assert data["total"] == 1
    assert [step["id"] for step in data["steps"]] == ["ONLY"]


def test_json_status_has_all_live_fields(tmp_path: Path) -> None:
    code, data = _run_json("status", "home", cwd=tmp_path)
    assert code == 0
    assert len(data["steps"]) == 13
    assert data["current_step"] == "M0"
    assert data["done_count"] == 0
    assert data["total"] == len(data["steps"])


def test_missing_m25_blocks_m3_even_if_m0_to_m2_exist(
    tmp_path: Path, rules: Dict[str, Any]
) -> None:
    _seed_through(tmp_path, "home", rules, "M2")
    code, data = _run_json("can-enter", "M3", "home", cwd=tmp_path)
    assert code == 1
    assert any(reason.startswith("M2.5:") for reason in data["missing"])


def test_invalid_utf8_evidence_fails_closed(tmp_path: Path, rules: Dict[str, Any]) -> None:
    path = tmp_path / ".project/menu-briefs/home.confirm.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff" * 300)
    step = flow_eval.evaluate(tmp_path, "home", rules)["steps"][0]
    assert step["status"] == "pending"
    assert any("cannot read UTF-8" in reason for reason in step["missing"])


def test_rules_path_traversal_is_rejected(tmp_path: Path) -> None:
    bad = textwrap.dedent(
        """\
        steps:
          - id: M0
            title: "bad path"
            outputs:
              - path: "../{menu}.md"
        """
    )
    with pytest.raises(flow_eval.ConfigError, match="forbidden"):
        flow_eval.validate_rules(flow_eval.load_yaml_text(bad, force_mini=True))


def test_mini_yaml_fallback_loads_default_rules() -> None:
    loaded = flow_eval.load_rules(DEFAULT_RULES_PATH, force_mini=True)
    assert len(loaded["steps"]) == 13
    assert loaded["steps"][1]["outputs"][0]["must_contain_any"] == [
        "NO_INPUT",
        "benchmark_source:",
    ]
