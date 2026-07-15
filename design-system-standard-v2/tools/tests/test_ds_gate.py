import json
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
GATE = TOOLS_DIR / "ds-gate.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run_gate(*args):
    return subprocess.run(
        [sys.executable, str(GATE), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_complete_design_system_passes_all_layers():
    result = run_gate("--file", FIXTURES / "designsystem-pass.md", "--layer", "all", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["errors"] == []


def test_incomplete_h_layer_reports_h2_and_h5():
    result = run_gate("--file", FIXTURES / "designsystem-fail-h.md", "--layer", "H")

    assert result.returncode == 1
    assert "[H2]" in result.stdout
    assert "[H5]" in result.stdout


def test_missing_h0_reports_h0(tmp_path):
    source = (FIXTURES / "designsystem-pass.md").read_text(encoding="utf-8")
    start = source.index("## H0 เอกสารโปรเจกต์ที่อ่าน")
    end = source.index("## H1 บัตรประจำตัวโครงการ")
    design_system = tmp_path / "DesignSystem.md"
    design_system.write_text(source[:start] + source[end:], encoding="utf-8")

    result = run_gate("--file", design_system, "--layer", "H")

    assert result.returncode == 1
    assert "[H0]" in result.stdout


def test_h4_without_language_switch_fails(tmp_path):
    source = (FIXTURES / "designsystem-pass.md").read_text(encoding="utf-8")
    source = source.replace("- ปุ่มสลับภาษา: TH/EN\n", "")
    design_system = tmp_path / "DesignSystem.md"
    design_system.write_text(source, encoding="utf-8")

    result = run_gate("--file", design_system, "--layer", "H")

    assert result.returncode == 1
    assert "[H4]" in result.stdout


def test_incomplete_f_layer_reports_f5_and_f7():
    result = run_gate("--file", FIXTURES / "designsystem-fail-f.md", "--layer", "F")

    assert result.returncode == 1
    assert "[F5]" in result.stdout
    assert "[F7]" in result.stdout


def test_u3_without_six_part_page_structure_fails(tmp_path):
    source = (FIXTURES / "designsystem-pass.md").read_text(encoding="utf-8")
    start = source.index("โครงหน้า 6 ส่วน:\n")
    end = source.index("\n\n## U4 Anti-patterns", start)
    design_system = tmp_path / "DesignSystem.md"
    design_system.write_text(source[:start] + source[end:], encoding="utf-8")

    result = run_gate("--file", design_system, "--layer", "U")

    assert result.returncode == 1
    assert "[U3]" in result.stdout


def test_f5_plain_numbers_without_units_fail(tmp_path):
    source = (FIXTURES / "designsystem-pass.md").read_text(encoding="utf-8")
    start = source.index("## F5 Emotion 6 แกน")
    end = source.index("## F6 Function → Component")
    false_tokens = """## F5 Emotion 6 แกน
| แกน | คะแนน | ค่า token ที่แปลงแล้ว |
|---|---|---|
| ทางการ | 4 | chroma 20, radius 4, motion 300 |
| อบอุ่น | 4 | hue 35, radius 10, saturation 60 |
| พลัง | 2 | saturation 38, motion 292, rotation 0 |
| ชัด | 5 | contrast 7, weight 600, border 2 |
| หนาแน่น | 2 | spacing 0.94, row 51, gap 16 |

"""
    design_system = tmp_path / "DesignSystem.md"
    design_system.write_text(source[:start] + false_tokens + source[end:], encoding="utf-8")

    result = run_gate("--file", design_system, "--layer", "F")

    assert result.returncode == 1
    assert "[F5]" in result.stdout


def test_d_layer_checks_only_d17(tmp_path):
    design_system = tmp_path / "DesignSystem.md"
    design_system.write_text(
        "# Design System\n\n## ข้อห้ามที่เกี่ยว (D17)\n- ห้ามเปลี่ยนเทคโนโลยีหลักโดยไม่ขออนุมัติ\n",
        encoding="utf-8",
    )

    result = run_gate("--file", design_system, "--layer", "D", "--json")

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["checked"] == ["D17"]
    assert payload["errors"] == []


def test_init_creates_todo_template_that_fails_gate_and_never_overwrites(tmp_path):
    design_system = tmp_path / ".project" / "DesignSystem.md"

    created = run_gate("--file", design_system, "--init")
    assert created.returncode == 0
    original = design_system.read_text(encoding="utf-8")
    assert "TODO" in original

    checked = run_gate("--file", design_system, "--layer", "all")
    assert checked.returncode == 1
    assert "ยังมี TODO" in checked.stdout

    repeated = run_gate("--file", design_system, "--init")
    assert repeated.returncode == 1
    assert "มีไฟล์อยู่แล้ว" in repeated.stdout
    assert design_system.read_text(encoding="utf-8") == original


def test_missing_file_exits_two_and_recommends_init(tmp_path):
    missing = tmp_path / "missing.md"

    result = run_gate("--file", missing)

    assert result.returncode == 2
    assert "ไม่พบไฟล์" in result.stdout
    assert "--init" in result.stdout
