#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Tests for scaffold-setup-skill.py"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scaffold-setup-skill.py"
TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "setup-skill-template"


def run_scaffold(tmp: Path, **kwargs) -> tuple[int, dict]:
    """Run the scaffold script and return (exit_code, parsed_json)."""
    target_dir = kwargs.get("target_dir", str(tmp / "output"))
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    module_code = kwargs.get("module_code", "tst")
    module_name = kwargs.get("module_name", "Test Module")

    yaml_path = tmp / "module.yaml"
    csv_path = tmp / "module-help.csv"
    yaml_path.write_text(kwargs.get("yaml_content", f'code: {module_code}\nname: "{module_name}"\n'))
    csv_path.write_text(
        kwargs.get(
            "csv_content",
            "module,skill,display-name,menu-code,description,action,args,phase,after,before,required,output-location,outputs\n"
            f'{module_name},{module_code}-example,Example,EX,An example skill,do-thing,,anytime,,,false,output_folder,artifact\n',
        )
    )

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--target-dir", target_dir,
        "--module-code", module_code,
        "--module-name", module_name,
        "--module-yaml", str(yaml_path),
        "--module-csv", str(csv_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        data = {"raw_stdout": result.stdout, "raw_stderr": result.stderr}
    return result.returncode, data


def test_basic_scaffold():
    """Test that scaffolding creates the expected structure."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        target_dir = tmp / "output"
        target_dir.mkdir()

        code, data = run_scaffold(tmp, target_dir=str(target_dir))
        assert code == 0, f"Script failed: {data}"
        assert data["status"] == "success"
        assert data["setup_skill"] == "tst-setup"

        setup_dir = target_dir / "tst-setup"
        assert setup_dir.is_dir()
        assert (setup_dir / "SKILL.md").is_file()
        assert (setup_dir / "scripts" / "merge-config.py").is_file()
        assert (setup_dir / "scripts" / "merge-help-csv.py").is_file()
        assert (setup_dir / "scripts" / "cleanup-legacy.py").is_file()
        assert (setup_dir / "assets" / "module.yaml").is_file()
        assert (setup_dir / "assets" / "module-help.csv").is_file()


def test_skill_md_frontmatter_substitution():
    """Test that SKILL.md placeholders are replaced."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        target_dir = tmp / "output"
        target_dir.mkdir()

        code, data = run_scaffold(
            tmp,
            target_dir=str(target_dir),
            module_code="xyz",
            module_name="XYZ Studio",
        )
        assert code == 0

        skill_md = (target_dir / "xyz-setup" / "SKILL.md").read_text()
        assert "xyz-setup" in skill_md
        assert "XYZ Studio" in skill_md
        assert "{setup-skill-name}" not in skill_md
        assert "{module-name}" not in skill_md
        assert "{module-code}" not in skill_md


def test_template_frontmatter_uses_quoted_name_placeholder():
    """Test that the template frontmatter is valid before substitution."""
    template_skill_md = (TEMPLATE_DIR / "SKILL.md").read_text()
    assert 'name: "{setup-skill-name}"' in template_skill_md


def test_generated_files_written():
    """Test that module.yaml and module-help.csv contain generated content."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        target_dir = tmp / "output"
        target_dir.mkdir()

        custom_yaml = 'code: abc\nname: "ABC Module"\ndescription: "Custom desc"\n'
        custom_csv = "module,skill,display-name,menu-code,description,action,args,phase,after,before,required,output-location,outputs\nABC Module,bmad-abc-thing,Do Thing,DT,Does the thing,run,,anytime,,,false,output_folder,report\n"

        code, data = run_scaffold(
            tmp,
            target_dir=str(target_dir),
            module_code="abc",
            module_name="ABC Module",
            yaml_content=custom_yaml,
            csv_content=custom_csv,
        )
        assert code == 0

        yaml_content = (target_dir / "abc-setup" / "assets" / "module.yaml").read_text()
        assert "ABC Module" in yaml_content
        assert "Custom desc" in yaml_content

        csv_content = (target_dir / "abc-setup" / "assets" / "module-help.csv").read_text()
        assert "bmad-abc-thing" in csv_content
        assert "DT" in csv_content


def test_anti_zombie_replaces_existing():
    """Test that an existing setup skill is replaced cleanly."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        target_dir = tmp / "output"
        target_dir.mkdir()

        # First scaffold
        run_scaffold(tmp, target_dir=str(target_dir))
        stale_file = target_dir / "tst-setup" / "stale-marker.txt"
        stale_file.write_text("should be removed")

        # Second scaffold should remove stale file
        code, data = run_scaffold(tmp, target_dir=str(target_dir))
        assert code == 0
        assert not stale_file.exists()


def test_missing_target_dir():
    """Test error when target directory doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nonexistent = tmp / "nonexistent"

        # Write valid source files
        yaml_path = tmp / "module.yaml"
        csv_path = tmp / "module-help.csv"
        yaml_path.write_text('code: tst\nname: "Test"\n')
        csv_path.write_text("header\n")

        cmd = [
            sys.executable,
            str(SCRIPT),
            "--target-dir", str(nonexistent),
            "--module-code", "tst",
            "--module-name", "Test",
            "--module-yaml", str(yaml_path),
            "--module-csv", str(csv_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 2
        data = json.loads(result.stdout)
        assert data["status"] == "error"


def test_missing_source_file():
    """Test error when module.yaml source doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        target_dir = tmp / "output"
        target_dir.mkdir()

        # Remove the yaml after creation to simulate missing file
        yaml_path = tmp / "module.yaml"
        csv_path = tmp / "module-help.csv"
        csv_path.write_text("header\n")
        # Don't create yaml_path

        cmd = [
            sys.executable,
            str(SCRIPT),
            "--target-dir", str(target_dir),
            "--module-code", "tst",
            "--module-name", "Test",
            "--module-yaml", str(yaml_path),
            "--module-csv", str(csv_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 2
        data = json.loads(result.stdout)
        assert data["status"] == "error"


if __name__ == "__main__":
    tests = [
        test_basic_scaffold,
        test_skill_md_frontmatter_substitution,
        test_template_frontmatter_uses_quoted_name_placeholder,
        test_generated_files_written,
        test_anti_zombie_replaces_existing,
        test_missing_target_dir,
        test_missing_source_file,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
