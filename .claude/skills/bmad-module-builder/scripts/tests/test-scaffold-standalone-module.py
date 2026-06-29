#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Tests for scaffold-standalone-module.py"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scaffold-standalone-module.py"


def make_skill_dir(tmp: Path, name: str = "my-skill") -> Path:
    """Create a minimal skill directory with SKILL.md and assets/module.yaml."""
    skill_dir = tmp / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: A test skill\n---\n# My Skill\n")
    assets = skill_dir / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "module.yaml").write_text(
        'code: tst\nname: "Test Module"\ndescription: "A test module"\nmodule_version: 1.0.0\n'
    )
    (assets / "module-help.csv").write_text(
        "module,skill,display-name,menu-code,description,action,args,phase,after,before,required,output-location,outputs\n"
        "Test Module,my-skill,Do Thing,DT,Does the thing,run,,anytime,,,false,output_folder,artifact\n"
    )
    return skill_dir


def run_scaffold(skill_dir: Path, **kwargs) -> tuple[int, dict]:
    """Run the standalone scaffold script and return (exit_code, parsed_json)."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--skill-dir", str(skill_dir),
        "--module-code", kwargs.get("module_code", "tst"),
        "--module-name", kwargs.get("module_name", "Test Module"),
    ]
    if "marketplace_dir" in kwargs:
        cmd.extend(["--marketplace-dir", str(kwargs["marketplace_dir"])])
    if kwargs.get("verbose"):
        cmd.append("--verbose")

    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        data = {"raw_stdout": result.stdout, "raw_stderr": result.stderr}
    return result.returncode, data


def test_basic_scaffold():
    """Test that scaffolding copies all expected template files."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        skill_dir = make_skill_dir(tmp)

        code, data = run_scaffold(skill_dir)
        assert code == 0, f"Script failed: {data}"
        assert data["status"] == "success"
        assert data["module_code"] == "tst"

        # module-setup.md placed alongside module.yaml in assets/
        assert (skill_dir / "assets" / "module-setup.md").is_file()
        # merge scripts placed in scripts/
        assert (skill_dir / "scripts" / "merge-config.py").is_file()
        assert (skill_dir / "scripts" / "merge-help-csv.py").is_file()
        # marketplace.json at parent level
        assert (tmp / ".claude-plugin" / "marketplace.json").is_file()


def test_marketplace_json_content():
    """Test that marketplace.json contains correct module metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        skill_dir = make_skill_dir(tmp, name="bmad-exc-tools")

        code, data = run_scaffold(
            skill_dir, module_code="exc", module_name="Excalidraw Tools"
        )
        assert code == 0

        marketplace = json.loads(
            (tmp / ".claude-plugin" / "marketplace.json").read_text()
        )
        assert marketplace["name"] == "bmad-exc"
        plugin = marketplace["plugins"][0]
        assert plugin["name"] == "bmad-exc"
        assert plugin["skills"] == ["./bmad-exc-tools"]
        assert plugin["description"] == "A test module"
        assert plugin["version"] == "1.0.0"


def test_does_not_overwrite_existing_scripts():
    """Test that existing scripts are skipped with a warning."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        skill_dir = make_skill_dir(tmp)

        # Pre-create a merge-config.py with custom content
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        existing_script = scripts_dir / "merge-config.py"
        existing_script.write_text("# my custom script\n")

        code, data = run_scaffold(skill_dir)
        assert code == 0

        # Should be skipped
        assert "scripts/merge-config.py" in data["files_skipped"]
        assert len(data["warnings"]) >= 1
        assert any("merge-config.py" in w for w in data["warnings"])

        # Content should be preserved
        assert existing_script.read_text() == "# my custom script\n"

        # merge-help-csv.py should still be created
        assert "scripts/merge-help-csv.py" in data["files_created"]


def test_creates_missing_subdirectories():
    """Test that scripts/ directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        skill_dir = make_skill_dir(tmp)

        # Verify scripts/ doesn't exist yet
        assert not (skill_dir / "scripts").exists()

        code, data = run_scaffold(skill_dir)
        assert code == 0
        assert (skill_dir / "scripts").is_dir()
        assert (skill_dir / "scripts" / "merge-config.py").is_file()


def test_preserves_existing_skill_files():
    """Test that existing skill files are not modified or deleted."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        skill_dir = make_skill_dir(tmp)

        # Add extra files
        (skill_dir / "build-process.md").write_text("# Build\n")
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "my-ref.md").write_text("# Reference\n")

        original_skill_md = (skill_dir / "SKILL.md").read_text()

        code, data = run_scaffold(skill_dir)
        assert code == 0

        # Original files untouched
        assert (skill_dir / "SKILL.md").read_text() == original_skill_md
        assert (skill_dir / "build-process.md").read_text() == "# Build\n"
        assert (refs_dir / "my-ref.md").read_text() == "# Reference\n"


def test_missing_skill_dir():
    """Test error when skill directory doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nonexistent = tmp / "nonexistent-skill"

        cmd = [
            sys.executable, str(SCRIPT),
            "--skill-dir", str(nonexistent),
            "--module-code", "tst",
            "--module-name", "Test",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 2
        data = json.loads(result.stdout)
        assert data["status"] == "error"


def test_missing_skill_md():
    """Test error when skill directory has no SKILL.md."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        skill_dir = tmp / "empty-skill"
        skill_dir.mkdir()
        (skill_dir / "assets").mkdir()
        (skill_dir / "assets" / "module.yaml").write_text("code: tst\n")

        cmd = [
            sys.executable, str(SCRIPT),
            "--skill-dir", str(skill_dir),
            "--module-code", "tst",
            "--module-name", "Test",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 2
        data = json.loads(result.stdout)
        assert data["status"] == "error"
        assert "SKILL.md" in data["message"]


def test_missing_module_yaml():
    """Test error when assets/module.yaml hasn't been written yet."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        skill_dir = tmp / "skill-no-yaml"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test\n---\n")

        cmd = [
            sys.executable, str(SCRIPT),
            "--skill-dir", str(skill_dir),
            "--module-code", "tst",
            "--module-name", "Test",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 2
        data = json.loads(result.stdout)
        assert data["status"] == "error"
        assert "module.yaml" in data["message"]


def test_custom_marketplace_dir():
    """Test that --marketplace-dir places marketplace.json in a custom location."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        skill_dir = make_skill_dir(tmp)
        custom_dir = tmp / "custom-root"
        custom_dir.mkdir()

        code, data = run_scaffold(skill_dir, marketplace_dir=custom_dir)
        assert code == 0

        # Should be at custom location, not default parent
        assert (custom_dir / ".claude-plugin" / "marketplace.json").is_file()
        assert not (tmp / ".claude-plugin" / "marketplace.json").exists()
        assert data["marketplace_json"] == str((custom_dir / ".claude-plugin" / "marketplace.json").resolve())


if __name__ == "__main__":
    tests = [
        test_basic_scaffold,
        test_marketplace_json_content,
        test_does_not_overwrite_existing_scripts,
        test_creates_missing_subdirectories,
        test_preserves_existing_skill_files,
        test_missing_skill_dir,
        test_missing_skill_md,
        test_missing_module_yaml,
        test_custom_marketplace_dir,
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
