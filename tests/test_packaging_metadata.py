from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_faster_whisper_is_not_a_base_dependency():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    assert not any(dep.startswith("faster-whisper") for dep in deps)

    voice_extra = data["project"]["optional-dependencies"]["voice"]
    assert any(dep.startswith("faster-whisper") for dep in voice_extra)


def test_manifest_includes_bundled_skills():
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "graft skills" in manifest
    assert "graft optional-skills" in manifest


def test_dev_extra_declares_pytest_split():
    """`scripts/run_tests.sh` requires pytest_split. Declare it in the dev
    extra so a fresh `pip install -e .[dev]` (or `uv sync --extra dev`)
    provisions everything the canonical test runner needs, instead of
    forcing the script to self-bootstrap into the venv (#22401)."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev_extra = data["project"]["optional-dependencies"]["dev"]

    assert any(dep.startswith("pytest-split") for dep in dev_extra), (
        f"pytest-split missing from dev extra: {dev_extra}"
    )
