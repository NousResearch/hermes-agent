from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_faster_whisper_is_not_a_base_dependency():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    assert not any(dep.startswith("faster-whisper") for dep in deps)

    voice_extra = data["project"]["optional-dependencies"]["voice"]
    assert any(dep.startswith("faster-whisper") for dep in voice_extra)


def test_document_processing_libraries_are_base_dependencies():
    """Office/PDF helpers must be present in the agent venv.

    WebUI startup installs agent dependencies, not WebUI server dependencies,
    so these need to live in the base install rather than a WebUI-only target.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    required_document_packages = {
        "python-docx",
        "openpyxl",
        "python-pptx",
        "pypdf",
        "pdfplumber",
    }
    declared_packages = {
        dep.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].strip()
        for dep in deps
    }

    assert required_document_packages <= declared_packages


def test_manifest_includes_bundled_skills():
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "graft skills" in manifest
    assert "graft optional-skills" in manifest
