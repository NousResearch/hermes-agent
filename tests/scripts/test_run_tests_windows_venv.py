from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_tests.sh"


def test_runner_supports_posix_and_windows_virtualenv_layouts():
    script = RUNNER.read_text(encoding="utf-8")

    assert 'if [ -f "$candidate/bin/activate" ]' in script
    assert 'PYTHON="$candidate/bin/python"' in script
    assert 'if [ -f "$candidate/Scripts/python.exe" ]' in script
    assert 'PYTHON="$candidate/Scripts/python.exe"' in script
    assert 'PYTHON="$VENV/bin/python"' not in script


def test_runner_preserves_windows_home_and_utf8_in_clean_environment():
    script = RUNNER.read_text(encoding="utf-8")

    assert 'USERPROFILE="${USERPROFILE:-$HOME}"' in script
    assert "PYTHONUTF8=1" in script
