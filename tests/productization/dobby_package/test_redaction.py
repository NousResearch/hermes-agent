import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REDACTION_SCRIPT = REPO_ROOT / "packaging" / "dobby-package" / "scripts" / "redaction-check.sh"
FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "preflight"


def run_redaction_check(*paths):
    return subprocess.run(
        ["bash", str(REDACTION_SCRIPT), *(str(path) for path in paths)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )


def test_redaction_check_allows_safe_fixture():
    result = run_redaction_check(FIXTURE_ROOT / "redaction-safe.txt")

    assert result.returncode == 0, result.stdout + result.stderr


def test_redaction_check_rejects_secret_shaped_fixture_without_echoing_value(tmp_path):
    raw_secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
    fixture = tmp_path / "diagnostics.txt"
    fixture.write_text(f"OPENAI_API_KEY={raw_secret}\n", encoding="utf-8")

    result = run_redaction_check(fixture)

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "openai-style key" in output
    assert raw_secret not in output
