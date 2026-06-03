from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def test_install_ps1_rejects_unborn_git_repo_before_update_path() -> None:
    script = INSTALL_PS1.read_text(encoding="utf-8")

    assert "rev-parse --verify HEAD" in script
    assert "$headOk = ($LASTEXITCODE -eq 0)" in script
    assert "$revParseOk -and $statusOk -and $headOk" in script
    assert 'later fails at "git checkout main"' in script
