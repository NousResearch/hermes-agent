from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "desktop-update" / "windows.ps1"


def test_failed_update_does_not_relaunch_modified_installation():
    source = SCRIPT.read_text(encoding="utf-8-sig")

    assert "$updateAttempted = $false" in source
    assert source.count("$updateAttempted = $true") >= 2
    failure_branch = source.split("if ($finalCode -ne 0) {", 1)[1].split("} else {", 1)[0]
    assert "if (-not $updateAttempted) { [void](Start-DesktopRelaunch) }" in failure_branch


def test_pre_update_failures_can_still_relaunch():
    source = SCRIPT.read_text(encoding="utf-8-sig")
    assert "only pre-update hand-off failures may" in source
    assert "if (-not $updateAttempted)" in source
