import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_prepare_android_linux_assets_script_exists_and_is_wired_into_gradle():
    script = (REPO_ROOT / "scripts/prepare_android_linux_assets.py").read_text(encoding="utf-8")
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")

    assert "def prepare_assets" in script
    assert "resolve_dependency_closure" in script
    assert "prepareHermesAndroidLinuxAssets" in gradle
    assert "generated/hermes-linux-assets" in gradle
    assert "assets.srcDir" in gradle


def test_prepare_android_linux_assets_script_imports_from_android_workdir():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/prepare_android_linux_assets.py"), "--help"],
        cwd=REPO_ROOT / "android",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Prepare Android Linux CLI assets" in result.stdout
