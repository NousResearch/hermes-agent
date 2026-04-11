from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_device_screen_guides_direct_shared_folder_and_accessibility_targeting():
    device_screen = (
        REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt"
    ).read_text(encoding="utf-8")

    assert "android_shared_folder_list/read/write" in device_screen
    assert "android_ui_snapshot and target controls with android_ui_action" in device_screen
    assert "Hermes now ships a local Linux command suite inside the Android app." in device_screen
