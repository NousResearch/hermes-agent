from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_device_screen_guides_direct_shared_folder_and_accessibility_targeting():
    device_screen = (
        REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt"
    ).read_text(encoding="utf-8")
    strings = (
        REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt"
    ).read_text(encoding="utf-8")

    assert 'Text(strings.deviceGuideStep(1))' in device_screen
    assert 'Text(strings.deviceGuideStep(2))' in device_screen
    assert "android_shared_folder_list/read/write" in strings
    assert "android_ui_snapshot and target controls with android_ui_action" in device_screen
    assert "Hermes now ships a local Linux command suite inside the Android app." in strings
    assert 'Wi-Fi + connectivity' in device_screen
    assert 'Bluetooth' in device_screen
    assert 'USB + NFC' in device_screen
    assert 'Notifications + background runtime' in device_screen
    assert 'Overlay permission' in device_screen
    assert 'Resizable window support' in device_screen
