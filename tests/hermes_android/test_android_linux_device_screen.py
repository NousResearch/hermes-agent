from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_device_screen_mentions_linux_command_suite_and_terminal_usage():
    device_screen = (
        REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt"
    ).read_text(encoding="utf-8")

    assert 'Text("Linux command suite"' in device_screen
    assert 'terminal/process' in device_screen
    assert 'Hermes now ships a local Linux command suite inside the Android app.' in device_screen
    assert 'Ask Hermes to use terminal for commands like' in device_screen
    assert 'background runtime' in device_screen
