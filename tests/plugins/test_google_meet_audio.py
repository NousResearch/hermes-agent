"""Tests for plugins.google_meet.audio_bridge (v2).

Covers the platform gating and pactl / system_profiler plumbing
without actually invoking those tools on the host.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


# ---------------------------------------------------------------------------
# Linux setup / teardown
# ---------------------------------------------------------------------------


def _linux_pactl_result(stdout: str) -> MagicMock:
    """Build a fake CompletedProcess-ish object for subprocess.run."""
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = 0
    return m


def test_setup_linux_loads_null_sink_and_virtual_source():
    from plugins.google_meet.audio_bridge import AudioBridge

    calls: list[list[str]] = []

    def _fake_run(argv, **kwargs):
        calls.append(list(argv))
        # First call = null-sink → module id 42
        # Second call = virtual-source → module id 43
        if "module-null-sink" in argv:
            return _linux_pactl_result("42\n")
        if "module-virtual-source" in argv:
            return _linux_pactl_result("43\n")
        raise AssertionError(f"unexpected pactl invocation: {argv}")

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Linux"), \
         patch("plugins.google_meet.audio_bridge.subprocess.run",
               side_effect=_fake_run):
        br = AudioBridge()
        info = br.setup()

    # Two pactl load-module calls, in order.
    assert len(calls) == 2
    assert calls[0][0] == "pactl" and calls[0][1] == "load-module"
    assert "module-null-sink" in calls[0]
    assert any(a.startswith("sink_name=hermes_meet_sink") for a in calls[0])
    assert calls[1][0] == "pactl" and calls[1][1] == "load-module"
    assert "module-virtual-source" in calls[1]
    assert any(a.startswith("source_name=hermes_meet_src") for a in calls[1])
    assert any("master=hermes_meet_sink.monitor" in a for a in calls[1])

    # Dict shape.
    assert info["platform"] == "linux"
    assert info["device_name"] == "hermes_meet_src"
    assert info["write_target"] == "hermes_meet_sink"
    assert info["sample_rate"] == 48000
    assert info["channels"] == 2
    assert info["module_ids"] == [42, 43]

    # Properties.
    assert br.device_name == "hermes_meet_src"
    assert br.write_target == "hermes_meet_sink"


def test_teardown_linux_unloads_modules_in_reverse_order():
    from plugins.google_meet.audio_bridge import AudioBridge

    def _setup_run(argv, **kwargs):
        if "module-null-sink" in argv:
            return _linux_pactl_result("42\n")
        return _linux_pactl_result("43\n")

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Linux"), \
         patch("plugins.google_meet.audio_bridge.subprocess.run",
               side_effect=_setup_run):
        br = AudioBridge()
        br.setup()

    unload_calls: list[list[str]] = []

    def _teardown_run(argv, **kwargs):
        unload_calls.append(list(argv))
        return _linux_pactl_result("")

    with patch("plugins.google_meet.audio_bridge.subprocess.run",
               side_effect=_teardown_run):
        br.teardown()

    # Two unload calls, in reverse order: 43 (virtual-source) then 42 (sink).
    assert [c[1] for c in unload_calls] == ["unload-module", "unload-module"]
    assert unload_calls[0][2] == "43"
    assert unload_calls[1][2] == "42"

    # Second teardown is a no-op.
    with patch("plugins.google_meet.audio_bridge.subprocess.run") as run_mock:
        br.teardown()
    run_mock.assert_not_called()


def test_setup_linux_parses_module_id_from_multi_line_output():
    """Some pactl builds include trailing whitespace / notices."""
    from plugins.google_meet.audio_bridge import AudioBridge

    def _fake_run(argv, **kwargs):
        if "module-null-sink" in argv:
            return _linux_pactl_result("42   \n")
        return _linux_pactl_result("43\n")

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Linux"), \
         patch("plugins.google_meet.audio_bridge.subprocess.run",
               side_effect=_fake_run):
        br = AudioBridge()
        info = br.setup()

    assert info["module_ids"] == [42, 43]


def test_setup_linux_pactl_missing_raises_clean_error():
    from plugins.google_meet.audio_bridge import AudioBridge

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Linux"), \
         patch("plugins.google_meet.audio_bridge.subprocess.run",
               side_effect=FileNotFoundError("pactl")):
        br = AudioBridge()
        with pytest.raises(RuntimeError, match="pactl"):
            br.setup()


# ---------------------------------------------------------------------------
# macOS setup
# ---------------------------------------------------------------------------

_BH_PRESENT = (
    "Audio:\n"
    "    Devices:\n"
    "        BlackHole 2ch:\n"
    "          Manufacturer: Existential Audio\n"
)

_BH_ABSENT = (
    "Audio:\n"
    "    Devices:\n"
    "        MacBook Pro Microphone:\n"
    "          Default Input: Yes\n"
)


def test_setup_darwin_returns_blackhole_when_present():
    from plugins.google_meet.audio_bridge import AudioBridge

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Darwin"), \
         patch("plugins.google_meet.audio_bridge.subprocess.check_output",
               return_value=_BH_PRESENT) as check:
        br = AudioBridge()
        info = br.setup()

    check.assert_called_once()
    argv = check.call_args.args[0]
    assert argv[0] == "system_profiler"
    assert "SPAudioDataType" in argv

    assert info["platform"] == "darwin"
    assert info["device_name"] == "BlackHole 2ch"
    assert info["write_target"] == "BlackHole 2ch"
    assert info["module_ids"] == []
    assert info["sample_rate"] == 48000
    assert info["channels"] == 2

    # teardown is a no-op on darwin (no modules to unload).
    with patch("plugins.google_meet.audio_bridge.subprocess.run") as run_mock:
        br.teardown()
    run_mock.assert_not_called()


def test_setup_darwin_raises_when_blackhole_missing():
    from plugins.google_meet.audio_bridge import AudioBridge

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Darwin"), \
         patch("plugins.google_meet.audio_bridge.subprocess.check_output",
               return_value=_BH_ABSENT):
        br = AudioBridge()
        with pytest.raises(RuntimeError, match="BlackHole"):
            br.setup()


# ---------------------------------------------------------------------------
# Windows / unsupported
# ---------------------------------------------------------------------------


def test_setup_windows_raises():
    from plugins.google_meet.audio_bridge import AudioBridge

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Windows"):
        br = AudioBridge()
        with pytest.raises(RuntimeError, match="not supported"):
            br.setup()


# ---------------------------------------------------------------------------
# chrome_fake_audio_flags
# ---------------------------------------------------------------------------


def test_chrome_fake_audio_flags_linux():
    from plugins.google_meet.audio_bridge import chrome_fake_audio_flags

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Linux"):
        flags = chrome_fake_audio_flags(
            {"platform": "linux", "device_name": "hermes_meet_src"}
        )
    assert "--use-fake-ui-for-media-stream" in flags


def test_chrome_fake_audio_flags_darwin():
    from plugins.google_meet.audio_bridge import chrome_fake_audio_flags

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Darwin"):
        flags = chrome_fake_audio_flags(
            {"platform": "darwin", "device_name": "BlackHole 2ch"}
        )
    assert "--use-fake-ui-for-media-stream" in flags


def test_chrome_fake_audio_flags_windows_raises():
    from plugins.google_meet.audio_bridge import chrome_fake_audio_flags

    with patch("plugins.google_meet.audio_bridge.platform.system",
               return_value="Windows"):
        with pytest.raises(RuntimeError):
            chrome_fake_audio_flags({"platform": "windows"})


def test_property_access_before_setup_raises():
    from plugins.google_meet.audio_bridge import AudioBridge

    br = AudioBridge()
    with pytest.raises(RuntimeError):
        _ = br.device_name
    with pytest.raises(RuntimeError):
        _ = br.write_target
