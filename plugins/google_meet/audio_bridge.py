"""Virtual audio bridge for feeding generated speech into Chrome's mic.

v2 module. Provisions a platform-specific virtual audio device so the
Meet bot's Chromium instance can be pointed at an input source we
control. The OpenAI Realtime client writes PCM bytes into this device;
Chrome reads them as if they were coming from a microphone.

Linux (primary): uses pactl (PulseAudio) to create a null-sink plus a
virtual source whose master is the null-sink's monitor. Callers set
PULSE_SOURCE=<source_name> in Chrome's env and pass the fake-mic flag.

macOS: requires BlackHole 2ch to be installed. This module only
verifies its presence and returns the device name; routing OS default
input is left to the user (or a future switchaudio-osx integration) to
avoid surprising the user's system audio state.

Windows: not supported in v2.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional


_BLACKHOLE_DEVICE = "BlackHole 2ch"


class AudioBridge:
    """Manages a virtual audio device for Chrome fake-mic input.

    Call ``setup()`` once before launching the Meet bot and
    ``teardown()`` when the session ends. ``teardown()`` is idempotent.
    """

    def __init__(self, name_prefix: str = "hermes_meet") -> None:
        self._name_prefix = name_prefix
        self._platform: Optional[str] = None
        self._device_name: Optional[str] = None
        self._write_target: Optional[str] = None
        self._module_ids: list[int] = []
        self._torn_down = False

    # ── public properties ─────────────────────────────────────────────────

    @property
    def device_name(self) -> str:
        if not self._device_name:
            raise RuntimeError("AudioBridge not set up yet")
        return self._device_name

    @property
    def write_target(self) -> str:
        if not self._write_target:
            raise RuntimeError("AudioBridge not set up yet")
        return self._write_target

    # ── lifecycle ─────────────────────────────────────────────────────────

    def setup(self) -> dict:
        """Provision the virtual audio device.

        Returns a dict describing the device. Raises RuntimeError on
        unsupported platforms or when required system tools are missing.
        """
        system = platform.system()
        if system == "Linux":
            return self._setup_linux()
        if system == "Darwin":
            return self._setup_darwin()
        if system == "Windows":
            raise RuntimeError("windows not supported in v2")
        raise RuntimeError(f"unsupported platform: {system}")

    def teardown(self) -> None:
        """Release the virtual audio device. Idempotent."""
        if self._torn_down:
            return
        # Only Linux needs explicit unloading.
        if self._platform == "linux" and self._module_ids:
            # Unload in reverse order (virtual-source before null-sink).
            for mod_id in reversed(self._module_ids):
                try:
                    subprocess.run(
                        ["pactl", "unload-module", str(mod_id)],
                        check=False,
                        capture_output=True,
                    )
                except Exception:
                    # Best-effort teardown — never raise from here.
                    pass
            self._module_ids = []
        self._torn_down = True

    # ── platform impls ────────────────────────────────────────────────────

    def _setup_linux(self) -> dict:
        sink_name = f"{self._name_prefix}_sink"
        src_name = f"{self._name_prefix}_src"

        try:
            sink_out = subprocess.run(
                [
                    "pactl",
                    "load-module",
                    "module-null-sink",
                    f"sink_name={sink_name}",
                    f"sink_properties=device.description=HermesMeetSink",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "pactl not found — install PulseAudio/pipewire-pulse"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"pactl load-module null-sink failed: {exc.stderr or exc}"
            ) from exc

        sink_mod_id = self._parse_module_id(sink_out.stdout)

        try:
            src_out = subprocess.run(
                [
                    "pactl",
                    "load-module",
                    "module-virtual-source",
                    f"source_name={src_name}",
                    f"master={sink_name}.monitor",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            # Roll back the null-sink we just created so we don't leak it.
            subprocess.run(
                ["pactl", "unload-module", str(sink_mod_id)],
                check=False,
                capture_output=True,
            )
            raise RuntimeError(
                f"pactl load-module virtual-source failed: {exc.stderr or exc}"
            ) from exc

        src_mod_id = self._parse_module_id(src_out.stdout)

        self._platform = "linux"
        self._device_name = src_name
        self._write_target = sink_name
        self._module_ids = [sink_mod_id, src_mod_id]
        self._torn_down = False

        return {
            "platform": "linux",
            "device_name": src_name,
            "sample_rate": 48000,
            "channels": 2,
            "module_ids": list(self._module_ids),
            "write_target": sink_name,
        }

    def _setup_darwin(self) -> dict:
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPAudioDataType"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "system_profiler not found (macOS-only command)"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"system_profiler failed: {exc.output}"
            ) from exc

        if "BlackHole" not in out:
            raise RuntimeError(
                "BlackHole virtual audio device not installed. "
                "Install via: brew install blackhole-2ch"
            )

        self._platform = "darwin"
        self._device_name = _BLACKHOLE_DEVICE
        self._write_target = _BLACKHOLE_DEVICE
        self._module_ids = []
        self._torn_down = False

        return {
            "platform": "darwin",
            "device_name": _BLACKHOLE_DEVICE,
            "sample_rate": 48000,
            "channels": 2,
            "module_ids": [],
            "write_target": _BLACKHOLE_DEVICE,
        }

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_module_id(stdout: str) -> int:
        """pactl load-module prints the new module ID to stdout."""
        text = (stdout or "").strip()
        if not text:
            raise RuntimeError("pactl load-module returned empty stdout")
        # Take the last whitespace-separated token on the first non-empty line.
        first = text.splitlines()[0].strip()
        token = first.split()[-1]
        try:
            return int(token)
        except ValueError as exc:
            raise RuntimeError(
                f"could not parse pactl module id from: {stdout!r}"
            ) from exc


def chrome_fake_audio_flags(bridge_info: dict) -> list[str]:
    """Return Chrome flags for using the fake audio input.

    The PulseAudio source is selected via the ``PULSE_SOURCE`` env var,
    which callers must set in Chrome's environment before launch:

        env["PULSE_SOURCE"] = bridge_info["device_name"]

    On macOS the caller must ensure the system default audio input is
    set to the returned BlackHole device (we do not flip that switch).
    """
    system = platform.system()
    if system == "Linux":
        # Chromium on Linux picks up the PulseAudio source selected via
        # PULSE_SOURCE env var; the fake-ui flag skips the permission
        # prompt so the bot can pick "use my mic" without user input.
        return ["--use-fake-ui-for-media-stream"]
    if system == "Darwin":
        return ["--use-fake-ui-for-media-stream"]
    if system == "Windows":
        raise RuntimeError("windows not supported in v2")
    raise RuntimeError(f"unsupported platform: {system}")
