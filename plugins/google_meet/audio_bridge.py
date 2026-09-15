"""Virtual audio bridge for feeding generated speech into Chrome's mic.

Linux: pactl creates a null-sink plus a virtual source on the sink's monitor; callers set
``PULSE_SOURCE=<source_name>`` in Chrome's env. macOS: only verifies BlackHole 2ch is installed
(the default-input switch is left to the user). Windows: unsupported.
"""

from __future__ import annotations

import contextlib
import platform
import subprocess
from typing import Optional


_BLACKHOLE_DEVICE = "BlackHole 2ch"


def _pactl(*args: str, check: bool) -> subprocess.CompletedProcess:
    return subprocess.run(["pactl", *args], check=check, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', stdin=subprocess.DEVNULL)


class AudioBridge:
    """Virtual audio device for Chrome fake-mic input: ``setup()`` before launch, ``teardown()`` after."""

    def __init__(self, name_prefix: str = "hermes_meet") -> None:
        self._name_prefix = name_prefix
        self._platform: Optional[str] = None
        self._device_name: Optional[str] = None
        self._write_target: Optional[str] = None
        self._module_ids: list[int] = []

    def _ready(self, value: Optional[str]) -> str:
        if not value:
            raise RuntimeError("AudioBridge not set up yet")
        return value

    device_name = property(lambda self: self._ready(self._device_name))
    write_target = property(lambda self: self._ready(self._write_target))

    def setup(self) -> dict:
        """Provision the device; raises RuntimeError on unsupported platforms or missing tools."""
        system = platform.system()
        impl = {"Linux": self._setup_linux, "Darwin": self._setup_darwin}.get(system)
        if impl is None:
            raise RuntimeError("windows not supported in v2" if system == "Windows"
                               else f"unsupported platform: {system}")
        return impl()

    def teardown(self) -> None:
        """Release the virtual audio device. Idempotent; never raises."""
        for mod_id in reversed(self._module_ids):  # linux only; virtual-source before null-sink
            with contextlib.suppress(Exception):
                _pactl("unload-module", str(mod_id), check=False)
        self._module_ids = []

    def _finish(self, tag: str, device: str, write_target: str, module_ids: list[int]) -> dict:
        self._platform = tag
        self._device_name = device
        self._write_target = write_target
        self._module_ids = module_ids
        return {"platform": tag, "device_name": device, "sample_rate": 48000, "channels": 2,
                "module_ids": list(module_ids), "write_target": write_target}

    def _setup_linux(self) -> dict:
        sink_name = f"{self._name_prefix}_sink"
        src_name = f"{self._name_prefix}_src"
        try:
            sink_out = _pactl(
                "load-module", "module-null-sink", f"sink_name={sink_name}",
                "sink_properties=device.description=HermesMeetSink", check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("pactl not found — install PulseAudio/pipewire-pulse") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"pactl load-module null-sink failed: {exc.stderr or exc}") from exc
        sink_mod_id = self._parse_module_id(sink_out.stdout)
        try:
            src_out = _pactl(
                "load-module", "module-virtual-source", f"source_name={src_name}",
                f"master={sink_name}.monitor", check=True)
        except subprocess.CalledProcessError as exc:
            # Roll back the null-sink we just created so we don't leak it.
            _pactl("unload-module", str(sink_mod_id), check=False)
            raise RuntimeError(f"pactl load-module virtual-source failed: {exc.stderr or exc}") from exc
        return self._finish("linux", src_name, sink_name, [sink_mod_id, self._parse_module_id(src_out.stdout)])

    def _setup_darwin(self) -> dict:
        try:
            out = subprocess.check_output(["system_profiler", "SPAudioDataType"], text=True, encoding='utf-8',
                                          errors='replace', stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise RuntimeError("system_profiler not found (macOS-only command)") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"system_profiler failed: {exc.output}") from exc
        if "BlackHole" not in out:
            raise RuntimeError("BlackHole virtual audio device not installed. "
                               "Install via: brew install blackhole-2ch")
        return self._finish("darwin", _BLACKHOLE_DEVICE, _BLACKHOLE_DEVICE, [])

    @staticmethod
    def _parse_module_id(stdout: str) -> int:
        """pactl load-module prints the new module ID as the last token of its first line."""
        text = (stdout or "").strip()
        if not text:
            raise RuntimeError("pactl load-module returned empty stdout")
        token = text.splitlines()[0].strip().split()[-1]
        try:
            return int(token)
        except ValueError as exc:
            raise RuntimeError(f"could not parse pactl module id from: {stdout!r}") from exc


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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
# ---- END PLUGIN-COMPAT ----
