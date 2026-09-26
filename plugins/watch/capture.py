"""Per-OS screen+audio capture arguments for ffmpeg.

Screen capture is the one part of this plugin that is genuinely different on
every host, and the differences are not stylistic — they are three unrelated
capture APIs with three device-naming schemes:

  * Windows — ``gdigrab`` for the desktop; audio needs a ``dshow`` device whose
    NAME the user must supply, because Windows has no default loopback device
    and enumerating one is a per-machine answer.
  * macOS — ``avfoundation``, addressed by numeric INDEX, not name. Screen
    indices come after camera indices, so the screen is rarely 0 and the number
    shifts when a webcam is plugged in. System audio needs a loopback device
    (BlackHole / Loopback) — macOS has no built-in one.
  * Linux — ``x11grab`` against ``$DISPLAY``, audio via PulseAudio. Wayland
    cannot be captured this way at all, and says so rather than recording a
    black rectangle.

The device-naming problem is why this module returns a *plan* — argv plus any
notes about what the user still has to decide — instead of just spawning. The
notes are the difference between "recording failed" and "your Mac needs a
loopback device to hear itself; here is the one line that installs one."

Pure functions over an explicit platform argument: the whole point is to be
able to check Windows argv from Linux CI, so nothing here reads
``sys.platform`` on its own behalf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

WINDOWS = "win32"
MACOS = "darwin"
LINUX = "linux"


@dataclass(frozen=True)
class CapturePlan:
    """ffmpeg input arguments plus what the user still needs to know.

    Attributes:
        args: Input-side argv (everything up to, not including, the output
            file). Encoder and output flags are the recorder's business.
        notes: Advisory lines worth printing — a missing loopback device, a
            guessed screen index. Never fatal.
        blocked: Why capture cannot work here at all, or ``None``. Wayland is
            the real case: no amount of argv fixes it.
        audio: Whether the plan actually captures audio, after resolving what
            the host can do. A caller must not promise the model an audio track
            this is False for.
    """

    args: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    blocked: Optional[str] = None
    audio: bool = False


def _framerate_args(fps: float) -> list[str]:
    return ["-framerate", f"{fps:g}"]


def windows_plan(
    fps: float,
    audio_device: Optional[str],
    region: Optional[tuple[int, int, int, int]] = None,
) -> CapturePlan:
    """gdigrab desktop capture, optionally with a named dshow audio device.

    ``audio_device`` is the exact string ``ffmpeg -list_devices true -f dshow -i
    dummy`` prints, e.g. ``"Stereo Mix (Realtek Audio)"``. There is no
    guessable default: Stereo Mix is disabled by default on most modern Windows
    installs, and the alternative is a virtual cable the user chose to install.
    Asking is correct; picking one and recording silence is not.
    """
    args = ["-f", "gdigrab", *_framerate_args(fps)]

    if region is not None:
        x, y, width, height = region
        args += [
            "-offset_x", str(x),
            "-offset_y", str(y),
            "-video_size", f"{width}x{height}",
        ]

    args += ["-i", "desktop"]

    notes: list[str] = []
    has_audio = bool(audio_device)
    if has_audio:
        args += ["-f", "dshow", "-i", f"audio={audio_device}"]
    else:
        notes.append(
            "No audio device set — recording video only. List candidates with: "
            "ffmpeg -list_devices true -f dshow -i dummy   "
            "then pass --audio-device \"<exact name>\"."
        )

    return CapturePlan(args=args, notes=notes, audio=has_audio)


def macos_plan(
    fps: float,
    screen_index: int,
    audio_index: Optional[int],
) -> CapturePlan:
    """avfoundation capture, addressed by device index.

    Video and audio are ONE input here (``-i "<screen>:<audio>"``), not two,
    which is the trap that makes a Linux/Windows-shaped implementation silently
    drop the audio track on macOS.
    """
    spec = f"{screen_index}:{audio_index}" if audio_index is not None else f"{screen_index}:none"
    args = ["-f", "avfoundation", *_framerate_args(fps), "-i", spec]

    notes = [
        "Screen device index is per-machine and shifts when cameras are "
        "attached — verify with: ffmpeg -f avfoundation -list_devices true -i \"\"",
    ]
    if audio_index is None:
        notes.append(
            "No audio index set — recording video only. macOS cannot capture "
            "system audio without a loopback device (e.g. brew install blackhole-2ch)."
        )

    return CapturePlan(args=args, notes=notes, audio=audio_index is not None)


def linux_plan(
    fps: float,
    display: Optional[str],
    wayland_display: Optional[str],
    pulse_source: Optional[str],
    region: Optional[tuple[int, int, int, int]] = None,
) -> CapturePlan:
    """x11grab capture, with the Wayland refusal made explicit.

    A Wayland session with no ``DISPLAY`` has no XWayland to grab and no
    unprivileged screen-capture path at all — the compositor withholds it by
    design. x11grab against such a session produces a black video, which is a
    far worse outcome than refusing, so it refuses. A session with BOTH set is
    Wayland running XWayland, where x11grab works and is used.
    """
    if wayland_display and not display:
        return CapturePlan(
            blocked=(
                "Wayland session without XWayland: ffmpeg cannot capture the "
                "screen here. Use a portal-based recorder (wf-recorder, "
                "GNOME/KDE built-in) and pass the file to `hermes watch review`."
            )
        )

    if not display:
        return CapturePlan(
            blocked="No DISPLAY set — nothing to capture (headless session?)."
        )

    size_args: list[str] = []
    grab = display
    if region is not None:
        x, y, width, height = region
        size_args = ["-video_size", f"{width}x{height}"]
        grab = f"{display}+{x},{y}"

    args = ["-f", "x11grab", *_framerate_args(fps), *size_args, "-i", grab]

    notes: list[str] = []
    has_audio = bool(pulse_source)
    if has_audio:
        args += ["-f", "pulse", "-i", pulse_source]
    else:
        notes.append(
            "No audio source set — recording video only. List sources with: "
            "pactl list short sources   then pass --audio-device <name> "
            "(a '.monitor' source captures what your speakers play)."
        )

    return CapturePlan(args=args, notes=notes, audio=has_audio)


def capture_plan(
    platform: str,
    *,
    fps: float = 15.0,
    audio_device: Optional[str] = None,
    screen_index: int = 1,
    audio_index: Optional[int] = None,
    display: Optional[str] = None,
    wayland_display: Optional[str] = None,
    region: Optional[tuple[int, int, int, int]] = None,
) -> CapturePlan:
    """Dispatch to the host's capture backend.

    ``platform`` is passed in rather than sniffed so the argv for every host is
    checkable from any host.
    """
    if platform == WINDOWS:
        return windows_plan(fps, audio_device, region)
    if platform == MACOS:
        audio = audio_index
        if audio is None and audio_device is not None:
            # avfoundation indexes devices; a name here is a category error
            # worth naming rather than silently ignoring.
            try:
                audio = int(audio_device)
            except ValueError:
                return CapturePlan(
                    blocked=(
                        f"macOS addresses audio devices by INDEX, not name "
                        f"(got {audio_device!r}). Find it with: "
                        'ffmpeg -f avfoundation -list_devices true -i ""'
                    )
                )
        return macos_plan(fps, screen_index, audio)
    if platform.startswith(LINUX):
        return linux_plan(fps, display, wayland_display, audio_device, region)

    return CapturePlan(blocked=f"Screen capture is not supported on {platform!r}.")


def encoder_args(crf: int = 23, preset: str = "veryfast", audio: bool = True) -> list[str]:
    """Output-side encoder args for the LIVE capture pass.

    Deliberately cheap and lossy-ish: this file is an intermediate that
    `prepare` will re-encode anyway, and the live pass is competing with
    whatever the user is actually doing (a game, a DAW) for CPU. Quality spent
    here is quality thrown away one step later.

    Hardware encoders are not used automatically. nvenc/qsv/videotoolbox each
    fail differently when absent, and a capture that dies three seconds in
    because the flag was wrong is worse than one that costs a few percent CPU.

    ``audio=False`` emits ``-an``: a plan with no audio input must not carry
    audio codec flags, which ffmpeg rejects when there is no stream to apply
    them to.
    """
    args = [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
    ]
    if audio:
        args += ["-c:a", "aac", "-b:a", "128k"]
    else:
        args += ["-an"]
    return args
