"""Tests for ``_compute_voice_message_waveform``.

The Discord voice-message bubble renders a ``waveform`` (base64-encoded
uint8 bytearray, max 256 samples, at most one sample per 100ms) as a
loudness bar graph. We want real samples so speech shows up as speech
instead of a flat line.

These tests fake ``subprocess.run`` so they never touch ffmpeg on the
host — ffmpeg behavior is well-defined; what we care about is that the
bucketing, RMS, and dBFS mapping are correct.
"""

import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.Client = MagicMock
        discord_mod.File = MagicMock
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
        discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3, grey=4, secondary=5)
        discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
        discord_mod.Interaction = object
        discord_mod.Embed = MagicMock
        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        discord_mod.opus = SimpleNamespace(is_loaded=lambda: True)

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.discord import (  # noqa: E402
    _compute_voice_message_waveform,
    _probe_audio_duration_seconds,
    _VOICE_WAVEFORM_MAX_SAMPLES,
    _VOICE_WAVEFORM_SAMPLES_PER_SEC,
)


def _int16_bytes(samples):
    """Pack a list of int16 samples as little-endian bytes."""
    return b"".join(struct.pack("<h", int(s)) for s in samples)


def _mock_ffmpeg(pcm_bytes: bytes):
    """Patch subprocess.run to pretend ffmpeg decoded audio_path → pcm_bytes."""
    return patch(
        "gateway.platforms.discord.subprocess.run",
        return_value=SimpleNamespace(stdout=pcm_bytes, stderr=b"", returncode=0),
    )


def test_target_sample_count_caps_at_256(tmp_path):
    """Audio > 25.6 s must be capped at 256 samples per Discord spec."""
    # 60 seconds * 48000 hz = 2,880,000 samples; use a tiny stream, we only
    # care about len(waveform). The mocked ffmpeg returns 48,000 samples
    # (big enough to populate every bucket).
    pcm = _int16_bytes([0] * 48_000)
    audio_path = tmp_path / "long.ogg"
    audio_path.write_bytes(b"fake-ogg-header")

    with _mock_ffmpeg(pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=60.0)

    assert len(waveform) == _VOICE_WAVEFORM_MAX_SAMPLES


def test_target_sample_count_scales_with_duration(tmp_path):
    """Spec: one sample per 100ms → 3 s clip → 30 samples."""
    pcm = _int16_bytes([0] * 48_000)
    audio_path = tmp_path / "short.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=3.0)

    assert len(waveform) == 3 * _VOICE_WAVEFORM_SAMPLES_PER_SEC


def test_tiny_duration_still_yields_at_least_one_sample(tmp_path):
    """duration < 100ms must still produce at least one waveform sample."""
    pcm = _int16_bytes([0] * 4_800)
    audio_path = tmp_path / "tiny.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=0.05)

    assert len(waveform) >= 1


def test_silent_audio_maps_to_zero(tmp_path):
    """Pure silence (all int16 zeros) should map to the waveform floor (0)."""
    pcm = _int16_bytes([0] * 48_000)
    audio_path = tmp_path / "silent.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=1.0)

    # All samples should be at or near 0 (we use max(rms,1) to avoid log(0),
    # so the actual value is dbfs(1/32767) = ~-90 dBFS, clamped to 0).
    assert all(b == 0 for b in waveform), f"silent audio should map to zero, got {set(waveform)}"


def test_full_scale_audio_maps_near_max(tmp_path):
    """A full-scale square wave (±32767) should saturate toward 255."""
    sign = 1
    samples = []
    for _ in range(48_000):
        samples.append(sign * 32767)
        sign = -sign
    pcm = _int16_bytes(samples)
    audio_path = tmp_path / "loud.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=1.0)

    # At 0 dBFS the mapping is ((0 - (-60)) / 60) * 255 = 255, clamped.
    assert max(waveform) >= 250, f"full-scale audio should saturate high, got max={max(waveform)}"


def test_varying_loudness_produces_non_flat_waveform(tmp_path):
    """A clip with a quiet half and a loud half must NOT produce a flat line."""
    quiet = [100] * 24_000        # ~-50 dBFS
    loud = [16000, -16000] * 12_000  # ~-6 dBFS
    pcm = _int16_bytes(quiet + loud)
    audio_path = tmp_path / "varied.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=1.0)

    # First half should be dimmer than second half (loud).
    first_half_mean = sum(waveform[: len(waveform) // 2]) / max(1, len(waveform) // 2)
    second_half_mean = sum(waveform[len(waveform) // 2 :]) / max(1, len(waveform) - len(waveform) // 2)
    assert second_half_mean > first_half_mean + 20, (
        f"varied audio should produce a varied waveform; got "
        f"first_half_mean={first_half_mean:.1f}, second_half_mean={second_half_mean:.1f}"
    )
    # And definitely not the flat fallback.
    assert len(set(waveform)) > 1


def test_waveform_values_are_uint8(tmp_path):
    """Every byte of the waveform must be in [0, 255]."""
    pcm = _int16_bytes([12345, -12345] * 24_000)
    audio_path = tmp_path / "mid.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=1.0)

    assert all(0 <= b <= 255 for b in waveform)


def test_ffmpeg_missing_falls_back_to_flat(tmp_path):
    """If ffmpeg isn't on the host, we ship a flat waveform and keep going."""
    audio_path = tmp_path / "x.ogg"
    audio_path.write_bytes(b"fake")

    with patch(
        "gateway.platforms.discord.subprocess.run",
        side_effect=FileNotFoundError("ffmpeg"),
    ):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=3.0)

    assert len(waveform) == 3 * _VOICE_WAVEFORM_SAMPLES_PER_SEC
    assert all(b == 128 for b in waveform)


def test_ffmpeg_error_falls_back_to_flat(tmp_path):
    """A decode error (corrupt audio etc.) must not break voice-message sending."""
    audio_path = tmp_path / "corrupt.ogg"
    audio_path.write_bytes(b"not really ogg")

    with patch(
        "gateway.platforms.discord.subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=1, cmd=["ffmpeg"], stderr=b"invalid data"
        ),
    ):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=5.0)

    assert len(waveform) == 5 * _VOICE_WAVEFORM_SAMPLES_PER_SEC
    assert all(b == 128 for b in waveform)


def test_ffmpeg_timeout_falls_back_to_flat(tmp_path):
    """If ffmpeg hangs past the 30s cap, fall back — don't block sending."""
    audio_path = tmp_path / "slow.ogg"
    audio_path.write_bytes(b"fake")

    with patch(
        "gateway.platforms.discord.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=30),
    ):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=2.0)

    assert len(waveform) == 2 * _VOICE_WAVEFORM_SAMPLES_PER_SEC
    assert all(b == 128 for b in waveform)


def test_empty_ffmpeg_output_falls_back_to_flat(tmp_path):
    """ffmpeg returns 0 bytes on some edge cases — still need a valid waveform."""
    audio_path = tmp_path / "empty.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(b""):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=4.0)

    assert len(waveform) == 4 * _VOICE_WAVEFORM_SAMPLES_PER_SEC
    assert all(b == 128 for b in waveform)


def test_dbfs_mapping_hits_expected_bucket(tmp_path):
    """Worked numeric example.

    For a steady tone whose amplitude is int16=3277 (~one-tenth of full
    scale) the RMS is 3277 (DC tone), 20*log10(3277/32767) = -20 dBFS,
    which maps to ((-20 - (-60)) / 60) * 255 ≈ 170. Accept 165..175 to
    allow RMS rounding on the bucketed mean.
    """
    pcm = _int16_bytes([3277] * 48_000)
    audio_path = tmp_path / "neg20db.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=1.0)

    for byte in waveform:
        assert 165 <= byte <= 175, (
            f"-20 dBFS tone should map to ~170, got {byte} in {list(waveform)}"
        )


def test_short_pcm_shrinks_target_samples_instead_of_raising(tmp_path):
    """PCM shorter than the desired sample count must not raise ValueError.

    Regression guard for the reshape bug found in peer review — previously
    `pcm.size < target_samples` would end up in `reshape(target, window)`
    with insufficient data and throw ValueError out of the helper.
    """
    # 5 int16 samples, target_samples would have been 30 for 3s of audio.
    pcm = _int16_bytes([1000, 2000, 3000, 4000, 5000])
    audio_path = tmp_path / "tiny.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=3.0)

    # Helper shrinks target_samples to match available PCM (5 samples) rather
    # than raising. 5 samples is a perfectly valid waveform to Discord.
    assert 1 <= len(waveform) <= 5
    assert all(0 <= b <= 255 for b in waveform)


@pytest.mark.parametrize(
    "bad_duration", [float("nan"), float("inf"), float("-inf"), -5.0, 0.0]
)
def test_nonfinite_or_negative_duration_does_not_crash(tmp_path, bad_duration):
    """Corrupt mutagen probe can hand the helper NaN / inf / negative.

    The helper must coerce to a sane default rather than raise.
    """
    pcm = _int16_bytes([0] * 48_000)
    audio_path = tmp_path / "weird.ogg"
    audio_path.write_bytes(b"fake")

    with _mock_ffmpeg(pcm):
        # Would raise ValueError/OverflowError from int(round(...)) without the guard.
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=bad_duration)

    assert 1 <= len(waveform) <= _VOICE_WAVEFORM_MAX_SAMPLES


def test_very_long_duration_uses_flat_fallback(tmp_path):
    """Past ~10 minutes, buffering full PCM in memory isn't worth a bar graph.

    Helper should return a flat 128-waveform of target size without spawning
    ffmpeg (verified via the mock's call_count).
    """
    audio_path = tmp_path / "hour.ogg"
    audio_path.write_bytes(b"fake")

    with patch("gateway.platforms.discord.subprocess.run") as mock_run:
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=3600.0)

    assert mock_run.call_count == 0, "ffmpeg must not run for oversized clips"
    assert len(waveform) == _VOICE_WAVEFORM_MAX_SAMPLES
    assert all(b == 128 for b in waveform)


def test_ffmpeg_argv_uses_file_prefix_against_leading_dash(tmp_path):
    """Defense against a malicious filename starting with '-'.

    The helper passes the path as ``file:<path>`` so ffmpeg treats it as an
    input URL even when it begins with a dash; otherwise ffmpeg would parse
    e.g. ``-filter:...`` as a CLI flag. The filesystem never produces such
    names in practice, but this is defense in depth.
    """
    pcm = _int16_bytes([1000] * 48_000)
    # Realistic audio_path would never be user-controlled, but make sure
    # the argv construction is insulated regardless.
    audio_path = tmp_path / "-evil.ogg"
    audio_path.write_bytes(b"fake")

    captured_argv = {}

    def _capture(*args, **kwargs):
        captured_argv["argv"] = args[0] if args else kwargs.get("args")
        return SimpleNamespace(stdout=pcm, stderr=b"", returncode=0)

    with patch("gateway.platforms.discord.subprocess.run", side_effect=_capture):
        _compute_voice_message_waveform(str(audio_path), duration_secs=1.0)

    argv = captured_argv["argv"]
    assert argv is not None
    # Find the `-i` argument and verify the following arg starts with "file:".
    i_index = argv.index("-i")
    assert argv[i_index + 1].startswith("file:"), (
        f"audio_path must be prefixed with file: to avoid dash-flag injection; argv={argv}"
    )
    assert "-nostdin" in argv, "ffmpeg must be run with -nostdin so it never hangs on piped input"


def test_post_decode_failure_falls_back_instead_of_raising(tmp_path):
    """Numeric path guard: if anything in the frombuffer/reshape/log10 chain
    raises unexpectedly, the helper must fall back rather than propagate.
    """
    audio_path = tmp_path / "ok.ogg"
    audio_path.write_bytes(b"fake")

    # Odd byte count → np.frombuffer raises ValueError: buffer size must be
    # a multiple of element size (int16=2 bytes).
    odd_pcm = b"\x00\x01\x02"

    with _mock_ffmpeg(odd_pcm):
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=2.0)

    assert len(waveform) == 2 * _VOICE_WAVEFORM_SAMPLES_PER_SEC
    assert all(b == 128 for b in waveform)


@pytest.mark.parametrize(
    "env_value, expect_ffmpeg_run, expect_flat",
    [
        ("",       True,  False),  # unset → default on
        ("true",   True,  False),
        ("1",      True,  False),
        ("false",  False, True),   # explicit opt-out
        ("0",      False, True),
        ("no",     False, True),
        ("off",    False, True),
        ("  False ", False, True), # whitespace / case tolerated
    ],
)
def test_env_toggle_controls_real_waveform(tmp_path, monkeypatch, env_value, expect_ffmpeg_run, expect_flat):
    """Operators can opt out via DISCORD_VOICE_MESSAGE_WAVEFORM.

    When disabled, the helper must not invoke ffmpeg at all (the whole
    point of the toggle is to save cycles). When enabled or unset, the
    real pipeline runs.
    """
    if env_value == "":
        monkeypatch.delenv("DISCORD_VOICE_MESSAGE_WAVEFORM", raising=False)
    else:
        monkeypatch.setenv("DISCORD_VOICE_MESSAGE_WAVEFORM", env_value)

    pcm = _int16_bytes([3277] * 48_000)  # ~-20 dBFS tone
    audio_path = tmp_path / "toggle.ogg"
    audio_path.write_bytes(b"fake")

    with patch(
        "gateway.platforms.discord.subprocess.run",
        return_value=SimpleNamespace(stdout=pcm, stderr=b"", returncode=0),
    ) as mock_run:
        waveform = _compute_voice_message_waveform(str(audio_path), duration_secs=1.0)

    if expect_ffmpeg_run:
        assert mock_run.call_count == 1
        # Non-flat waveform from the -20 dBFS tone (~170 per earlier test).
        assert len(set(waveform)) == 1 and 160 <= waveform[0] <= 180
    else:
        assert mock_run.call_count == 0, "toggle must prevent ffmpeg subprocess"
        assert all(b == 128 for b in waveform) is expect_flat


class TestProbeAudioDuration:
    """``_probe_audio_duration_seconds`` — ffprobe wrapper used to keep the
    voice-message ``duration_secs`` in sync with what Discord's backend
    will eventually derive. Wrong values produce the "1:04 → 0:14"
    UI flicker that this replaces.
    """

    def test_parses_float_stdout(self, tmp_path):
        audio_path = tmp_path / "x.ogg"
        audio_path.write_bytes(b"fake")
        with patch(
            "gateway.platforms.discord.subprocess.run",
            return_value=SimpleNamespace(stdout="14.237\n", stderr="", returncode=0),
        ):
            assert _probe_audio_duration_seconds(str(audio_path)) == pytest.approx(14.237)

    def test_uses_file_prefix_argv(self, tmp_path):
        audio_path = tmp_path / "-evil.ogg"
        audio_path.write_bytes(b"fake")
        captured = {}

        def _cap(*args, **kwargs):
            captured["argv"] = args[0] if args else kwargs.get("args")
            return SimpleNamespace(stdout="1.5", stderr="", returncode=0)

        with patch("gateway.platforms.discord.subprocess.run", side_effect=_cap):
            _probe_audio_duration_seconds(str(audio_path))

        argv = captured["argv"]
        assert argv[0] == "ffprobe"
        i_idx = argv.index("-i")
        assert argv[i_idx + 1].startswith("file:"), (
            f"ffprobe input must be prefixed file: to block dash-flag injection; argv={argv}"
        )

    @pytest.mark.parametrize("stdout", ["", "not-a-number", "N/A"])
    def test_non_numeric_stdout_returns_none(self, tmp_path, stdout):
        audio_path = tmp_path / "x.ogg"
        audio_path.write_bytes(b"fake")
        with patch(
            "gateway.platforms.discord.subprocess.run",
            return_value=SimpleNamespace(stdout=stdout, stderr="", returncode=0),
        ):
            assert _probe_audio_duration_seconds(str(audio_path)) is None

    @pytest.mark.parametrize("value", ["0", "-3", "inf", "nan"])
    def test_invalid_durations_return_none(self, tmp_path, value):
        audio_path = tmp_path / "x.ogg"
        audio_path.write_bytes(b"fake")
        with patch(
            "gateway.platforms.discord.subprocess.run",
            return_value=SimpleNamespace(stdout=value, stderr="", returncode=0),
        ):
            assert _probe_audio_duration_seconds(str(audio_path)) is None

    def test_ffprobe_missing_returns_none(self, tmp_path):
        audio_path = tmp_path / "x.ogg"
        audio_path.write_bytes(b"fake")
        with patch(
            "gateway.platforms.discord.subprocess.run",
            side_effect=FileNotFoundError("ffprobe"),
        ):
            assert _probe_audio_duration_seconds(str(audio_path)) is None

    def test_ffprobe_timeout_returns_none(self, tmp_path):
        audio_path = tmp_path / "x.ogg"
        audio_path.write_bytes(b"fake")
        with patch(
            "gateway.platforms.discord.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=10),
        ):
            assert _probe_audio_duration_seconds(str(audio_path)) is None

    def test_ffprobe_nonzero_exit_returns_none(self, tmp_path):
        audio_path = tmp_path / "x.ogg"
        audio_path.write_bytes(b"fake")
        with patch(
            "gateway.platforms.discord.subprocess.run",
            side_effect=subprocess.CalledProcessError(returncode=1, cmd=["ffprobe"], stderr="bad file"),
        ):
            assert _probe_audio_duration_seconds(str(audio_path)) is None


def _make_send_voice_harness(tmp_path):
    """Shared scaffolding for the two end-to-end send_voice duration tests.

    Returns ``(adapter, audio_path, captured_payload)``. ``captured_payload``
    is populated with whatever JSON body send_voice posts to the Discord
    REST route — we inspect ``duration_secs`` on it.
    """
    from unittest.mock import AsyncMock
    from gateway.config import PlatformConfig
    from gateway.platforms.discord import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="t"))
    audio_path = tmp_path / "reply.ogg"
    audio_path.write_bytes(b"fake-ogg")

    channel = MagicMock()
    channel.id = 555
    channel.send = AsyncMock(return_value=SimpleNamespace(id=1234))
    captured_payload = {}

    async def _fake_http_request(route, form=None, **kwargs):
        import json as _json
        for part in form or []:
            if part.get("name") == "payload_json":
                captured_payload.update(_json.loads(part["value"]))
        return {"id": "9001"}

    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(return_value=channel),
        http=SimpleNamespace(request=_fake_http_request),
    )
    return adapter, audio_path, captured_payload


@pytest.mark.asyncio
async def test_send_voice_prefers_ffprobe_over_mutagen_for_duration(tmp_path, monkeypatch):
    """The voice-message ``duration_secs`` must come from ffprobe when
    available, so Discord's UI value matches what its backend will derive
    — no more "1:04 → 0:14" flicker.

    We fake ffprobe's stdout directly by patching ``subprocess.run`` (same
    approach the ffmpeg-side tests use). Mocking the real subprocess is
    more robust across xdist workers than setattr-ing the helper.
    """
    adapter, audio_path, captured_payload = _make_send_voice_harness(tmp_path)
    # Skip waveform compute so the test only exercises duration probing.
    monkeypatch.setenv("DISCORD_VOICE_MESSAGE_WAVEFORM", "false")

    def _fake_subprocess_run(argv, *args, **kwargs):
        # Both ffprobe and ffmpeg land here — route by argv[0].
        if argv and argv[0] == "ffprobe":
            return SimpleNamespace(stdout="14.2\n", stderr="", returncode=0)
        # Waveform is disabled so ffmpeg shouldn't run, but be safe.
        return SimpleNamespace(stdout=b"", stderr=b"", returncode=0)

    with patch("gateway.platforms.discord.subprocess.run", side_effect=_fake_subprocess_run):
        result = await adapter.send_voice("555", str(audio_path))

    assert result.success is True
    assert captured_payload.get("attachments"), f"no attachments in payload: {captured_payload}"
    assert captured_payload["attachments"][0]["duration_secs"] == pytest.approx(14.2)


@pytest.mark.asyncio
async def test_send_voice_falls_back_when_ffprobe_unavailable(tmp_path, monkeypatch):
    """When ffprobe is missing, the legacy mutagen / byte-rate path must
    still pick up — a duration-probing gap can't break voice-message send.
    """
    adapter, audio_path, captured_payload = _make_send_voice_harness(tmp_path)
    monkeypatch.setenv("DISCORD_VOICE_MESSAGE_WAVEFORM", "false")

    # ffprobe not found → helper returns None → send_voice falls through.
    with patch(
        "gateway.platforms.discord.subprocess.run",
        side_effect=FileNotFoundError("ffprobe"),
    ):
        result = await adapter.send_voice("555", str(audio_path))

    assert result.success is True
    # With both ffprobe and the OGG-opus mutagen probe unavailable, the
    # byte-rate fallback runs on the 8-byte "fake-ogg" file:
    #   max(1.0, 8 / 2000) == 1.0
    assert captured_payload["attachments"][0]["duration_secs"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_send_voice_offloads_waveform_computation_to_thread(tmp_path):
    """Regression guard: the ffmpeg + numpy work must run in a worker
    thread so a slow decode (or the 30s timeout) can't block the Discord
    event loop and stall unrelated bot activity.

    Approach — make the helper raise if it's called on the event-loop
    thread, then invoke ``send_voice`` and assert the voice-message send
    still completed. If the helper were being called synchronously on
    the loop thread, it'd raise and bubble up.

    Matches the existing async-offload pattern used for ``pcm_to_wav``
    and STT elsewhere in this adapter.
    """
    import asyncio
    import threading
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from gateway.platforms.discord import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="t"))

    # Minimal audio file on disk so os.path.exists passes.
    audio_path = tmp_path / "reply.ogg"
    audio_path.write_bytes(b"fake-ogg")

    # Mock Discord HTTP layer enough to return a synthetic message id.
    channel = MagicMock()
    channel.id = 555
    channel.send = AsyncMock(return_value=SimpleNamespace(id=1234))
    http_mock = MagicMock()
    http_mock.request = AsyncMock(return_value={"id": "9001"})
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(return_value=channel),
        http=http_mock,
    )

    loop_thread_id = threading.get_ident()
    offloaded = {"hit": False}

    def asserting_helper(_audio_path, _duration_secs):
        # Will run either on the loop thread (bad — regression) or in a
        # worker thread (good — what the fix guarantees).
        if threading.get_ident() == loop_thread_id:
            raise AssertionError(
                "_compute_voice_message_waveform must be offloaded via "
                "asyncio.to_thread — it's running on the event loop thread"
            )
        offloaded["hit"] = True
        return bytes([128] * 10)  # any valid waveform

    with patch(
        "gateway.platforms.discord._compute_voice_message_waveform",
        side_effect=asserting_helper,
    ):
        result = await adapter.send_voice("555", str(audio_path))

    assert offloaded["hit"], "the waveform helper must be invoked at least once"
    assert result.success is True
