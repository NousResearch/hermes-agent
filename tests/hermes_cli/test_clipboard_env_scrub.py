"""Clipboard + thumbnail OS helpers must not inherit Hermes credentials.

Same rule as the voice TTS/STT scrub (#56332 / #70342), the voice-mode
playback scrub and the ffmpeg/ffprobe media helpers: a third-party binary
Hermes shells out to has no business seeing provider API keys, bot tokens or
``SUDO_PASSWORD``.

``pngpaste`` / ``osascript`` / ``xclip`` / ``wl-paste`` / PowerShell run on the
image-paste path, and ImageMagick ``convert`` runs on the attachment path.
"""

from unittest.mock import MagicMock, patch


_SECRETS = {
    "OPENAI_API_KEY": "sk-test",
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "TELEGRAM_BOT_TOKEN": "secret-token",
    "SUDO_PASSWORD": "hunter2",
}


def _seed_secrets(monkeypatch):
    for key, value in _SECRETS.items():
        monkeypatch.setenv(key, value)


def _assert_scrubbed(env, what):
    assert env is not None, f"{what} inherited the full process environment"
    for key in _SECRETS:
        assert key not in env, f"{key} leaked into {what}"


def test_macos_clipboard_probe_scrubs_credentials(monkeypatch):
    """`osascript` evaluates AppleScript — it must not carry credentials."""
    _seed_secrets(monkeypatch)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        return result

    from hermes_cli import clipboard

    with patch.object(clipboard.subprocess, "run", side_effect=fake_run):
        clipboard._macos_has_image()

    _assert_scrubbed(captured.get("env"), "osascript")


def test_pngpaste_scrubs_credentials(tmp_path, monkeypatch):
    _seed_secrets(monkeypatch)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        result = MagicMock()
        result.returncode = 1
        return result

    from hermes_cli import clipboard

    with patch.object(clipboard.subprocess, "run", side_effect=fake_run):
        clipboard._macos_pngpaste(tmp_path / "clip.png")

    _assert_scrubbed(captured.get("env"), "pngpaste")


def test_powershell_clipboard_scrubs_credentials(monkeypatch):
    """The Windows clipboard path shells out to PowerShell."""
    _seed_secrets(monkeypatch)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        return result

    from hermes_cli import clipboard

    with patch.object(clipboard.subprocess, "run", side_effect=fake_run):
        clipboard._run_powershell("powershell.exe", "echo ok", 5)

    _assert_scrubbed(captured.get("env"), "PowerShell")


def test_simplex_thumbnail_convert_scrubs_credentials(monkeypatch):
    """ImageMagick `convert` on the attachment thumbnail path.

    ``subprocess`` is imported inside the helper, so drive the real spawn
    through the module-level scrub helper the call sites now pass.
    """
    _seed_secrets(monkeypatch)

    from plugins.platforms.simplex import adapter as sx

    _assert_scrubbed(sx._scrubbed_media_env(), "ImageMagick convert")
