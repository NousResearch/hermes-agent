"""macOS candidate promotion must not turn a signing failure into success."""
from pathlib import Path

import pytest

from hermes_cli import main_desktop as producer


pytestmark = pytest.mark.macos_only


@pytest.mark.parametrize("fixup_ok", [False, True])
def test_unverified_signature_never_replaces_release(tmp_path, monkeypatch, fixup_ok):
    desktop = tmp_path / "apps/desktop"
    old = desktop / "release/mac-arm64/Hermes.app/Contents/MacOS/Hermes"
    staged = desktop / ".staging-test/mac-arm64/Hermes.app/Contents/MacOS/Hermes"
    old.parent.mkdir(parents=True)
    staged.parent.mkdir(parents=True)
    old.write_bytes(b"previous working package")
    staged.write_bytes(b"unsigned invalid executable")
    # The signing operation is external; promotion and release writes stay real.
    monkeypatch.setattr(producer, "_desktop_macos_relaunchable_fixup", lambda *a, **k: fixup_ok)
    with pytest.raises(SystemExit) as error:
        producer._promote_staged_desktop_app(desktop, desktop / ".staging-test")
    assert error.value.code != 0
    assert old.read_bytes() == b"previous working package"


def test_configured_identity_failure_does_not_fall_back_to_adhoc(tmp_path, monkeypatch):
    app = tmp_path / "release/mac-arm64/Hermes.app"
    exe = app / "Contents/MacOS/Hermes"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"unsigned")
    monkeypatch.setattr(producer, "_desktop_macos_local_signing_identity", lambda: "Configured Identity")

    def fail_sign(*args, **kwargs):
        raise RuntimeError("signer unavailable")

    monkeypatch.setattr(producer, "_desktop_macos_local_codesign", fail_sign)
    # If called, the legacy signer would report success: observable return must remain false.
    monkeypatch.setattr(producer, "_macos_legacy_adhoc_resign", lambda *a: True)
    assert producer._desktop_macos_relaunchable_fixup(tmp_path, publisher_signing_configured=False) is False
