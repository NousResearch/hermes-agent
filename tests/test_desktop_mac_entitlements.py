"""Pin macOS Info.plist privacy usage descriptions declared by the Desktop
electron-builder config (`apps/desktop/package.json -> build.mac.extendInfo`).

Each entry is a key/value pair that lands in the packaged Hermes.app's
Info.plist via electron-builder's `extendInfo` merge. Missing or mis-stated
keys cause macOS to either silently deny the related API or surface a
mysteriously-worded system permission prompt at runtime (TCC's
`kTCCServiceMediaLibrary`, `kTCCServiceAppleEvents`, etc.).

The Desktop renderer initializes Chromium's audio stack on user gesture
(completion chimes, TTS playback, voice mode). On macOS 26+, that init can
register the helper with the media subsystem and surface as a "Hermes wants
to access Music" prompt unless the Info.plist disclaims it explicitly. This
test pins every usage-description string the desktop currently relies on so
accidental drops break CI instead of breaking users.

Why this test exists
--------------------

The project has a recurring class of bug: a macOS privacy-sensitive API is
called at runtime, but the Info.plist doesn't declare the corresponding
`NS*UsageDescription` key, so the system prompt is either silent (with a
generic "denied" error to the agent) or worded in a way that confuses the
user ("Hermes wants to access Music" when Hermes never touches the Music
library). The closed-PR family (#59486 / its duplicates #59833, #59915,
#59950, #60013 for Contacts; #39854 for Calendar; #64582 for Reminders)
established that the right fix shape is: add the key + pin it in a test.
This file is the canonical test for that pattern at the Desktop layer.

When adding a new NS*UsageDescription key to `build.mac.extendInfo`, add a
matching row to EXPECTED_USAGE_DESCRIPTIONS below. The drift-protection
assertion at the bottom of this file will fail otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_JSON = REPO_ROOT / "apps" / "desktop" / "package.json"


def _load_extend_info() -> dict[str, str]:
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    return data["build"]["mac"]["extendInfo"]


@pytest.fixture(scope="module")
def extend_info() -> dict[str, str]:
    return _load_extend_info()


# Each entry: (Info.plist key, required substring, plain-language reason).
# Required-substring checks let future copy edits pass while still catching
# silent drops of the key itself.
EXPECTED_USAGE_DESCRIPTIONS: list[tuple[str, str, str]] = [
    (
        "NSMicrophoneUsageDescription",
        "microphone",
        "Microphone capture is required for voice input mode.",
    ),
    (
        "NSAudioCaptureUsageDescription",
        "audio",
        "Audio capture backs the voice conversation pipeline.",
    ),
    (
        "NSAppleMusicUsageDescription",
        "Music",
        "Disclaim MediaLibrary access so the system audio stack does not "
        "surface a misleading Apple Music permission prompt (kTCCServiceMediaLibrary) "
        "when the renderer initializes audio for completion chimes, TTS, or voice.",
    ),
]


@pytest.mark.parametrize(("key", "required_substring", "reason"), EXPECTED_USAGE_DESCRIPTIONS)
def test_required_privacy_usage_descriptions_are_declared(
    extend_info: dict[str, str], key: str, required_substring: str, reason: str
) -> None:
    """Each macOS privacy usage key Hermes relies on must be present in
    build.mac.extendInfo, with copy that names the protected resource.

    A missing key causes macOS to silently deny the underlying API or surface
    a generic system prompt with no usage description, which reads to users
    as the app misbehaving. Pin the keys so the next refactor can't drop one
    by accident.
    """
    value = extend_info.get(key)
    assert value is not None, (
        f"Info.plist privacy usage description `{key}` is missing from "
        f"apps/desktop/package.json build.mac.extendInfo. macOS will surface "
        f"a misleading system prompt or silently deny the related API.\n"
        f"Reason: {reason}"
    )
    assert required_substring.lower() in value.lower(), (
        f"`{key}` exists but does not mention '{required_substring}'. "
        f"Current value: {value!r}. Reason: {reason}"
    )


def test_extend_info_keys_have_no_trailing_whitespace(extend_info: dict[str, str]) -> None:
    """electron-builder merges extendInfo into Info.plist verbatim; trailing
    whitespace in a usage string renders as a system prompt that breaks off
    mid-sentence. Pin the hygiene."""
    for key, value in extend_info.items():
        assert value == value.strip(), (
            f"`{key}` in build.mac.extendInfo has leading/trailing whitespace: {value!r}"
        )
        # electron-builder writes strings as-is; newlines would render as
        # literal control chars in the macOS prompt.
        assert "\n" not in value and "\r" not in value, (
            f"`{key}` contains a newline; macOS will render it as a control "
            f"character in the system permission prompt."
        )


def test_extend_info_does_not_silently_drop_unexpected_keys(
    extend_info: dict[str, str],
) -> None:
    """If a future PR adds a new privacy-sensitive key, this test will start
    failing — forcing the author to update EXPECTED_USAGE_DESCRIPTIONS and
    document why the new key is needed. This is the safety net the closed PR
    family (#59486, #59915, #59950, #60013) established for the Contacts and
    Apple Events keys; the same shape applies here."""
    declared_keys = {key for key, _, _ in EXPECTED_USAGE_DESCRIPTIONS}
    # Non-privacy keys (CFBundleDisplayName etc.) are exempt — this test
    # only governs NS*UsageDescription entries.
    privacy_keys_in_plist = {
        key for key in extend_info if key.startswith("NS") and key.endswith("UsageDescription")
    }
    missing = privacy_keys_in_plist - declared_keys
    assert not missing, (
        f"extendInfo declares privacy usage keys {sorted(missing)} that this "
        f"test does not pin. Add them to EXPECTED_USAGE_DESCRIPTIONS with a "
        f"reason, or remove them from the build config."
    )
