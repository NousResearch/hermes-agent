"""Run Hermes's own wake-word detector (tools/wake_word.py: the bundled "hey hermes"
openWakeWord model on the local microphone) for the native desktop.

``python -I wake_listener.py <hermes-agent root>``. stdout lines: READY, WAKE, or
UNAVAILABLE <reason>. stdin lines: PAUSE (release the mic for dictation), RESUME, STOP.
"""

import os
import sys


def main() -> int:
    wire = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.path.insert(0, sys.argv[1])
    import hermes_bootstrap  # noqa: F401  -- the launcher's own environment setup

    from tools import wake_word

    # Only detection runs here; the desktop does its own speech-to-text and voice. The
    # detector's packages install on first use, through Hermes's package manager.
    needs = wake_word.check_wake_word_requirements()
    if not (needs.get("deps_available") and needs.get("audio_available")):
        import pm
        wire.write("INSTALLING\n")
        try:
            pm.ensure_import("audio-io")
            pm.ensure_import(f"wake-{needs.get('provider') or 'openwakeword'}")
        except Exception as exc:
            wire.write(f"UNAVAILABLE Could not install the wake-word model: {exc}\n")
            return 3
        needs = wake_word.check_wake_word_requirements()
    if not (needs.get("deps_available") and needs.get("audio_available")):
        wire.write("UNAVAILABLE " + (needs.get("hint") or "wake-word dependencies are not installed") + "\n")
        return 3
    owner = object()
    try:
        wake_word.start_listening(lambda: wire.write("WAKE\n"), owner=owner)
    except Exception as exc:  # engine/model/microphone failure: report it, the desktop falls back
        wire.write(f"UNAVAILABLE {exc}\n")
        return 4
    wire.write("READY\n")
    for line in sys.stdin:
        command = line.strip()
        if command == "PAUSE":
            wake_word.pause_listening(owner=owner)
        elif command == "RESUME":
            wake_word.resume_listening(owner=owner)
        elif command == "STOP":
            break
    wake_word.stop_listening(owner=owner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
