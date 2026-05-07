## Windows Emulator Visual Debug Harness

This release improves local Android app debugging from a Windows laptop.

- Adds `scripts/windows-visual-control.ps1` for full desktop screenshots,
  focused window screenshots, host mouse movement/clicks, and keyboard or
  clipboard input.
- Gives Hermes debugging a fallback path for BlueStacks, Android Studio
  Emulator, and other visible emulator windows when ADB framebuffer screenshots
  are unavailable, blank, or not representative of the real host window.
- Keeps the existing ADB-based harnesses for Android UI dumps, taps, swipes,
  text input, key events, and wide emulator captures.
- Documents the host visual-control path in the Android Tasker/Shizuku
  capability map.
- Keeps Git/Git Credential Manager automation noninteractive on this Windows
  workstation so unattended pushes fail fast instead of opening GUI prompts.
