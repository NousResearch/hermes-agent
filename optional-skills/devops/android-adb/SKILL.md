---
name: android-adb
description: Android device automation via ADB for testing and debugging.
version: 1.0.0
author: amit-nayar
license: MIT
metadata:
  hermes:
    tags: [android, adb, mobile, testing, automation, devops]
    category: devops
    related_skills: [docker-management, cli]
---

# Android ADB Skill

Automate Android physical devices and emulators via ADB for app installation, UI inspection, gesture inputs, navigation, and log debugging.

## When to Use

Use this skill when you need to:
- List, select, or manage connected Android physical devices or emulators.
- Capture screenshots or dump the UI hierarchy tree for inspection.
- Perform tap, swipe, scroll, text input, or key event interactions on an Android screen.
- Install, launch, clear, or uninstall APK packages.
- Stream or filter Android logcat logs during app debugging.

## Prerequisites

- Android SDK Platform-Tools (`adb`) installed and available on `PATH` or `ANDROID_HOME`.
- A connected Android physical device with USB Debugging enabled, or a running Android emulator (`adb devices`).

## How to Run

Use `terminal` to invoke `adb` commands or automation scripts:

### 1. Device Discovery & State
```bash
adb devices -l
adb shell getprop ro.build.version.release
```

### 2. Capture Screenshot & UI Dump
```bash
# Capture screenshot to host
adb exec-out screencap -p > /tmp/screen.png

# Dump UI hierarchy XML
adb shell uiautomator dump /sdcard/window_dump.xml
adb pull /sdcard/window_dump.xml /tmp/window_dump.xml
```

### 3. Interaction & Gestures
```bash
# Tap by screen coordinates (x, y)
adb shell input tap 500 1000

# Input text
adb shell input text "hello_world"

# Press Key Event (e.g. HOME=3, BACK=4, ENTER=66)
adb shell input keyevent 3
```

### 4. App Installation & Launch
```bash
# Install APK
adb install -r path/to/app.apk

# Launch App
adb shell am start -n com.example.app/.MainActivity

# Force Stop App
adb shell am force-stop com.example.app
```

### 5. Debugging & Logs
```bash
# Filter crash logcat output
adb logcat *:E | grep -i "com.example.app"
```

## Quick Reference

| Action | ADB Command |
|---|---|
| List devices | `adb devices` |
| Take screenshot | `adb exec-out screencap -p > screen.png` |
| Dump UI XML | `adb shell uiautomator dump && adb pull /sdcard/window_dump.xml` |
| Tap screen | `adb shell input tap <x> <y>` |
| Type text | `adb shell input text "<text>"` |
| Key event | `adb shell input keyevent <keycode>` |
| Install APK | `adb install -r <apk_path>` |
| View logs | `adb logcat` |

## Procedure

1. **Verify Connection**: Run `adb devices` using `terminal` to confirm at least one device or emulator is authorized.
2. **Inspect Screen**: Capture a screenshot or dump the UI tree to identify target element bounds/coordinates.
3. **Dispatch Interaction**: Send input events (`tap`, `text`, `keyevent`, `swipe`) as required by the task.
4. **Verify State**: Confirm the expected UI state change by re-dumping UI hierarchy or taking a new screenshot.
5. **Collect Diagnostics**: If an operation fails, run `adb logcat` to retrieve crash trace dumps.

## Pitfalls

- Ensure spaces in `input text` strings are escaped or replaced with `%s` (`adb shell input text "hello%sworld"`).
- Verify target device authorization status (`device` vs `unauthorized` or `offline`) before running batch actions.
- Coordinate tap locations depend on target screen resolution; dumping UI XML first yields accurate element bounding boxes.

## Verification

Run `adb devices` to verify connection and check that screenshot/log output files are generated as expected.
