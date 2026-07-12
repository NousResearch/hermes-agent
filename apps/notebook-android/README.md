# Hermes Notebook for Android and BOOX

Native, offline-safe stylus client for the Hermes Notebook platform.

## Product guarantees

- Pen samples preserve pressure, tilt, orientation, historical points, and eraser input.
- Palm-cancelled gestures never become committed strokes.
- Every completed edit is atomically persisted before network delivery matters.
- Handwriting recognition runs on-device through ML Kit after its language model is downloaded.
- Hermes connections require HTTPS and use `X-Notebook-Token`.
- Port 8793 remains localhost-only; configure the app with the secure public notebook bridge URL.
- BOOX is detected from Android manufacturer/brand metadata and reported as `platform: boox`.

## Build

Open this directory in Android Studio or run `gradlew.bat assembleDebug` after installing JDK 17 and Android SDK 36.

Install the debug APK from `app/build/outputs/apk/debug/app-debug.apk` with ADB or BOOX's APK installer.

Physical-device testers should follow [BOOX-TESTING.md](BOOX-TESTING.md).

## Current milestone

This first native milestone includes pen capture, eraser/palm handling, atomic offline recovery,
HTTPS configuration, notebook session continuity, and synchronous Hermes replies. Before public
distribution it still needs device-keystore token storage, queued retry/outbox behavior, stroke
rendering benchmarks on physical BOOX hardware, accessibility review, signed release builds, and
an updater/distribution policy.
