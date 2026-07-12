# Hermes Notebook for Android and BOOX

![Hermes Notebook on handwriting-first tablets](docs/hermes-notebook-hero.png)

Native, offline-safe stylus client for the Hermes Notebook platform.

Hermes Notebook turns stylus notes into normal Hermes conversations: capture
ink locally, recognize handwriting on-device, and send the transcription
through the authenticated Notebook gateway to the same sessions, memory, tools,
and agent personality used by other Hermes channels.

> [!IMPORTANT]
> This APK is a client, not a standalone Hermes installation. Sending pages to
> Hermes requires the Kindle/Notebook gateway adapter introduced by
> [NousResearch/hermes-agent PR #61687](https://github.com/NousResearch/hermes-agent/pull/61687)
> to be installed, enabled, configured, and running on the Hermes host. Without
> that PR's `plugins/platforms/kindle` adapter, the app can draw and save pages
> offline but cannot connect to Hermes.

## Required Hermes dependency

The complete system has two required halves:

```text
Android / BOOX APK
        -> private HTTPS endpoint
Kindle/Notebook gateway adapter from PR #61687
        -> Hermes Gateway, sessions, tools, and memory
```

The Android app does not include Hermes, the Gateway, or a public relay. The
Hermes operator must deploy the PR branch, enable `platforms/kindle`, configure
its token and allowed user, and provide the tester with a private HTTPS endpoint.
See [BOOX-TESTING.md](BOOX-TESTING.md#hermes-operator-setup) for the exact setup.

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

## Handwriting recognition and ambiguity

The Android client sends both the recognized text and device metadata to the
Notebook gateway. Recognition quality depends on the device, pen sampling, the
downloaded ML Kit language model, and the writing itself. For the most reliable
results:

- write dark ink on a plain background with visible gaps between words;
- avoid joining short acronyms into one continuous glyph;
- keep the tablet online for the first recognition so ML Kit can download its
  English model;
- verify names, dates, amounts, and commands before allowing consequential tool
  work; and
- preserve the offline page when recognition or delivery fails so it can be
  retried without rewriting the note.

Companion bridges should retain raw and cleaned OCR readings, attach uncertainty
context, and ask for clarification before broad searches when a meaningful term
is ambiguous. The adapter accepts `ocr_raw` and `ocr_cleaned` metadata for this
purpose; it must not treat uncertain OCR as authoritative user intent.

## Current milestone

This first native milestone includes pen capture, eraser/palm handling, atomic offline recovery,
HTTPS configuration, notebook session continuity, and synchronous Hermes replies. Before public
distribution it still needs device-keystore token storage, queued retry/outbox behavior, stroke
rendering benchmarks on physical BOOX hardware, accessibility review, signed release builds, and
an updater/distribution policy.
