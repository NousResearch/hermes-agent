# Hermes Android MVP support matrix

This file locks the Android APK MVP support matrix for branch `feat/termux-install-path`.

Later Android tasks should match these values until this file and `docs/plans/2026-04-10-android-apk-ci-port-plan.md` are updated together.

## Repo-grounded constraints

- `pyproject.toml` currently requires Python `>=3.11`.
- `scripts/install.sh` and `setup-hermes.sh` both provision Python `3.11` today.
- The repo's current Android story is still the documented Termux path (`constraints-termux.txt` and `website/docs/getting-started/termux.md`), so the first APK release should stay intentionally narrow.
- Chaquopy 17.0 requires `minSdk >= 24`, supports Python `3.10` through `3.14`, and requires the build host Python major/minor version to match the app runtime Python version.

## Locked MVP support matrix

| Area | Locked value | Why this is locked for MVP |
| --- | --- | --- |
| min SDK | 24 | This is the Chaquopy 17.0 floor, so it is the lowest Android API level we can support without taking older plugin baggage into the MVP. |
| target SDK | 35 | Hold the first APK release on Android 15 / API 35 while the Android lane is being created. This stays current enough without also taking API 36 churn during the initial port. |
| CI emulator API | 35 (`x86_64`, Google APIs image) | One modern emulator target is enough for the MVP smoke lane. Do not start with an emulator matrix. |
| Embedded Python runtime | Chaquopy 17.0.0 + Python 3.11 | This matches the repo-wide Python 3.11 baseline already used by packaging, install, and setup flows. |
| Build-host Python for CI/local Android builds | 3.11 | Chaquopy requires the build host Python major/minor version to match the packaged app Python version. |

Implementation guardrail for later tasks: when `android/app/build.gradle.kts` is created, keep `compileSdk = 35`, `minSdk = 24`, and `targetSdk = 35` unless this matrix is revised first.

## Artifact and ABI policy

- Pull requests: produce one universal debug APK.
- GitHub releases: produce one signed universal release APK and one signed release AAB.
- No ABI splits in MVP.
- Keep the MVP ABI set narrow and 64-bit only:
  - `arm64-v8a` for physical devices
  - `x86_64` for CI and emulator coverage
- Do not ship separate per-ABI artifacts in the first public release.
- Do not expand the first public release to Play upload, Play-managed delivery, or a wider ABI matrix before GitHub release artifacts work end-to-end.
- 32-bit ABIs stay out of the MVP even though Python 3.11 could support them through Chaquopy; keeping the first release 64-bit only is the narrower support posture for this branch.

## What later subagents should treat as fixed

Until this file changes, later Android tasks should assume:

1. `android/app/build.gradle.kts` uses SDK 24/35/35 (`min` / `target` / `compile`).
2. Android CI installs Python 3.11 on the runner before invoking Gradle.
3. The first Android workflow only needs a single API 35 emulator target.
4. Artifact publishing stays universal-only, with no ABI split configuration.
5. The MVP runtime ABI filters stay limited to `arm64-v8a` and `x86_64`.
