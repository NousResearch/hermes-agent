# Full Hermes Agent on Android: implementation plan

## Status

The APK is experimental. This document is a buildable-work plan, not a claim that the full agent already runs inside Android. The existing Termux installation remains the documented Android path for the full CLI until the APK passes the acceptance checks below.

## Why copying the desktop dependencies is not enough

The Android app currently configures Chaquopy 17.1.0 and Python 3.14 by default (with a Gradle property to select 3.13). That config selects an interpreter; it does not package the Hermes Python application or prove that its dependencies have Android/arm64-compatible wheels.

The main project's packaging rules also intentionally block ordinary PyPI wheel builds. Do not work around that by adding `pip install hermes-agent` to the APK build. Desktop wheels and native libraries cannot be assumed to work on Android.

## Target architecture

1. **Inventory the runtime.** Derive a dependency report from the exact current `pyproject.toml`, lock file, and imports used by the agent. Classify every dependency as pure Python, Android-compatible wheel available, needs a Termux/system binary, desktop-only, or unresolved.
2. **Define an explicit Android profile.** Prefer one shared Hermes core with platform adapters and an explicit Android capability profile. Do not silently omit required tools or claim desktop parity when a tool is unavailable.
3. **Package the core reproducibly.** Build the Android-compatible Python dependency set for supported ABIs, include the application source and required resources, and fail the build when a required dependency is missing. Avoid downloading arbitrary packages at app startup.
4. **Add an Android runtime bridge.** Use an app-owned service and structured IPC between Kotlin and Python. Keep long-running work visible through an Android foreground service where permitted; handle process death and battery restrictions.
5. **Reuse the existing gateway for remote capabilities.** Browser/desktop control, Docker-dependent tools, and other host-specific integrations should run on a paired Hermes gateway/worker when they cannot safely run on-device. Android must show whether a task is local or remote.
6. **Preserve identity and user data.** Reuse existing profile, memory, soul, and configuration formats where feasible. Migrate data explicitly; never delete Termux sessions or configuration as part of APK installation.
7. **Gate side effects.** Sending messages, submitting forms, purchases, deleting files, and modifying important data require a fresh, task-specific user approval by default. Log decisions and provide stop/revoke controls.

## Required validation before calling the APK a full agent

- Clean Gradle debug build succeeds in CI for the supported ABIs.
- APK contains the intended Hermes application code and all required Android-compatible dependencies.
- Instrumentation test starts the Python runtime and imports the core entry point.
- Smoke test runs a harmless task through the actual agent, not a mock screen.
- Tests cover missing dependencies, app restart, process death, permission denial, and gateway disconnection.
- Install and smoke-test on a real Android device (including the target Android version) before release.
- Publish a capability matrix so unavailable desktop-only integrations are clearly marked.

## First implementation milestone

This branch establishes the acceptance criteria and safe packaging strategy. The next code PR should implement the dependency inventory and a CI check against the real dependency lock, then package the smallest verified core. Do not label this planning milestone as a completed full-agent APK.
