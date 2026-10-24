# Hermes Agent on Android

This guide distinguishes the supported Termux workflow from the experimental APK files in this repository. **Do not assume that the APK provides the same features as the desktop/server agent.**

## Recommended: Termux on Android

Hermes' documented Android path is Termux on a 64-bit ARM (aarch64/arm64) device. Use the official [Hermes Termux installation guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux) as the source of truth for current commands.

1. Install Termux from a trusted source linked by the [Termux project](https://termux.dev/).
2. Follow the current Hermes Termux guide.
3. If installation fails, save the complete error output and check the supported Python version and architecture before changing dependencies.
4. Start Hermes from Termux using the command documented by the installed package.

## APK in this repository: experimental

The repository has advertised an [Android APK release](https://github.com/luciano2-web/hermes-agent/releases/tag/v1.0.0-android). A download does **not** prove that it installs, starts, or supports all Hermes features on every Android device. Treat it as experimental until a reproducible build and real-device tests pass.

### Python compatibility status

The Android Gradle configuration now selects **Chaquopy 17.1.0 with Python 3.14**, matching the current Hermes runtime dependency markers. Chaquopy 12.0.1 was not compatible with this configuration.

This change fixes the declared interpreter-version mismatch only. It does **not** mean the full Hermes Agent is packaged in the APK yet:

- The main repository intentionally blocks ordinary PyPI wheel builds in `setup.py`.
- Desktop Python wheels cannot be assumed to work on Android/arm64.
- Android-compatible wheels or a deliberately reduced Android dependency profile still need to be prepared and tested.
- The Android source tree, Gradle project configuration, permissions, background service, and app startup must all be verified by a clean build and device tests before calling the APK functional.

Do not add `pip install hermes-agent` to the Android build as a workaround: the project explicitly rejects that distribution route and many dependencies may not have Android-compatible wheels.

### Build prerequisites

- A supported desktop build environment (building directly on an Android tablet is not currently documented as supported).
- JDK 17, Android SDK platform 34, and Android build-tools 34.0.0.
- Android Gradle plugin and Gradle wrapper versions compatible with Chaquopy 17.1.
- An Android-compatible Python dependency strategy.

From a desktop terminal, run:

```bash
cd android
./gradlew tasks
./gradlew assembleDebug --stacktrace
```

These are validation commands, not a claim that the current project builds. Do not publish a release until the build, install, startup, and core interaction have been tested on a real device or emulator.

## Building a signed release APK in GitHub Actions

A debug APK can be built without signing secrets. To request a signed release build, run the **Android APK Build** workflow manually on `main` and enable **Build a release APK (signed)**.

Configure these repository Actions secrets first:

- `ANDROID_SIGNING_KEY`: the release keystore file encoded as one base64 string.
- `ANDROID_STORE_PASSWORD`: the keystore password.
- `ANDROID_KEY_PASSWORD`: the key password.
- `ANDROID_KEY_ALIAS`: the key alias (optional; defaults to `hermes`).

The workflow builds the release APK from source and fails if signing credentials are missing; it must never relabel or upload a debug APK as a signed release. Keep the keystore and passwords out of the repository and logs. A signed APK still does not mean the full Hermes Python package is bundled or functional on Android.

## Android limitations

- Android may pause or terminate background processes to save battery. A terminal session is not a guarantee of continuous 24/7 execution.
- Some desktop/server integrations depend on Docker, system services, native libraries, or tools unavailable or restricted on Android.
- Never include API keys, tokens, passwords, private conversations, or unredacted personal information in bug reports.

## Troubleshooting

- **App not installed:** check the download, package conflicts, free storage, Android version, and the exact installer error.
- **Crash at startup:** collect relevant Android logs and report the first exception.
- **Python/module import error:** check the Python version and whether each dependency has a compatible Android/arm64 wheel.
- **Background work stops:** verify battery optimization and foreground-service behavior on the target Android version.

## Official references

- [Hermes Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux)
- [Hermes installation guide](https://hermes-agent.nousresearch.com/docs/getting-started/installation)
- [Chaquopy version compatibility](https://chaquo.com/chaquopy/doc/current/versions.html)
- [Android SDK command-line tools](https://developer.android.com/tools/sdkmanager)
- [Termux project](https://termux.dev/)
