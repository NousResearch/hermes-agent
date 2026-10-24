# Hermes Agent on Android

This guide distinguishes the supported Termux workflow from the experimental APK files in this repository. **Do not assume that the APK provides the same features as the desktop/server agent.** Check the release notes and test the exact build on your device before relying on it.

## Recommended: Termux on Android

Hermes' documented Android path is Termux on a 64-bit ARM (aarch64/arm64) device. Use the official [Hermes Termux installation guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux) as the source of truth for the current repository and package commands.

1. Install Termux from a trusted source linked by the [Termux project](https://termux.dev/).
2. Open Termux and follow the current Hermes Termux guide. Do not copy a guessed package filename: package versions and repository URLs change.
3. If installation fails, save the complete error output and check the supported Python version and architecture before changing dependencies.
4. Start Hermes from Termux using the command documented by the installed package.

### Android-specific limitations

- Android may pause or terminate background processes to save battery. A terminal session is not a guarantee of continuous 24/7 execution.
- Some desktop/server integrations depend on Docker, system services, native libraries, or tools that are unavailable or restricted on Android.
- Keep Termux and Hermes updated using their documented update procedures. Avoid mixing package managers or installing the desktop/server installer inside Termux unless the official guide explicitly instructs you to.
- Do not paste API keys into screenshots, public issues, or chat logs.

## APK in this repository: experimental

The repository has advertised an APK at:

- [Android APK release](https://github.com/luciano2-web/hermes-agent/releases/tag/v1.0.0-android)
- [APK file](https://github.com/luciano2-web/hermes-agent/releases/download/v1.0.0-android/Hermes-Agent-1.0.0-debug.apk)

Availability of a download does **not** prove that it installs, starts, or supports all Hermes features on every Android device. Treat this APK as experimental until the project publishes reproducible build instructions, a verified checksum, tested Android versions, and CI results.

Before installing:
1. Review the release assets and release notes.
2. Verify the publisher and checksum if one is published. If there is no checksum or verifiable build provenance, understand that you cannot independently confirm the downloaded file matches the source.
3. Only install if you trust the source. Android may require allowing your browser or file manager to install unknown apps; turn that permission back off after installation if you enabled it solely for this file.
4. Do not uninstall a working installation or clear app data as a first troubleshooting step; either action can remove local data.

If installation fails, record the device Android version, CPU architecture, APK version, exact error text, and (for crashes) relevant log output. Remove secrets and personal data before sharing logs.

## Building the APK from source

The current Android project should be treated as **unverified until a clean build succeeds in CI**. The Gradle project declares Android SDK 34, minimum Android API 24, and Chaquopy Python 3.10, while the main Python project currently declares Python `>=3.11,<3.15`. That version mismatch is a compatibility risk: do not assume that installing the Python package from PyPI into the APK will work.

The repository's build script also attempts to install Android SDK components using the legacy `android update sdk` command. Modern Android SDK command-line tools use `sdkmanager`; if the legacy command is missing, install the required SDK packages with Android Studio's SDK Manager or `sdkmanager`, then rerun only after checking the script's other requirements.

### Build prerequisites

- A desktop operating system supported by Android Studio/Gradle (building directly on an Android tablet is not documented as supported).
- JDK 17, Android SDK platform 34, and Android build-tools 34.0.0.
- The Android Gradle wrapper and all required dependencies.
- A compatible Python runtime/dependency strategy. Resolve the Python 3.10 versus project Python requirement before claiming the APK embeds the full agent.

From a desktop terminal, inspect the project and run the wrapper's validation tasks first:

```bash
cd android
./gradlew tasks
./gradlew assembleDebug --stacktrace
```

On Windows, use `gradlew.bat` instead of `./gradlew`. These commands are diagnostic examples, not a guarantee that the current project builds. Do not create a release build or publish an APK until the build, installation, startup, and core chat flow have been tested on a real device or emulator.

## Troubleshooting checklist

- **“App not installed”**: check whether the APK download completed, whether the package conflicts with an existing install, available storage, and Android's reported error. Do not assume that reinstalling is safe for local data.
- **Parse error / incompatible package**: re-download from the release page and check the device architecture and Android version against verified release metadata.
- **Crash at startup**: collect Android logs and report the first relevant exception; clearing app data is a last resort because it can erase configuration.
- **Python/module import error**: check the Python version, wheel availability for Android/arm64, and whether every dependency is actually packaged. Desktop PyPI compatibility does not imply Android compatibility.
- **Network/provider errors**: check connectivity and provider configuration. Model requests generally require internet access unless a separately configured local model is genuinely supported.
- **Background work stops**: check Android battery optimization and foreground-service behavior. Do not promise 24/7 execution unless it has been tested on the target Android version.

## Reporting an Android bug

Open an issue in this repository with:
- device model and Android version;
- CPU architecture (for example, arm64-v8a);
- installation method (Termux or APK) and exact version/release;
- command or steps that reproduce the problem;
- complete error message and relevant redacted logs;
- expected result and actual result.

Never include API keys, tokens, passwords, private conversations, or unredacted personal information.

## Official references

- [Hermes Termux guide](https://hermes-agent.nousresearch.com/docs/getting-started/termux)
- [Hermes installation guide](https://hermes-agent.nousresearch.com/docs/getting-started/installation)
- [Android SDK command-line tools](https://developer.android.com/tools/sdkmanager)
- [Termux project](https://termux.dev/)
