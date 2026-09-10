# Android / host toolchain

Portable paths use `${HOME}`. Adjust if your distro or company image differs.

## System packages (Debian/Ubuntu)

```bash
sudo apt update && sudo apt install -y \
  git curl wget unzip \
  openjdk-17-jdk \
  libgl1-mesa-dev libvulkan-dev \
  adb scrcpy ffmpeg
```

macOS: install JDK 17 (Temurin), Android command-line tools, and
`brew install android-platform-tools scrcpy` (or SDK platform-tools).

## Android SDK (command-line)

```bash
mkdir -p "${HOME}/android-sdk/cmdline-tools"
cd "${HOME}/android-sdk/cmdline-tools"
# Pick current commandlinetools zip from Google's Android studio downloads page
wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip -q commandlinetools-linux-*.zip
mv cmdline-tools latest

export ANDROID_HOME="${HOME}/android-sdk"
yes | "${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager" --licenses
"${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager" \
  "platform-tools" \
  "platforms;android-34" \
  "build-tools;34.0.0" \
  "ndk;25.2.9519653"
```

Ensure `adb` and `apksigner` resolve:

```bash
export PATH="${ANDROID_HOME}/platform-tools:${ANDROID_HOME}/build-tools/34.0.0:${PATH}"
command -v adb apksigner
```

## Godot binary

```bash
mkdir -p "${HOME}/bin"
cd "${HOME}/bin"
# Download Godot 4.5+ stable linux.x86_64 (or macOS universal) from godotengine.org
# mv Godot_v4.5-stable_linux.x86_64 godot && chmod +x godot
export GODOT="${HOME}/bin/godot"
"$GODOT" --version
```

## gltfpack (optional)

```bash
# From zeux/meshoptimizer releases — place binary on PATH as gltfpack
command -v gltfpack || echo "optional"
```

## Debug keystore

```bash
mkdir -p "${HOME}/.android"
keytool -genkeypair -v -keystore "${HOME}/.android/debug.keystore" \
  -storepass android -alias androiddebugkey -keypass android \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -dname "CN=Android Debug,O=Android,C=US" 2>/dev/null || true
```

## Godot editor Android paths

In Editor Settings → Export → Android, set SDK and Java paths to the same
locations as `ANDROID_HOME` / `JAVA_HOME`. Without this, GUI export fails even
when CLI tools work.

## Quest device

1. Developer Mode on the headset (Meta developer account).
2. USB link; accept debugging prompt in headset.
3. `adb devices` shows `device` (not `unauthorized`).
