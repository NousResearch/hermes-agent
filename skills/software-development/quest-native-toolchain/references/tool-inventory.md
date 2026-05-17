# Tool Inventory

Full inventory of the Quest native toolchain. Each entry includes purpose, recommended version, install command, and verification step.

## Core Engine

### Godot
- **Purpose:** Game engine, scene editor, headless exporter
- **Recommended Version:** 4.5-stable or later
- **Install Path:** `$HOME/bin/godot`
- **Download:** https://downloads.tuxfamily.org/godotengine/

```bash
mkdir -p "$HOME/bin"
cd "$HOME/bin"
wget https://downloads.tuxfamily.org/godotengine/4.5-stable/Godot_v4.5-stable_linux.x86_64.zip
unzip -q Godot_v4.5-stable_linux.x86_64.zip
mv Godot_v4.5-stable_linux.x86_64 godot
chmod +x godot
```

- **Verify:** `godot --version`
- **Notes:** Forward+ renderer is unavailable on Quest; always use Mobile renderer for VR.

### Blender
- **Purpose:** 3D modeling, animation, UV unwrap, GLB export
- **Recommended Version:** 4.2 LTS or 4.x
- **Install Path:** `/usr/bin/blender`
- **Download:** https://www.blender.org/download/

```bash
sudo apt install -y blender
```

- **Verify:** `blender --version`
- **Notes:** Enable the glTF 2.0 exporter (built-in). For MCP integration, see `mcp-server-setup` skill.

## Android Build Stack

### Android SDK (Command-Line Tools)
- **Purpose:** ADB, apksigner, aapt, build-tools, platform images
- **Recommended Version:** cmdline-tools 11076708 or later
- **Install Path:** `$HOME/android-sdk`

```bash
mkdir -p "$HOME/android-sdk/cmdline-tools"
cd "$HOME/android-sdk/cmdline-tools"
wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip -q commandlinetools-linux-*.zip
mv cmdline-tools latest
```

- **Verify:** `sdkmanager --version`

### Android NDK
- **Purpose:** Native C/C++ compilation for arm64-v8a
- **Recommended Version:** 25.2.9519653 (r25c)
- **Install Path:** `$HOME/android-sdk/ndk/25.2.9519653`

```bash
sdkmanager "ndk;25.2.9519653"
```

- **Verify:** `ls $ANDROID_HOME/ndk/25.2.9519653/ndk-build`
- **Notes:** Quest uses arm64-v8a ABI. NDK 25 is the last release before the LLVM toolchain restructuring in NDK 26.

### Platform Tools
- **Purpose:** ADB, fastboot, systrace
- **Recommended Version:** Latest (auto-updated via sdkmanager)
- **Install Path:** `$HOME/android-sdk/platform-tools`

```bash
sdkmanager "platform-tools"
```

- **Verify:** `adb version`

### Build Tools
- **Purpose:** apksigner, zipalign, aapt2
- **Recommended Version:** 34.0.0 or later
- **Install Path:** `$HOME/android-sdk/build-tools/34.0.0`

```bash
sdkmanager "build-tools;34.0.0"
```

- **Verify:** `apksigner --version`

### Platforms
- **Purpose:** Android API headers and libraries
- **Recommended Version:** android-29 (minimum for OpenXR) and android-34
- **Install Path:** `$HOME/android-sdk/platforms/android-29`

```bash
sdkmanager "platforms;android-29" "platforms;android-34"
```

### JDK 17
- **Purpose:** Gradle daemon, apksigner, keytool
- **Recommended Version:** OpenJDK 17
- **Install Path:** `/usr/lib/jvm/java-17-openjdk-amd64`

```bash
sudo apt install -y openjdk-17-jdk openjdk-17-jre
```

- **Verify:** `javac -version`
- **Notes:** Godot 4.5+ requires JDK 17. Do not use JDK 21 for Godot Android builds.

## Optimization Tools

### gltfpack
- **Purpose:** Mesh quantization, LOD generation, texture compression for GLB
- **Recommended Version:** 0.20 or later
- **Install Path:** `$HOME/bin/gltfpack`
- **Download:** https://github.com/zeux/gltfpack/releases

```bash
mkdir -p "$HOME/bin"
cd "$HOME/bin"
wget https://github.com/zeux/gltfpack/releases/download/v0.20/gltfpack-0.20-linux.zip
unzip -q gltfpack-0.20-linux.zip
chmod +x gltfpack
```

- **Verify:** `gltfpack -h`
- **Notes:** Use `-si 0.5` for 50% triangle reduction, `-tc 2048` for texture cap, `-noq` to disable quantization if Godot import fails.

## Sideload and Debug

### scrcpy
- **Purpose:** Mirror Quest display to Linux desktop, inject input
- **Recommended Version:** 2.4 or later (snap/APT)
- **Install Path:** `/usr/bin/scrcpy`

```bash
sudo apt install -y scrcpy
```

- **Verify:** `scrcpy --version`
- **Notes:** Works over USB and Wi-Fi. Requires ADB debugging enabled on Quest.

## OpenXR Runtime

### Monado
- **Purpose:** Open-source OpenXR runtime for Linux desktop testing
- **Recommended Version:** Latest main branch
- **Install Path:** `$HOME/src/monado/install`

```bash
# See SKILL.md Monado section for full build instructions
```

- **Verify:** `xrinfo` after setting `XR_RUNTIME_JSON`
- **Notes:** Supports simulated HMD for testing OpenXR initialization without physical hardware.

## Environment Summary

Ensure these are in your shell profile:

```bash
export ANDROID_HOME="$HOME/android-sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/25.2.9519653"
export GODOT="$HOME/bin/godot"
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
export PATH="$ANDROID_HOME/build-tools/34.0.0:$PATH"
export PATH="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH"
export PATH="$HOME/bin:$PATH"
```
