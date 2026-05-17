---
name: quest-native-toolchain
description: |
  Use when setting up, maintaining, or troubleshooting the complete Meta Quest native development stack on Ubuntu Linux. Covers Godot, Blender MCP, OpenXR, Android SDK/NDK, asset optimization tools, sideloading with ADB, scrcpy mirroring, Monado runtime, and environment variable configuration. Use for initial workstation setup, CI toolchain installation, or resolving missing dependencies in Quest VR build pipelines.
version: "1.0.0"
author: hermes-vr-devkit
license: MIT
metadata:
  hermes:
    tags: [quest, vr, openxr, godot, blender, android, linux, meta-quest, toolchain, sideload, mcp, monado]
    related_skills: [godot-quest-dev, blender-godot-pipeline, godot-xr-interactions, mcp-server-setup]
---

# Quest Native Toolchain

Complete Meta Quest native development stack for Ubuntu Linux. One-shot install paths, validated tool versions, and project-agnostic environment setup.

## When to Use

- Setting up a fresh Ubuntu workstation for Quest VR development
- Installing or upgrading the Android SDK, NDK, or platform tools
- Configuring Godot + Blender MCP + OpenXR in a single environment
- Troubleshooting missing dependencies (`adb`, `apksigner`, `gltfpack`, etc.)
- Preparing CI/CD agents with the full Quest build stack
- Setting up Monado for local OpenXR runtime testing without a headset

## Tool Inventory (Summary)

| Tool | Install Path | Purpose |
|------|--------------|---------|
| Godot | `$HOME/bin/godot` | Game engine, headless export |
| Blender | `/usr/bin/blender` | 3D modeling, MCP integration |
| Android SDK | `$HOME/android-sdk` | Build tools, platform tools, NDK |
| Android NDK | `$HOME/android-sdk/ndk/25.2.9519653` | Native C++ builds for Quest |
| JDK 17 | `/usr/lib/jvm/java-17-openjdk-amd64` | Gradle builds, apksigner |
| ADB | `$HOME/android-sdk/platform-tools/adb` | Device communication |
| gltfpack | `$HOME/bin/gltfpack` | Mesh/texture optimization |
| scrcpy | `/usr/bin/scrcpy` | Screen mirroring + control |
| Monado | `$HOME/src/monado/install` | Local OpenXR runtime |

For full install commands and version pinning, see `references/tool-inventory.md`.

## Cloned Repositories

| Repository | Clone Path | Purpose |
|------------|------------|---------|
| `godot_openxr_vendors` | `$HOME/src/godot_openxr_vendors` | Meta OpenXR plugin for Godot |
| `monado` | `$HOME/src/monado` | Open-source OpenXR runtime |
| `OpenXR-SDK-Source` | `$HOME/src/OpenXR-SDK-Source` | Native C++ OpenXR samples |
| `hermes-vr-devkit` | `$HOME/src/hermes-vr-devkit` | Skills, scripts, templates |

## One-Shot Installation

### 1. System Dependencies

```bash
sudo apt update && sudo apt install -y \
  git git-lfs curl wget unzip p7zip-full \
  build-essential cmake ninja-build pkg-config \
  openjdk-17-jdk openjdk-17-jre \
  libgl1-mesa-dev libvulkan-dev libx11-dev libxrandr-dev \
  libwayland-dev wayland-protocols libxkbcommon-dev \
  ffmpeg libsdl2-2.0-0 adb
```

### 2. Android SDK + NDK (Command-Line)

```bash
mkdir -p "$HOME/android-sdk/cmdline-tools"
cd "$HOME/android-sdk/cmdline-tools"
wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip -q commandlinetools-linux-*.zip
mv cmdline-tools latest
```

Then install packages and accept licenses. Detailed steps in `references/android-sdk-setup.md`.

### 3. Godot (Headless + Editor)

```bash
mkdir -p "$HOME/bin"
cd "$HOME/bin"
wget https://downloads.tuxfamily.org/godotengine/4.5-stable/Godot_v4.5-stable_linux.x86_64.zip
unzip -q Godot_v4.5-stable_linux.x86_64.zip
mv Godot_v4.5-stable_linux.x86_64 godot
chmod +x godot
```

### 4. Blender (Official Repository)

```bash
sudo apt install -y blender
# Or download from https://www.blender.org/download/
```

### 5. gltfpack

```bash
mkdir -p "$HOME/bin"
cd "$HOME/bin"
wget https://github.com/zeux/gltfpack/releases/download/v0.20/gltfpack-0.20-linux.zip
unzip -q gltfpack-0.20-linux.zip
chmod +x gltfpack
```

### 6. scrcpy

```bash
sudo apt install -y scrcpy
# Or build from source for latest features
```

## Key Environment Variables

Add to `~/.bashrc` or `~/.profile`:

```bash
export ANDROID_HOME="$HOME/android-sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
export PATH="$ANDROID_HOME/build-tools/34.0.0:$PATH"
export PATH="$HOME/bin:$PATH"

export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export GODOT="$HOME/bin/godot"

# Native build
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/25.2.9519653"
export PATH="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH"
```

Reload:

```bash
source ~/.bashrc
```

## Native C++ Sample Validation (Quest-XR Build)

Validate the entire toolchain by building the Khronos OpenXR SDK sample for Android.

### 1. Clone and Prepare

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"
git clone https://github.com/KhronosGroup/OpenXR-SDK-Source.git
cd OpenXR-SDK-Source
```

### 2. Build with Android NDK

```bash
export ANDROID_NDK_HOME="$HOME/android-sdk/ndk/25.2.9519653"
mkdir -p build-android && cd build-android

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

### 3. Expected Artifacts

```bash
ls -l src/tests/hello_xr/libhello_xr.so
# Should produce an arm64 shared library
```

If this builds successfully, your NDK, CMake, and toolchain are correctly configured for Quest native development.

## scrcpy: Quest Mirroring and Control

Mirror the Quest display to your Linux desktop and inject input.

### Basic Usage

```bash
# USB connection (Quest in developer mode)
scrcpy --serial=YOUR_QUEST_SERIAL

# Over Wi-Fi (after adb tcpip 5555)
adb tcpip 5555
adb connect QUEST_IP:5555
scrcpy --serial=QUEST_IP:5555
```

### Performance Flags for VR

```bash
scrcpy \
  --max-fps=30 \
  --max-size=1024 \
  --bit-rate=4M \
  --crop=1632:1224:100:100 \
  --no-control
```

### Common Flags

| Flag | Purpose |
|------|---------|
| `--no-control` | View-only, no input injection |
| `--record=file.mp4` | Record session |
| `--fullscreen` | Fullscreen mirror |
| `--rotation=1` | Rotate 90 degrees |

## What Meta Does NOT Provide on Linux

| Tool | Linux Status | Workaround |
|------|--------------|------------|
| Meta Quest Link (Air/Cable) | Not available | Use ALVR, WiVRn, or Virtual Desktop |
| Meta Quest Developer Hub | No Linux build | Use `adb`, `scrcpy`, and command-line tools |
| Meta XR Simulator | Windows-only | Use Monado + `monado-gui` for local testing |
| Oculus Runtime | Windows-only | Use Monado or SteamVR on Linux |
| Meta Build Utils | Windows-only | Use Android SDK + NDK directly |

Linux developers rely entirely on open-source alternatives and direct ADB interaction.

## Monado: Local OpenXR Runtime

Monado is the open-source OpenXR runtime for Linux. Use it to test OpenXR logic without a physical Quest.

### Build Monado

```bash
mkdir -p "$HOME/src" && cd "$HOME/src"
git clone https://gitlab.freedesktop.org/monado/monado.git
cd monado

mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$HOME/src/monado/install" \
  -DXRT_BUILD_DRIVER_SIMULATED=ON \
  -DXRT_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
make install
```

### Run with Simulated HMD

```bash
export XR_RUNTIME_JSON="$HOME/src/monado/install/share/openxr/1/openxr_monado.json"

# In one terminal
monado-service &

# In another terminal
./your-openxr-application
```

### Verify Runtime

```bash
# List available runtimes
ls /usr/share/openxr/1/openxr_runtime.json ~/.config/openxr/1/openxr_monado.json

# Check active runtime
xrinfo
```

## Asset Optimization Quick Reference

| Tool | Command | Purpose |
|------|---------|---------|
| gltfpack | `gltfpack -si 0.5 -tc 2048 -noq -kn -i raw.glb -o opt.glb` | Reduce mesh + texture size |
| Godot import | Set "Lossy" compression, VRAM max 2K | Runtime texture budget |
| Blender decimate | Modifier > Decimate, ratio 0.5 | Quick LOD generation |

## Validation Checklist

After setup, confirm each layer:

- [ ] `adb devices` shows Quest serial when plugged in
- [ ] `adb shell getprop ro.product.model` returns Quest model
- [ ] `apksigner --version` returns a version number
- [ ] `$GODOT --version` returns 4.5-stable or higher
- [ ] `blender --version` returns 4.x
- [ ] `gltfpack -h` prints help
- [ ] `scrcpy --version` prints version
- [ ] `javac -version` prints 17.x
- [ ] Native C++ OpenXR sample builds successfully
- [ ] Monado `xrinfo` shows Monado runtime

## Troubleshooting

### `adb: command not found`

Platform tools not on PATH. Add `$ANDROID_HOME/platform-tools` to `~/.bashrc`.

### `apksigner: command not found`

Build-tools not on PATH. Add `$ANDROID_HOME/build-tools/34.0.0` to `~/.bashrc`.

### Godot cannot find Android SDK

Set in Godot Editor Settings: `export/android/android_sdk_path = "/home/USER/android-sdk"`
Or run: `$GODOT --headless --editor-settings ...` (see `godot-quest-dev` skill).

### NDK not found during native build

Ensure `ANDROID_NDK_HOME` is exported and points to a valid NDK directory.

### Monado fails to start

Install udev rules for your GPU and ensure your user is in the `video` and `render` groups:

```bash
sudo usermod -aG video,render $USER
```

## See Also

- `references/tool-inventory.md` -- Full tool inventory with install commands, versions, purposes
- `references/android-sdk-setup.md` -- Step-by-step Android SDK installation, NDK, platform tools, license acceptance
- `godot-quest-dev` skill -- Godot project setup, export, and Quest deployment
- `blender-godot-pipeline` skill -- Asset pipeline from Blender to Godot
- `godot-xr-interactions` skill -- XR interaction patterns and locomotion
- `mcp-server-setup` skill -- Blender MCP server configuration
