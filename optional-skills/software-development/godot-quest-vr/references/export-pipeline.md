# Export, sign, install

## One-time per project

```bash
cd /path/to/your-godot-project
"${GODOT}" --headless --install-android-build-template
```

Do **not** manually unzip `android_source.zip` into `android/`. Godot tracks
internal state via `android/.build_version`. After upgrading Godot, delete
`android/` and reinstall the template.

## Meta OpenXR Vendors plugin

Use GodotVR `godot_openxr_vendors` **4.3.1-stable** with Godot 4.5
(5.0.x targets 4.6+).

```bash
wget https://github.com/GodotVR/godot_openxr_vendors/releases/download/4.3.1-stable/godotopenxrvendorsaddon.zip
unzip -q godotopenxrvendorsaddon.zip
# Zip nests under asset/ — wrong depth breaks libopenxr_loader.so
mv asset/addons ./   # run from project root; merge with existing addons/
rm -rf asset godotopenxrvendorsaddon.zip
```

In Godot: Project → Project Settings → Plugins → enable **Godot OpenXR Vendors**.

## Export preset checklist

Create **Project → Export → Add → Android** (name it e.g. `Android Quest`):

- XR Mode: **OpenXR**
- Use Gradle Build: **on**
- Package → Signed: **off** for headless (sign with apksigner after)
- Package → App Category: **Game** (not Accessibility)
- Architectures: **arm64-v8a** only
- Min SDK: **29+**

Project Settings:

- Rendering → Renderer → **Mobile**
- Display → Window → VSync Mode → **Disabled**

## Headless export + sign

```bash
mkdir -p build
"${GODOT}" --headless --export-release "Android Quest" build/app-unsigned.apk

KEYSTORE="${KEYSTORE:-${HOME}/.android/debug.keystore}"
apksigner sign \
  --ks "${KEYSTORE}" --ks-pass pass:android --key-pass pass:android \
  --out build/app.apk build/app-unsigned.apk
apksigner verify build/app.apk
```

Godot headless commonly **ignores keystore fields** on OpenXR presets. Treat
unsigned export + `apksigner` as the supported path for CI.

## Install and launch

```bash
adb install -r build/app.apk
# Prefer launcher from Quest UI; for automation:
adb shell am start -n com.yourcompany.yourapp/com.godot.game.GodotApp
```

Replace the package name with your export package/unique name.

## Logs

```bash
adb logcat -c
adb logcat | rg -i 'godot|openxr|xr_|vulkan|androidruntime'
```

## Release signing

For store/sidequest release builds, use a real upload keystore and never
commit it. Debug keystore is fine for local immersion testing only.
