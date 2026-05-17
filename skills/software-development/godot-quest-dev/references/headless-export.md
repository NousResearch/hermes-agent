# Headless Export Workflow

Godot 4.5 supports fully automated headless export for CI/CD pipelines.

## One-Time Per Project

```bash
cd /path/to/your-godot-project
$GODOT --headless --install-android-build-template
# Creates android/ directory with correct .build_version
```

**CRITICAL:** Do NOT manually extract `android_source.zip`. The `--install-android-build-template` flag creates internal state that Godot's export system requires. Manual extraction is silently ignored.

**Build template version pinning:** After upgrading Godot, delete `android/` and re-run `--install-android-build-template`. Godot checks `android/.build_version`.

## Export Command

```bash
$GODOT --headless --export-release "Android Quest" myapp-unsigned.apk
```

## Sign Manually

Godot headless ignores keystore paths for OpenXR presets. Export unsigned, then sign:

```bash
apksigner sign --ks debug.keystore --ks-pass pass:android \
  --key-pass pass:android --out myapp.apk myapp-unsigned.apk
apksigner verify myapp.apk
```

## Full Pipeline Script

```bash
#!/bin/bash
set -e
PROJECT_DIR="${1:-.}"
GODOT="${GODOT:-$HOME/bin/godot}"
KEYSTORE="${KEYSTORE:-$HOME/.android/debug.keystore}"
APK_NAME="${2:-myapp}"

cd "$PROJECT_DIR"

# Export
$GODOT --headless --export-release "Android Quest" "${APK_NAME}-unsigned.apk"

# Sign
apksigner sign --ks "$KEYSTORE" --ks-pass pass:android \
  --key-pass pass:android --out "${APK_NAME}.apk" "${APK_NAME}-unsigned.apk"

# Verify
apksigner verify "${APK_NAME}.apk"

echo "Built: ${APK_NAME}.apk"
```

## Godot Editor Settings

Stored in `~/.config/godot/editor_settings-4.5.tres`:

```tres
[resource]
export/android/android_sdk_path = "/home/USER/android-sdk"
export/android/java_sdk_path = "/usr/lib/jvm/java-17-openjdk-amd64"
export/android/debug_keystore = "/home/USER/.android/debug.keystore"
export/android/debug_keystore_pass = "android"
```

## What Does NOT Work

- Manually extracting `android_source.zip` to `android/` -- Godot ignores it
- Headless export before running `--install-android-build-template` -- fails
- Setting `package/signed=true` in preset for headless -- Godot ignores keystore path
