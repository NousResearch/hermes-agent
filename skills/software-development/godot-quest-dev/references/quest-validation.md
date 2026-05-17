# Quest Build Validation Checklist

Run this checklist after every export to confirm the APK will launch in immersive VR mode.

## Pre-Export Checks

- [ ] Renderer set to **Mobile** (Project Settings -> Rendering -> Renderer)
- [ ] VSync disabled (Project Settings -> Display -> Window -> VSync Mode)
- [ ] Export preset `xr_features/xr_mode = 1` (OpenXR)
- [ ] Export preset `gradle_build/use_gradle_build = true`
- [ ] Export preset `package/app_category = Game` (NOT Accessibility)
- [ ] Export preset `package/signed = false` (we sign manually)
- [ ] Architectures = `arm64` only
- [ ] Minimum API = 29
- [ ] Meta OpenXR Vendors plugin enabled (Project Settings -> Plugins)
- [ ] `android/.build_version` matches Godot version
- [ ] `icon.svg` exists in project root (headless export requires it)

## Post-Export Checks

- [ ] APK file exists and is > 5MB
- [ ] `apksigner verify myapp.apk` passes
- [ ] `aapt dump xmltree myapp.apk AndroidManifest.xml | grep -i "vr\|game"` shows correct entries
- [ ] `libopenxr_loader.so` exists in APK: `unzip -l myapp.apk | grep libopenxr_loader`

## On-Device Checks

- [ ] `adb devices` shows Quest in dev mode
- [ ] `adb install -r myapp.apk` succeeds
- [ ] App launches without "controller required" dialog (use `am start`, not monkey)
- [ ] `adb logcat -s godot` shows OpenXR initialization
- [ ] `adb logcat -s XR` shows no errors
- [ ] `adb logcat -s VrApi | grep FPS` shows target framerate (72 or 90)
- [ ] No black screen, controllers render, tracking works

## Smoke Test Commands

```bash
# Install and launch
adb install -r myapp.apk
adb shell am start -n com.yourcompany.yourapp/com.godot.game.GodotApp

# Watch logs
adb logcat -s godot:V XR:V VrApi:V DEBUG:V *:S

# Check frame rate
adb logcat -s VrApi:V | grep FPS

# Screenshot (for remote debugging)
adb shell screencap -p /sdcard/screen.png
adb pull /sdcard/screen.png
```
