# Quest VR Manifest Requirements

For the APK to launch in **immersive VR mode** on Quest (not as a flat 2D screen), the merged `AndroidManifest.xml` must contain specific entries.

## Required Entries

```xml
<manifest ...>
    <uses-feature android:name="android.hardware.vr.headtracking"
                  android:version="1"
                  android:required="true" />

    <application android:label="@string/godot_project_name"
                 android:allowBackup="false"
                 android:isGame="true"
                 android:appCategory="game"
                 android:extractNativeLibs="true">

        <activity android:name="com.godot.game.GodotApp"
                  ...>
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
                <category android:name="com.oculus.intent.category.VR" />
            </intent-filter>
            <meta-data android:name="com.oculus.vr.focusaware"
                       android:value="true" />
        </activity>
    </application>
</manifest>
```

## The App Category Dropdown Override

**CRITICAL:** Godot's export preset has an **"App Category"** dropdown under Package options that defaults to `Accessibility`. This dropdown **overrides** any `android:appCategory` changes you make in manifest files. **Must set to `Game`** for Quest immersive mode.

## Two Manifest Files

Godot's build template maintains:
1. `android/build/AndroidManifest.xml` -- main manifest
2. `android/build/src/debug/AndroidManifest.xml` -- debug overlay

The debug overlay **overrides** the main. The Meta OpenXR Vendors plugin auto-injects correct entries, but always verify both files.

## Verification Commands

```bash
# Decompile APK and check manifest
unzip -p myapp.apk AndroidManifest.xml | xxd | head -100
# Or use aapt:
aapt dump xmltree myapp.apk AndroidManifest.xml | grep -E "headtracking|VR|focusaware|isGame|appCategory"

# Check on device
adb shell dumpsys package com.yourcompany.yourapp | grep -i "vr\|game"
```

## Post-Export Manifest Surgery (Last Resort)

If the App Category dropdown is not available or not working:

```bash
# 1. Decompile
apktool d myapp.apk -o myapp-decompiled

# 2. Patch manifest
sed -i 's/android:appCategory="[^"]*"/android:appCategory="game"/' myapp-decompiled/AndroidManifest.xml
sed -i 's/android:isGame="[^"]*"/android:isGame="true"/' myapp-decompiled/AndroidManifest.xml

# 3. Ensure VR category and focusaware metadata exist
# (Add manually if missing -- see Required Entries above)

# 4. Recompile
apktool b myapp-decompiled -o myapp-recompiled.apk

# 5. Re-sign
apksigner sign --ks debug.keystore --ks-pass pass:android \
  --key-pass pass:android --out myapp.apk myapp-recompiled.apk
```

## Common Manifest Errors

| Logcat Message | Missing Entry |
|----------------|---------------|
| `"isVrApplication":false` | `com.oculus.intent.category.VR` or `com.oculus.vr.focusaware` |
| `XR_ERROR_FORM_FACTOR_UNSUPPORTED` | `android.hardware.vr.headtracking` or `libopenxr_loader.so` |
| `SurfaceView rendering instead of VR swapchain` | `android:appCategory="game"` or `isGame="true"` |
| `VK_ERROR_SURFACE_LOST_KHR` | App launching in 2D mode -- check all manifest entries |
