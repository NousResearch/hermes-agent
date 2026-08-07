# Immersive VR manifest requirements

Quest OS launches an APK as a **flat 2D panel** unless the merged manifest
advertises VR correctly. The Meta OpenXR Vendors plugin injects most entries
when enabled; the export preset still has one override that bites everyone.

## Required markers

Merged `AndroidManifest.xml` should include:

- `uses-feature` `android.hardware.vr.headtracking` (required)
- MAIN activity intent-filter category `com.oculus.intent.category.VR`
- Activity meta-data `com.oculus.vr.focusaware` = `true`
- Application `android:isGame="true"` and `android:appCategory="game"`

## App Category dropdown override

Godot's export preset **Package → App Category** defaults to
**Accessibility**. That UI value **overrides** hand-edited
`android:appCategory` in template manifests.

**Always set App Category to Game** in the export preset (value `1` in
`export_presets.cfg` as `package/app_category=1` on many versions).

## Two manifests

Gradle template may keep:

1. `android/build/AndroidManifest.xml`
2. `android/build/src/debug/AndroidManifest.xml` (debug overlay wins)

Verify both after plugin install.

## Inspect APK

```bash
aapt dump xmltree build/app.apk AndroidManifest.xml \
  | rg -i 'headtracking|oculus|VR|isGame|appCategory|focusaware'
```

## Common logcat symptoms

| Symptom | Likely cause |
|---------|----------------|
| `isVrApplication: false` | Missing VR category or focusaware |
| `XR_ERROR_FORM_FACTOR_UNSUPPORTED` | Missing headtracking feature or OpenXR loader |
| SurfaceView / 2D panel | `app_category` not Game |
| `VK_ERROR_SURFACE_LOST_KHR` | Launched outside VR mode |

## Last-resort apktool surgery

Prefer fixing the export preset. If you must patch a built APK:

1. `apktool d app.apk -o unpacked`
2. Fix categories / isGame / VR intent in `AndroidManifest.xml`
3. `apktool b unpacked -o app-rebuilt.apk`
4. `apksigner sign ...` again
