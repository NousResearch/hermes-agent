# Hermes Agent v0.13.13

This release fixes the remaining F-Droid reproducible-build differences in the
Android APK.

## Android

- Forces GitHub Android release builds to use hash-based Python bytecode by
  setting `SOURCE_DATE_EPOCH`.
- Canonicalizes Chaquopy's generated `build.json` before Android asset merging.
- Keeps the universal signed APK layout introduced for F-Droid verification.

## F-Droid

- Bumps the package version to 0.13.13.
- Updates the F-Droid metadata template for Android version code 131390.
