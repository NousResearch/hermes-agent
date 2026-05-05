# Hermes Agent v0.13.12

Android/F-Droid reproducibility follow-up.

## Android

- Cleans the tracked `build/` directory before GitHub Android release builds so release APKs match F-Droid's clean source build.
- Canonicalizes Chaquopy `build.json` ordering before APK packaging.
- Keeps the Python 3.13 Android runtime, single universal APK output, and disabled Android dependency metadata from v0.13.11.

## Packaging

- Bumps the package version to 0.13.12.
- Updates the F-Droid metadata template for Android version code 131290.
