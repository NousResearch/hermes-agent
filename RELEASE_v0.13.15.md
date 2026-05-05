# Hermes Agent v0.13.15

This release adds the metadata needed for F-Droid autoupdate checks to track
future semver tags without being limited to the 0.13 release line.

## F-Droid

- Adds a tracked `fdroid/com.nousresearch.hermesagent.version` file containing
  the literal Android `versionName` and `versionCode` for each release tag.
- Updates the draft fdroiddata metadata template to use `AI Chat` and
  unrestricted tag checks.
