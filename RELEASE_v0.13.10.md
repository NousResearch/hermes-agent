# Hermes Agent v0.13.10

This release adds a signed Cloudflare-backed model catalog for Android local inference and makes tested model setup a single dropdown flow in Settings.

- Adds a Cloudflare Worker model index with D1 storage, KV-published signed JSON, Hugging Face webhook ingestion, scheduled scans, and an authenticated HTTP GET scan fallback for cron-only environments.
- Discovers small mobile-friendly model files under 5 GiB, including Unsloth GGUF quantizations and LiteRT-LM Gemma builds.
- Adds Android signature verification for the catalog before any detected model is shown.
- Adds one Settings dropdown for detected downloadable models and a single action to download, mark preferred, and start the local runtime.
- Keeps Gemma 4 LiteRT-LM and Qwen3.5 0.8B GGUF ranked as first-class Android choices.
- Updates the F-Droid metadata template for the v0.13.10 release tag and Android version code.

Validation:

- Worker TypeScript check passed with `npm run check`.
- Android Kotlin compile, debug unit tests, and debug APK assembly passed with JDK 21, Android SDK 35, and Python 3.12.
- Emulator smoke test loaded the signed catalog in Settings and showed 62 downloadable model choices from the deployed worker.
