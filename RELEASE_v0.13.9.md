# Hermes Agent v0.13.9

This release simplifies Android Settings around local inference and prepares the project for F-Droid submission work.

- Replaces Settings provider/model dropdowns with one-tap controls.
- Adds validated local model cards for Qwen3.5 0.8B GGUF, Gemma 4 E2B LiteRT-LM, and Gemma 3 1B LiteRT-LM.
- Queues model downloads through Android DownloadManager, marks completed model files as preferred, and starts the Hermes local runtime automatically.
- Keeps a remote provider fallback as explicit buttons and text fields instead of dropdown menus.
- Adds upstream Fastlane metadata and an F-Droid metadata template to make official F-Droid submission preparation concrete.
