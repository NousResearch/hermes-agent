# Android Alpha.9 Portal + Local Model Downloads Plan

> For Hermes: use subagent-driven-development to implement this plan task-by-task.

Goal: add a proper fullscreen/minimize affordance for the embedded Nous Portal page, introduce a resilient Hugging Face/local-model download manager with data-saver behavior and progress persistence, and improve wide-screen support across key Android pages.

Architecture: keep the current Compose shell and Android local runtime, add a Settings-hosted model-download hub backed by Android DownloadManager + persistent metadata, expose data-saver settings through AppSettings, and polish Portal/Auth/Chat/Device/Settings layouts with centered max-width content on wider devices.

Tech stack: Jetpack Compose Material3, Android DownloadManager, SharedPreferences/EncryptedSharedPreferences, existing Hermes Android settings/runtime stores, and existing release/CI Android workflow.
