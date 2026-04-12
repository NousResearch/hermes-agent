from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_settings_screen_wires_local_model_download_section_and_data_saver():
    settings_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    app_settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/AppSettingsStore.kt").read_text(encoding="utf-8")

    assert 'LocalModelDownloadsSection(' in settings_screen
    assert 'dataSaverMode = uiState.dataSaverMode' in settings_screen
    assert 'fun updateDataSaverMode(' in settings_view_model
    assert 'dataSaverMode' in app_settings
    assert 'KEY_DATA_SAVER_MODE' in app_settings


def test_local_model_download_view_model_and_store_support_resumable_download_state():
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")
    download_store = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/LocalModelDownloadStore.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")

    assert 'Saved Hugging Face token for gated model downloads' in downloads_view_model
    assert 'refreshDownloads()' in downloads_view_model
    assert 'restartDownloadOnMobileData(' in downloads_view_model
    assert 'ACTION_VIEW_DOWNLOADS' in downloads_view_model
    assert 'setPreferredDownload(' in downloads_view_model
    assert 'LocalModelDownloadRecord' in download_store
    assert 'preferred_download_id' in download_store
    assert 'allowMetered' in download_store
    assert 'allowRoaming' in download_store
    assert 'DownloadManager' in download_manager
    assert 'setAllowedOverMetered(allowMetered)' in download_manager
    assert 'setAllowedOverRoaming(allowRoaming)' in download_manager
    assert 'Restart on mobile data' in download_manager
    assert 'Paused because Android treats the current connection as roaming' in download_manager
    assert 'larger than your phone RAM' in download_manager
    assert 'supportsResume' in download_store


def test_local_model_download_ui_mentions_hugging_face_progress_resume_and_mobile_restart_guidance():
    downloads_ui = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")

    assert 'strings.localDownloadsExampleGuidance()' in downloads_ui
    assert 'strings.downloadManagerReliabilityDescription()' in downloads_ui
    assert 'strings.localDownloadStatusLine(item.runtimeFlavor, item.statusLabel)' in downloads_ui
    assert 'strings.restartOnMobileData()' in downloads_ui
    assert 'strings.openSystemDownloads()' in downloads_ui
    assert 'Qwen/Qwen2.5-1.5B-Instruct-GGUF' in strings
    assert 'litert-community/Phi-4-mini-instruct' in strings
    assert 'Google AI Edge Gallery currently relies on these curated LiteRT-LM repos' in strings
    assert 'Warning: this download is larger than your phone RAM' in download_manager


def test_portal_screen_exposes_fullscreen_and_minimize_controls():
    portal = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/portal/NousPortalScreen.kt").read_text(encoding="utf-8")

    assert 'Full screen portal' in portal
    assert 'Minimize portal' in portal
    assert 'ic_action_fullscreen' in portal
    assert 'ic_action_minimize' in portal
    assert 'isFullscreen' in portal
