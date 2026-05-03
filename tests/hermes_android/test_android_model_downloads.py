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

    assert 'Saved Hugging Face token for private or gated model downloads' in downloads_view_model
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
    assert 'findCompatibleRepoFile' in download_manager
    assert 'findFallbackRepoFile' in download_manager
    assert 'does not publish a .litertlm or .task file' in download_manager
    assert 'mobile-ready repo' in download_manager
    assert 'selectRepoFileForDownload(' in download_manager
    assert 'Downloading is allowed; the selected backend will decide at load time whether it can run this file.' in download_manager
    assert 'Paused because Android treats the current connection as roaming' in download_manager
    assert 'larger than your phone RAM' in download_manager
    assert 'supportsResume' in download_store


def test_model_catalog_prefers_verified_sub_5gb_litert_lm_mobile_models():
    catalog = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/ModelManagerViewModel.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")

    assert 'gemma-4-e2b-litert-lm' in catalog
    assert 'litert-community/gemma-4-E2B-it-litert-lm' in catalog
    assert '2_583_085_056' in catalog
    assert 'gemma-4-e4b-litert-lm' in catalog
    assert 'litert-community/gemma-4-E4B-it-litert-lm' in catalog
    assert '3_654_467_584' in catalog
    assert 'gemma-3-1b-it-litert-lm' in catalog
    assert 'litert-community/Gemma3-1B-IT' in catalog
    assert 'gemma-3-4b-it-vision-task' in catalog
    assert 'supportsImageInput = true' in catalog
    assert 'google/gemma-3n-E2B-it-litert-lm' in catalog
    assert 'google/gemma-3n-E4B-it-litert-lm' in catalog
    assert 'qwen3-0-6b-litert-lm' in catalog
    assert 'litert-community/Qwen3-0.6B' in catalog
    assert '614_236_160' in catalog
    assert 'qwen2-5-1-5b-instruct-litert-lm' in catalog
    assert 'phi-4-mini-instruct-litert-lm' not in catalog
    assert 'lower.endsWith(".litertlm") ||' in download_manager
    assert 'lower.endsWith(".task") && !isLiteRtWebTaskArtifact(lower)' in download_manager
    assert 'isLiteRtWebTaskArtifact(lower) -> Int.MAX_VALUE' in download_manager
    assert 'lower.endsWith(".litertlm") -> 0' in download_manager
    assert '"q4" in lower || "int4" in lower -> 0' in download_manager
    assert '"q8" in lower || "int8" in lower -> 1' in download_manager
    assert '"f32" in lower || "float32" in lower -> 20' in download_manager
    assert 'LiteRT-LM file may need extra RAM and cache space' in download_manager


def test_local_model_download_ui_mentions_hugging_face_progress_resume_and_mobile_restart_guidance():
    downloads_ui = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")

    assert 'strings.localDownloadsExampleGuidance()' in downloads_ui
    assert 'strings.downloadManagerReliabilityDescription()' in downloads_ui
    assert 'strings.localDownloadStatusLine(item.runtimeFlavor, item.statusLabel)' in downloads_ui
    assert 'strings.restartOnMobileData()' in downloads_ui
    assert 'strings.openSystemDownloads()' in downloads_ui
    assert 'Enter any Hugging Face repo' in strings
    assert 'Qwen/Qwen2.5-1.5B-Instruct-GGUF' in strings
    assert 'litert-community/Phi-4-mini-instruct' in strings
    assert 'lets the selected backend decide whether it can load it' in strings
    assert 'Warning: this download is larger than your phone RAM' in download_manager


def test_on_device_backend_preflights_required_model_extensions_before_launching_runtime():
    backend_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/OnDeviceBackendManager.kt").read_text(encoding="utf-8")

    assert 'preferredCompletedDownload(context)' in backend_manager
    assert 'Download any repo or file and mark it as preferred' in backend_manager
    assert 'matchesBackendArtifact' in backend_manager
    assert 'incompatiblePreferredDownloadStatus' in backend_manager
    assert 'lower.endsWith(".gguf")' in backend_manager
    assert 'isLiteRtLmArtifactPath(lower)' in backend_manager
    assert 'web/browser .task FlatBuffer' in backend_manager
    assert '.litertlm or .task' in backend_manager
    assert 'Download a $requiredExtension artifact and mark it as preferred first.' in backend_manager


def test_litert_runtime_rejects_web_task_flatbuffers_before_engine_start():
    proxy = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/LiteRtLmOpenAiProxy.kt").read_text(encoding="utf-8")

    assert 'validateModelArtifact(modelPath)' in proxy
    assert "header[4] == 'T'.code.toByte()" in proxy
    assert "header[7] == '3'.code.toByte()" in proxy
    assert 'web/browser .task FlatBuffer' in proxy
    assert 'download the .litertlm artifact instead' in proxy


def test_litert_proxy_bounds_generation_with_executor_timeout_and_cancel():
    proxy = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/LiteRtLmOpenAiProxy.kt").read_text(encoding="utf-8")

    assert 'Executors.newSingleThreadExecutor()' in proxy
    assert 'conversation.sendMessage(promptMessage, emptyMap())' in proxy
    assert 'future.get(timeoutMs, TimeUnit.MILLISECONDS)' in proxy
    assert 'generationTimeoutMs(requestJson)' in proxy
    assert 'conversation.cancelProcess()' in proxy
    assert 'executor.shutdownNow()' in proxy
    assert 'LiteRT-LM generation timed out after' in proxy


def test_litert_proxy_skips_gpu_when_opencl_is_missing():
    proxy = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/LiteRtLmOpenAiProxy.kt").read_text(encoding="utf-8")

    assert 'val openClAvailable = hasLoadableOpenClLibrary()' in proxy
    assert 'isTranslatedArm64OnX86(context) || !openClAvailable' in proxy
    assert '"/vendor/lib64/libOpenCL.so"' in proxy
    assert '"/system/vendor/lib64/libOpenCL.so"' in proxy
    assert 'System.load(file.absolutePath)' in proxy
    assert 'visionBackend = visionBackend' in proxy
    assert 'else -> "cpu"' in proxy


def test_release_build_recovers_existing_model_files_without_run_as_access():
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")
    backend_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/OnDeviceBackendManager.kt").read_text(encoding="utf-8")

    assert 'importExistingModelFiles(' in download_manager
    assert 'repairPreferredDownload(store, refreshed)' in download_manager
    assert 'downloadManagerId = -1L' in download_manager
    assert 'Imported existing model file from disk' in download_manager
    assert 'lower.endsWith(".litertlm") && "gemma-4" in lower -> 0' in download_manager
    assert 'lower.endsWith(".task") && !isLiteRtWebTaskArtifact(lower)' in download_manager
    assert 'HermesModelDownloadManager.refreshDownloads(context, store)' in backend_manager


def test_portal_screen_exposes_fullscreen_and_minimize_controls():
    portal = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/portal/NousPortalScreen.kt").read_text(encoding="utf-8")

    assert 'Full screen portal' in portal
    assert 'Minimize portal' in portal
    assert 'ic_action_fullscreen' in portal
    assert 'ic_action_minimize' in portal
    assert 'isFullscreen' in portal
