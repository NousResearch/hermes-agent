from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_localization_layer_covers_visible_chat_auth_portal_device_and_settings_copy():
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")
    chat = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    auth_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthViewModel.kt").read_text(encoding="utf-8")
    auth_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthScreen.kt").read_text(encoding="utf-8")
    device = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt").read_text(encoding="utf-8")
    tool_profile = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/ToolProfileCard.kt").read_text(encoding="utf-8")
    settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    downloads_section = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    portal = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/portal/NousPortalScreen.kt").read_text(encoding="utf-8")

    for key in [
        'chatCommandsTip',
        'providerLabel',
        'baseUrlLabel',
        'modelLabel',
        'apiKeyLabel',
        'toolProfileTitle',
        'deviceGuideTitle',
        'portalLoadingStatus',
        'authNotSignedIn',
        'cancelPendingSignIn',
        'authRefreshDescription',
        'authWaitingCallbackFor',
        'localDownloadsExampleGuidance',
        'downloadManagerReliabilityDescription',
        'localDownloadStatusLine',
        'restartOnMobileData',
        'openSystemDownloads',
    ]:
        assert key in strings

    assert 'strings.chatCommandsTip' in chat
    assert 'currentStrings()' in auth_view_model
    assert 'strings.authRefreshDescription()' in auth_screen
    assert 'strings.authWaitingCallbackFor(uiState.pendingMethodLabel)' in auth_screen
    assert 'strings.deviceGuideTitle' in device
    assert 'strings.toolProfileTitle' in tool_profile
    assert 'strings.providerLabel' in settings
    assert 'strings.localDownloadsExampleGuidance()' in downloads_section
    assert 'strings.downloadManagerReliabilityDescription()' in downloads_section
    assert 'strings.portalLoadingStatus' in portal


def test_settings_backend_toggles_sync_with_download_runtime_target_controls():
    settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    downloads_section = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")

    assert 'selectedBackend = uiState.onDeviceBackend' in settings
    assert 'onRuntimeFlavorSelected = viewModel::syncOnDeviceBackendWithRuntimeFlavor' in settings
    assert 'LaunchedEffect(selectedBackend)' in downloads_section
    assert 'effectiveRuntimeFlavor' in downloads_section
    assert 'onRuntimeFlavorSelected("GGUF")' in downloads_section
    assert 'onRuntimeFlavorSelected("LiteRT-LM")' in downloads_section
    assert 'fun syncOnDeviceBackendWithRuntimeFlavor(' in settings_view_model
    assert 'fun syncSelectedBackend(' in downloads_view_model
    assert 'AppSettingsStore(application)' in downloads_view_model


def test_mobile_repo_guidance_and_runtime_switches_keep_download_copy_in_sync():
    downloads_section = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")
    litert_proxy = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/LiteRtLmOpenAiProxy.kt").read_text(encoding="utf-8")

    assert 'strings.localDownloadsExampleGuidance()' in downloads_section
    assert 'runtimeFlavorOverride = effectiveRuntimeFlavor' in downloads_section
    assert 'inspectionStatus = ""' in downloads_view_model
    assert 'candidateSummary = ""' in downloads_view_model
    assert 'runtimeFlavorOverride' in downloads_view_model
    assert 'restartDownloadOnMobileData(' in downloads_view_model
    assert 'litert-community/gemma-4-E2B-it-litert-lm' in download_manager
    assert 'litert-community/gemma-4-E4B-it-litert-lm' in download_manager
    assert 'litert-community/Qwen2.5-1.5B-Instruct' in download_manager
    assert 'litert-community/DeepSeek-R1-Distill-Qwen-1.5B' in download_manager
    assert 'litert-community/Phi-4-mini-instruct' in download_manager
    assert 'Qwen/Qwen2.5-1.5B-Instruct-GGUF' in download_manager
    assert 'bartowski/microsoft_Phi-4-mini-instruct-GGUF' in download_manager
    assert 'llama.cpp + GGUF for Nemotron / Cascade families' in download_manager
    assert 'Backend.GPU() to "gpu"' in litert_proxy
    assert 'Backend.CPU() to "cpu"' in litert_proxy
    assert 'put("accelerator", runtimeBackendLabel)' in litert_proxy


def test_hugging_face_inspect_download_flow_runs_off_main_thread_and_supports_repo_page_resolution():
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")

    assert 'Dispatchers.IO' in downloads_view_model
    assert 'withContext(Dispatchers.IO)' in downloads_view_model
    assert 'findCompatibleRepoFile' in download_manager
    assert 'api/models/' in download_manager
    assert 'No compatible' in download_manager
    assert 'huggingface.co/' in download_manager


def test_chat_composer_matches_round_ui_spec():
    chat = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")

    assert 'RoundedCornerShape(28.dp)' in chat
    assert 'shape = RoundedCornerShape(28.dp)' in chat


def test_device_backend_exposes_deeper_radio_control_actions_and_status():
    bridge = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesSystemControlBridge.kt").read_text(encoding="utf-8")
    device = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt").read_text(encoding="utf-8")
    state_writer = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/device/DeviceStateWriter.kt").read_text(encoding="utf-8")

    for action in [
        'open_mobile_network_settings',
        'open_data_usage_settings',
        'open_hotspot_settings',
        'open_airplane_mode_settings',
    ]:
        assert action in bridge

    assert 'airplaneModeEnabled' in bridge
    assert 'isActiveNetworkMetered' in bridge
    assert 'Cellular + radio controls' in device
    assert 'airplane_mode_enabled' in state_writer
