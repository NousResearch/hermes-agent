from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_localization_layer_covers_visible_chat_auth_portal_device_and_settings_copy():
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")
    chat = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    auth_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthViewModel.kt").read_text(encoding="utf-8")
    device = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt").read_text(encoding="utf-8")
    tool_profile = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/ToolProfileCard.kt").read_text(encoding="utf-8")
    settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
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
    ]:
        assert key in strings

    assert 'strings.chatCommandsTip' in chat
    assert 'currentStrings()' in auth_view_model
    assert 'strings.deviceGuideTitle' in device
    assert 'strings.toolProfileTitle' in tool_profile
    assert 'strings.providerLabel' in settings
    assert 'strings.portalLoadingStatus' in portal


def test_settings_backend_toggles_sync_with_download_runtime_target_controls():
    settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    downloads_section = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")

    assert 'selectedBackend = uiState.onDeviceBackend' in settings
    assert 'onRuntimeFlavorSelected = viewModel::syncOnDeviceBackendWithRuntimeFlavor' in settings
    assert 'LaunchedEffect(selectedBackend)' in downloads_section
    assert 'onRuntimeFlavorSelected("GGUF")' in downloads_section
    assert 'onRuntimeFlavorSelected("LiteRT-LM")' in downloads_section
    assert 'fun syncOnDeviceBackendWithRuntimeFlavor(' in settings_view_model
    assert 'AppSettingsStore(application)' in downloads_view_model


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
