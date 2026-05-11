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
        'copyProviderSetupUrl',
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
        'operatorStandbyTitle',
        'operatorStandbyStatus',
        'operatorStandbyRunHistory',
        'operatorStandbyRemoteDispatch',
        'operatorStandbyLastDispatch',
        'operatorStandbyLastRun',
    ]:
        assert key in strings

    assert 'strings.chatCommandsTip' in chat
    assert 'currentStrings()' in auth_view_model
    assert 'LaunchedEffect(strings.language)' in auth_screen
    assert 'strings.authRefreshDescription()' in auth_screen
    assert 'strings.authWaitingCallbackFor(uiState.pendingMethodLabel)' in auth_screen
    assert 'strings.deviceGuideTitle' in device
    assert 'OperatorStandbyCard' in device
    assert 'strings.operatorStandbyTitle()' in device
    assert 'strings.operatorStandbyStatus(' in device
    assert 'strings.operatorStandbyRemoteDispatch(' in device
    assert 'strings.operatorStandbyLastDispatch(' in device
    assert 'strings.toolProfileTitle' in tool_profile
    assert 'strings.providerLabel' in settings
    assert 'strings.localDownloadsExampleGuidance()' in downloads_section
    assert 'strings.downloadManagerReliabilityDescription()' in downloads_section
    assert 'LaunchedEffect(strings.language)' in portal
    assert 'strings.portalLoadingStatus' in portal


def test_settings_backend_toggles_sync_with_download_runtime_target_controls():
    settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    downloads_section = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")

    assert 'selectedBackend = uiState.onDeviceBackend' in settings
    assert 'onRuntimeFlavorSelected = viewModel::syncOnDeviceBackendWithRuntimeFlavor' in settings
    assert 'onCompletedDownloadReady = viewModel::startLocalRuntimeForFlavor' in settings
    assert 'LaunchedEffect(selectedBackend)' in downloads_section
    assert 'pendingAutoStartRecordId' in downloads_section
    assert 'onRuntimeFlavorSelected(completed.runtimeFlavor)' in downloads_section
    assert 'onCompletedDownloadReady(completed.runtimeFlavor)' in downloads_section
    assert 'fun syncOnDeviceBackendWithRuntimeFlavor(' in settings_view_model
    assert 'fun startLocalRuntimeForFlavor(' in settings_view_model
    assert 'fun syncSelectedBackend(' in downloads_view_model
    assert 'fun startRecommendedModelDownload(' in downloads_view_model
    assert 'fun promoteDownloadedModelForAutoStart(' in downloads_view_model
    assert 'AppSettingsStore(application)' in downloads_view_model


def test_mobile_repo_guidance_and_runtime_switches_keep_download_copy_in_sync():
    downloads_section = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")
    litert_proxy = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/LiteRtLmOpenAiProxy.kt").read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "android/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")

    assert 'strings.localDownloadsExampleGuidance()' in downloads_section
    assert 'strings.quickLocalModelsTitle()' in downloads_section
    assert 'strings.downloadAndStart()' in downloads_section
    assert 'inspectionStatus = ""' in downloads_view_model
    assert 'candidateSummary = ""' in downloads_view_model
    assert 'runtimeFlavorOverride' in downloads_view_model
    assert 'RecommendedLocalModelPreset' in downloads_view_model
    assert 'qwen35-08b-q4km-gguf' in downloads_view_model
    assert 'revision = "6e5c4f1e395deb959c494953478fa5cec4b8008f"' in downloads_view_model
    assert 'Edge Gallery 1.0.13 MTP path' in downloads_view_model
    assert 'restartDownloadOnMobileData(' in downloads_view_model
    assert 'Enter any Hugging Face repo' in strings
    assert 'One-tap local models' in strings
    assert 'selectRepoFileForDownload(' in download_manager
    assert 'findCompatibleRepoFile' in download_manager
    assert 'findFallbackRepoFile' in download_manager
    assert 'compatibilityHintForFile' in download_manager
    assert 'does not publish a native LiteRT-LM artifact' in download_manager
    assert 'does not publish a .litertlm or .task file' in download_manager
    assert 'litert-community/gemma-4-E2B-it-litert-lm' in download_manager
    assert 'litert-community/gemma-4-E4B-it-litert-lm' in download_manager
    assert 'litert-community/Gemma3-1B-IT' in download_manager
    assert 'litert-community/Gemma3-4B-IT' in download_manager
    assert 'Downloading is allowed; the selected backend will decide at load time whether it can run this file.' in download_manager
    assert 'Backend.GPU() to "gpu"' in litert_proxy
    assert 'Backend.CPU() to "cpu"' in litert_proxy
    assert 'put("accelerator", runtimeBackendLabel)' in litert_proxy
    assert 'com.google.ai.edge.litertlm:litertlm-android:0.11.0' in gradle
    assert 'ExperimentalFlags.enableSpeculativeDecoding' in litert_proxy
    assert 'speculativeDecodingDecision(modelPath)' in litert_proxy
    assert 'ExperimentalFlags.enableSpeculativeDecoding = false' in litert_proxy
    assert litert_proxy.index('ExperimentalFlags.enableSpeculativeDecoding = enableMtp') < litert_proxy.index('candidate = Engine(')
    assert 'disabled: Gemma 4 MTP failed during $label engine initialization; retried without MTP' in litert_proxy
    assert 'Build.SUPPORTED_ABIS.any { it.startsWith("x86") }' in litert_proxy
    assert 'Capabilities(modelPath).use' in litert_proxy
    assert 'capabilities.hasSpeculativeDecodingSupport()' in litert_proxy
    assert 'chatTemplateExtraContext(requestJson)' in litert_proxy
    assert 'conversation.sendMessage(promptMessage, extraContext)' in litert_proxy
    assert 'put("speculative_decoding", engineInitResult.speculativeDecoding)' in litert_proxy
    assert 'put("speculative_decoding_supported", engineInitResult.speculativeDecodingSupported)' in litert_proxy
    assert 'put("mtp_policy", engineInitResult.speculativeDecodingPolicy)' in litert_proxy
    assert 'put("gpu_policy", engineInitResult.gpuPolicy)' in litert_proxy
    assert 'ARM Qualcomm/Adreno' in litert_proxy
    assert 'attempting LiteRT-LM GPU with CPU fallback even though OpenCL probe was not loadable' in litert_proxy
    assert 'libOpenCL.so' in manifest
    assert 'libvndksupport.so' in manifest


def test_android_linux_subsystem_reapplies_executable_bits_before_reusing_cached_prefix():
    bridge = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesLinuxSubsystemBridge.kt").read_text(encoding="utf-8")
    linux_subsystem = (REPO_ROOT / "hermes_android/linux_subsystem.py").read_text(encoding="utf-8")
    android_environment = (REPO_ROOT / "tools/environments/android_linux.py").read_text(encoding="utf-8")

    cached_state_block = bridge.split('readState(context)?.let { state ->', 1)[1]

    assert 'state.optString("execution_mode") == SYSTEM_SHELL_MODE' not in cached_state_block
    assert 'state.optString("native_library_dir") != currentNativeLibraryDir' in cached_state_block
    assert 'markExecutableTree(File(prefixDir, "bin"))' in cached_state_block
    assert 'markExecutableTree(File(prefixDir, "libexec"))' in cached_state_block
    assert 'launchShellProbe(shellPath, homeDir, buildRunEnvironment(state)).ready' in cached_state_block
    assert 'reset(context)' in cached_state_block
    assert '"HERMES_ANDROID_SHELL" to SYSTEM_SHELL_PATH' in bridge
    assert '"HERMES_ANDROID_LINUX_NATIVE_BASH" to state.optString("shell_path")' in bridge
    assert '"HERMES_ANDROID_NATIVE_LIB"' in linux_subsystem
    assert '"HERMES_ANDROID_ALLOW_PREFIX_BIN": ""' in linux_subsystem
    assert '"LD_LIBRARY_PATH": ld_library_path' in linux_subsystem
    assert 'self.execution_mode = os.environ.get("HERMES_ANDROID_EXECUTION_MODE", "android_system_shell").strip()' in android_environment
    assert 'run_env["LD_LIBRARY_PATH"]' in android_environment
    assert 'path_parts = [system_path]' in android_environment


def test_android_intent_bridge_can_open_generated_workspace_html_with_fileprovider():
    manifest = (REPO_ROOT / "android/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")
    paths = (REPO_ROOT / "android/app/src/main/res/xml/hermes_file_paths.xml").read_text(encoding="utf-8")
    intent_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesIntentBridge.kt"
    ).read_text(encoding="utf-8")
    automation_test = (
        REPO_ROOT / "android/app/src/androidTest/java/com/nousresearch/hermesagent/HermesAutomationInstrumentedTest.kt"
    ).read_text(encoding="utf-8")

    assert 'androidx.core.content.FileProvider' in manifest
    assert 'android:authorities="${applicationId}.files"' in manifest
    assert 'android:resource="@xml/hermes_file_paths"' in manifest
    assert 'name="hermes_home"' in paths
    assert 'path="hermes-home/"' in paths
    assert 'FileProvider.getUriForFile(' in intent_bridge
    assert 'Intent.FLAG_GRANT_READ_URI_PERMISSION' in intent_bridge
    assert '"html", "htm" -> "text/html"' in intent_bridge
    assert 'shouldAddBrowsableCategory(resolvedDataUri ?: intent.data)' in intent_bridge
    assert 'BROWSABLE_URI_SCHEMES = setOf("http", "https")' in intent_bridge
    assert 'fun intentAutomationCanOpenGeneratedHermesHtmlFileWhenBrowserIsAvailable()' in automation_test
    assert 'hermes-flappy-browser-smoke.html' in automation_test


def test_android_python_import_path_prefers_hermes_utils_before_chaquopy_requirements():
    python_path = (REPO_ROOT / "hermes_android/python_path.py").read_text(encoding="utf-8")
    config_bridge = (REPO_ROOT / "hermes_android/config_bridge.py").read_text(encoding="utf-8")
    server = (REPO_ROOT / "hermes_android/server.py").read_text(encoding="utf-8")
    bundled_assets = (REPO_ROOT / "hermes_android/bundled_assets.py").read_text(encoding="utf-8")
    gateway_config = (REPO_ROOT / "gateway/config.py").read_text(encoding="utf-8")
    shared_utils = (REPO_ROOT / "hermes_cli/shared_utils.py").read_text(encoding="utf-8")
    native_smoke = (
        REPO_ROOT
        / "android/app/src/androidTest/java/com/nousresearch/hermesagent/NativeAgentRuntimeSmokeTest.kt"
    ).read_text(encoding="utf-8")

    assert "def prefer_hermes_package_root()" in python_path
    assert 'not hasattr(loaded_utils, "atomic_replace")' in python_path
    assert 'sys.modules.pop("utils", None)' in python_path
    assert "prefer_hermes_package_root()" in config_bridge
    assert "prefer_hermes_package_root()" in server
    assert "prefer_hermes_package_root()" in bundled_assets
    assert "def atomic_replace(" in shared_utils
    assert "from hermes_cli.shared_utils import is_truthy_value" in gateway_config
    assert 'shellPath.endsWith("/libhermes_android_bash.so")' in native_smoke


def test_hugging_face_inspect_download_flow_runs_off_main_thread_and_supports_repo_page_resolution():
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")

    assert 'Dispatchers.IO' in downloads_view_model
    assert 'withContext(Dispatchers.IO)' in downloads_view_model
    assert 'selectRepoFileForDownload' in download_manager
    assert 'findCompatibleRepoFile' in download_manager
    assert 'findFallbackRepoFile' in download_manager
    assert 'api/models/' in download_manager
    assert 'Unable to infer a downloadable model artifact' in download_manager
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


def test_android_automation_exposes_operator_standby_history_for_remote_dispatch():
    bridge = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesAutomationBridge.kt").read_text(encoding="utf-8")
    store = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesAutomationStore.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceViewModel.kt").read_text(encoding="utf-8")
    device = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt").read_text(encoding="utf-8")

    assert 'HermesAutomationRunEvent' in store
    assert 'KEY_RUN_EVENTS' in store
    assert 'addRunEvent' in store
    assert 'listRunEvents' in store
    assert 'operator_standby_status' in bridge
    assert 'operator_devices' in bridge
    assert 'OpenGUI devices' in bridge
    assert 'compatible_device_queries' in bridge
    assert 'operator_execution_status' in bridge
    assert 'operator_command' in bridge
    assert 'parseOperatorCommand' in bridge
    assert 'operatorCommandTextFromArguments' in bridge
    assert 'openguiSlashCommandTextFromArguments' in bridge
    assert 'slash_subcommand' in bridge
    assert '/opengui <subcommand>' in bridge
    assert '"/opengui"' in bridge
    assert 'OPENGUI_COMPATIBLE_COMMAND_HELP' in bridge
    assert 'OPENGUI_SLASH_COMMANDS' in bridge
    assert 'operatorCommandAccess' in bridge
    assert 'allowed_guild_ids' in bridge
    assert 'allowed_channel_ids' in bridge
    assert 'allowed_user_ids' in bridge
    assert 'not_allowed' in bridge
    assert 'slash_command_schema' in bridge
    assert '!opengui' in bridge
    assert '/run <id>' in bridge
    assert 'opengui_im_command' in bridge
    assert 'OpenGUI /status [executionId]' in bridge
    assert 'run_history' in bridge
    assert 'run_remote_dispatch' in bridge
    assert 'submit_standby_dispatch' in bridge
    assert 'TRIGGER_REMOTE_DISPATCH' in bridge
    assert 'dispatchPayloadFromArguments' in bridge
    assert 'recordRemoteDispatchFailure' in bridge
    assert 'REMOTE_DISPATCH_FAILURE_AUTOMATION_ID' in bridge
    assert 'remoteDispatchFailureJson' in bridge
    assert 'No Android automation matched remote dispatch' in bridge
    assert 'standby:register' in bridge
    assert 'standby:heartbeat' in bridge
    assert 'standby:dispatch' in bridge
    assert 'device_name' in bridge
    assert 'heartbeat_interval_seconds' in bridge
    assert 'DISPATCH_EXECUTION_ID' in bridge
    assert 'standby_dispatch' in bridge
    assert 'supported_dispatch_channels' in bridge
    assert '"discord"' in bridge
    assert '"telegram"' in bridge
    assert '"feishu"' in bridge
    assert '"rest"' in bridge
    assert 'external_broadcast' in bridge
    assert 'OpenGUI standby payload' in bridge
    assert 'compatible_dispatch_payloads' in bridge
    assert 'remote_dispatch_count' in bridge
    assert 'last_dispatch_task_name' in bridge
    assert 'Tasker plugin' in bridge
    assert 'operator_devices' in chat_client
    assert 'operator_command' in chat_client
    assert 'IM command strings such as !opengui devices' in chat_client
    assert '/opengui devices' in chat_client
    assert 'raw slash payloads' in chat_client
    assert 'standby device listing' in chat_client
    assert 'automationStandbyStatus' in view_model
    assert 'operatorStandbyReady' in view_model
    assert 'externalTriggerCount' in view_model
    assert 'remoteDispatchCount' in view_model
    assert 'lastDispatchTaskName' in view_model
    assert 'OperatorStandbyCard(uiState = uiState)' in device
