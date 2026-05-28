from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_localization_layer_covers_visible_chat_auth_portal_device_and_settings_copy():
    strings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")
    chat = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")
    auth_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/auth/AuthViewModel.kt").read_text(encoding="utf-8")
    auth_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/auth/AuthScreen.kt").read_text(encoding="utf-8")
    device = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/device/DeviceScreen.kt").read_text(encoding="utf-8")
    tool_profile = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/ToolProfileCard.kt").read_text(encoding="utf-8")
    settings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    downloads_section = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    portal = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/portal/NousPortalScreen.kt").read_text(encoding="utf-8")

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
        'portalInitialStatus',
        'portalBlockedByOfflineAirplaneMode',
        'portalReloadDescription',
        'portalHttpError',
        'portalEnabledLabel',
        'authNotSignedIn',
        'cancelPendingSignIn',
        'authRefreshDescription',
        'authWaitingCallbackFor',
        'localDownloadsExampleGuidance',
        'downloadManagerReliabilityDescription',
        'localDownloadStatusLine',
        'importModelFromPhoneFiles',
        'offlineAirplaneLocalModelsOnly',
        'recommendedLocalModelDescription',
        'recommendedLocalModelTestedLabel',
        'localModelUiText',
        'restartOnMobileData',
        'openSystemDownloads',
        'operatorStandbyTitle',
        'operatorStandbyStatus',
        'operatorStandbyRunHistory',
        'operatorStandbyRemoteDispatch',
        'operatorStandbyLastDispatch',
        'operatorStandbyLastRun',
        'appearancePresetLabel',
        'chatDisplayModeLabel',
        'userRoleLabel',
        'hermesPreparingReply',
        'attachmentPreviewUnavailable',
        'activityToolContext',
        'conversationHistoryTitle',
        'voiceInputLabel',
        'voiceRecognitionUnavailable',
        'chatStatusText',
        'endpointStatusIndicatorLabel',
        'endpointStatusTroubleshootingHint',
        'newChatActionDescription',
        'clearConversationActionDescription',
        'speakLastReplyActionDescription',
        'accountsActionDescription',
        'settingsActionDescription',
        'portalActionDescription',
        'deviceActionDescription',
    ]:
        assert key in strings

    assert 'strings.chatCommandsTip' in chat
    assert 'strings.newChatActionDescription()' in chat
    assert 'strings.clearConversationActionDescription()' in chat
    assert 'strings.speakLastReplyActionDescription()' in chat
    assert 'strings.accountsActionDescription()' in app_shell
    assert 'strings.settingsActionDescription()' in app_shell
    assert 'strings.portalActionDescription()' in app_shell
    assert 'strings.deviceActionDescription()' in app_shell
    assert 'currentStrings()' in auth_view_model
    assert 'LaunchedEffect(strings.language)' in auth_screen
    assert 'strings.authRefreshDescription()' in auth_screen
    assert 'strings.authWaitingCallbackFor(uiState.pendingMethodLabel)' in auth_screen
    assert 'strings.deviceGuideTitle' in device
    assert 'strings.deviceLinuxSuiteTitle()' in device
    assert 'strings.deviceConnectivityTitle()' in device
    assert 'strings.deviceRuntimeTitle()' in device
    assert 'strings.deviceWorkspaceAccessTitle()' in device
    assert 'strings.deviceAccessibilityTitle()' in device
    assert 'strings.deviceGlobalActionLabel(action)' in device
    assert 'OperatorStandbyCard' in device
    assert 'strings.operatorStandbyTitle()' in device
    assert 'strings.operatorStandbyStatus(' in device
    assert 'strings.operatorStandbyRemoteDispatch(' in device
    assert 'strings.operatorStandbyLastDispatch(' in device
    assert 'strings.toolProfileTitle' in tool_profile
    assert 'strings.providerLabel' in settings
    assert 'strings.providerDisplayLabel(preset.id, preset.label)' in settings
    assert 'strings.providerCredentialInputHelp(ProviderPresets.apiKeyEnvVars(providerId))' in settings
    assert 'strings.appearanceTitle()' in settings
    assert 'strings.appearancePresetLabel(preset.id, preset.label)' in settings
    assert 'strings.offlineAirplaneModeTitle()' in settings
    assert 'strings.compactPromptLabel(expanded)' in chat
    assert 'strings.chatDisplayModeLabel(chatDisplayMode)' in chat
    assert 'strings.userRoleLabel()' in chat
    assert 'strings.hermesPreparingReply()' in chat
    assert 'strings.activityToolContext()' in chat
    assert 'strings.conversationHistoryTitle()' in chat
    assert 'strings.voiceInputLabel()' in chat
    assert 'strings.chatCommandHelp()' in (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatCommandRouter.kt"
    ).read_text(encoding="utf-8")
    assert 'strings.localDownloadsExampleGuidance()' in downloads_section
    assert 'strings.downloadManagerReliabilityDescription()' in downloads_section
    assert 'strings.importModelFromPhoneFiles()' in downloads_section
    assert 'strings.recommendedLocalModelDescription(preset.id, preset.description)' in downloads_section
    assert 'strings.localModelUiText(uiState.workerCatalogStatus)' in downloads_section
    assert 'LaunchedEffect(strings.language)' in portal
    assert 'strings.portalLoadingStatus' in portal
    assert 'strings.portalReloadDescription()' in portal
    assert 'strings.portalEnabledLabel()' in portal
    assert 'strings.inferenceLabel(inferenceUrl)' in portal


def test_screenshot_reported_custom_endpoint_i18n_and_ime_layout_regressions_are_guarded():
    strings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")
    settings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    chat = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "android/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")

    for key in [
        "providerDisplayLabel",
        "providerCredentialInputHelp",
        "appearanceTitle",
        "offlineAirplaneToggleLabel",
        "settingsSavedBackendRestarted",
        "chatCommandHelp",
        "compactPromptLabel",
        "recommendedLocalModelDescription",
        "localModelUiText",
        "appearancePresetLabel",
        "chatDisplayModeLabel",
        "userRoleLabel",
        "attachmentPreviewUnavailable",
        "conversationHistoryTitle",
        "voiceInputLabel",
        "voiceRecognitionUnavailable",
        "chatStatusText",
        "endpointStatusIndicatorLabel",
        "endpointStatusTroubleshootingHint",
        "portalInitialStatus",
        "portalReloadDescription",
        "newChatActionDescription",
        "accountsActionDescription",
    ]:
        assert key in strings

    assert 'providerId.trim().lowercase()) {\n            "custom"' in strings
    assert 'AppLanguage.CHINESE -> "自定义 OpenAI 兼容端点"' in strings
    assert 'customEndpointConnectionHint' in strings
    assert 'If the stream closes early' in strings
    assert 'AppLanguage.CHINESE -> "分支"' in strings
    assert 'alphaBadge = "预览版"' in strings
    assert 'sectionPortal = "门户"' in strings
    assert 'portalTitle = "提供商门户"' in strings
    assert 'val selectedProviderLabel = strings.providerDisplayLabel(' in settings
    assert 'uiState.provider,' in settings
    assert 'strings.appearancePresetLabel(preset.id, preset.label)' in settings
    assert 'strings.providerCredentialInputHelp(ProviderPresets.apiKeyEnvVars(providerId))' in settings
    assert 'if (providerId == "custom")' in settings
    assert 'strings.customEndpointConnectionHint()' in settings
    assert 'if (provider.isBlank())' in settings_view_model
    assert 'provider == "custom"' not in settings_view_model.split("private fun loadApiKeyForProvider", 1)[1].split("fun updateOnDeviceBackend", 1)[0]
    assert 'strings.settingsSavedBackendRestarted()' in settings_view_model
    assert 'android:windowSoftInputMode="adjustResize"' in manifest
    assert 'val contentPadding = if (tinyRuntimeViewport)' in chat
    assert 'PaddingValues(horizontal = 4.dp, vertical = 4.dp)' in chat
    assert 'PaddingValues(horizontal = 12.dp, vertical = 8.dp)' in chat
    assert '.widthIn(max = 960.dp)\n                        .padding(contentPadding),' in chat
    assert 'adding imePadding here double-lifts the composer on phones' in chat
    assert '.heightIn(max = 112.dp)\n            .testTag("HermesChatInput")' in chat
    assert 'maxLines = 4' in chat
    assert '.imePadding()' not in chat
    assert 'strings = strings' in chat
    assert '.testTag("HermesChatComposerFrame")' in chat
    assert '.testTag("HermesChatComposerCompact")' in chat
    assert 'val ultraNarrowComposer = maxWidth < 220.dp' in chat
    assert 'val stackedComposer = maxWidth < 340.dp' in chat
    assert 'UltraNarrowComposerSendButton(' in chat
    assert 'strings.chatDisplayModeLabel(chatDisplayMode)' in chat
    assert 'strings.chatStatusText(text)' in chat
    assert 'isEndpointStatusText(displayText)' in chat
    assert 'strings.endpointStatusIndicatorLabel()' in chat
    assert 'strings.endpointStatusTroubleshootingHint()' in chat
    assert 'strings.userRoleLabel()' in chat
    assert 'strings.attachmentPreviewUnavailable()' in chat
    assert 'strings.voiceRecognitionUnavailable()' in chat


def test_settings_backend_toggles_sync_with_download_runtime_target_controls():
    settings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    downloads_section = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")

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


def test_settings_secret_store_initialization_stays_off_startup_main_thread():
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    secure_store = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/data/SecureSecretsStore.kt").read_text(encoding="utf-8")
    initial_state_block = settings_view_model.split("private fun loadInitialState()", 1)[1].split("fun reload()", 1)[0]

    assert "private val secretsStore by lazy" in settings_view_model
    assert "apiKey = \"\"" in initial_state_block
    assert "defaultOnDeviceSummary(stored.onDeviceBackend)" in initial_state_block
    assert "preferredDownloadSummary" not in initial_state_block
    assert "refreshOnDeviceSummary(reloaded.onDeviceBackend)" in settings_view_model
    assert "refreshOnDeviceSummary(_uiState.value.onDeviceBackend)" not in settings_view_model
    assert "OnDeviceBackendManager.preferredDownloadSummary(getApplication(), backendValue)" in settings_view_model
    assert "withContext(Dispatchers.IO)" in settings_view_model.split("private fun refreshOnDeviceSummary", 1)[1]
    assert "loadApiKeyForProvider(_uiState.value.provider)" in settings_view_model
    assert "withContext(Dispatchers.IO)" in settings_view_model.split("private fun loadApiKeyForProvider", 1)[1]
    assert "EncryptedSharedPreferences.create(" in secure_store
    assert "private val preferences by lazy" in secure_store
    assert "MasterKey.Builder(appContext)" in secure_store


def test_mobile_repo_guidance_and_runtime_switches_keep_download_copy_in_sync():
    downloads_section = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/LocalModelDownloadsSection.kt").read_text(encoding="utf-8")
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")
    settings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    app_settings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/data/AppSettingsStore.kt").read_text(encoding="utf-8")
    backend_manager = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/backend/OnDeviceBackendManager.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")
    litert_proxy = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/backend/LiteRtLmOpenAiProxy.kt").read_text(encoding="utf-8")
    hardware_profile = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesAndroidHardwareProfile.kt").read_text(encoding="utf-8")
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
    assert 'revision = "7fa1d78473894f7e736a21d920c3aa80f950c0db"' in downloads_view_model
    assert 'gemma4-e4b-litert-lm' in downloads_view_model
    assert 'revision = "9695417f248178c63a9f318c6e0c56cb917cb837"' in downloads_view_model
    assert 'revisionMatches' in downloads_view_model
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
    assert 'LITERT_ALIAS_REVISIONS' in download_manager
    assert '7fa1d78473894f7e736a21d920c3aa80f950c0db' in download_manager
    assert '9695417f248178c63a9f318c6e0c56cb917cb837' in download_manager
    assert 'litert-community/gemma-4-E2B-it-litert-lm' in download_manager
    assert 'litert-community/gemma-4-E4B-it-litert-lm' in download_manager
    assert 'litert-community/Gemma3-1B-IT' in download_manager
    assert 'litert-community/Gemma3-4B-IT' in download_manager
    assert 'Downloading is allowed; the selected backend will decide at load time whether it can run this file.' in download_manager
    assert 'Backend.GPU() to "gpu"' in litert_proxy
    assert 'Backend.CPU() to "cpu"' in litert_proxy
    assert 'put("accelerator", runtimeBackendLabel)' in litert_proxy
    assert 'com.google.ai.edge.litertlm:litertlm-android:0.12.0' in gradle
    assert 'ExperimentalFlags.enableSpeculativeDecoding' in litert_proxy
    assert 'SpeculativeDecodingMode' in litert_proxy
    assert 'liteRtLmSpeculativeDecodingMode' in app_settings
    assert 'KEY_LITERT_LM_SPECULATIVE_DECODING_MODE' in app_settings
    assert 'customSystemPrompt' in app_settings
    assert 'MAX_CUSTOM_SYSTEM_PROMPT_CHARS' in app_settings
    assert 'KEY_CUSTOM_SYSTEM_PROMPT' in app_settings
    assert 'custom_system_prompt' in app_settings
    assert 'AgentPersonaCard' in settings
    assert 'AgentPersonaPrompt' in settings
    assert 'updateCustomSystemPrompt' in settings_view_model
    assert 'saveAgentPersona' in settings_view_model
    assert 'updateLiteRtLmSpeculativeDecodingMode' in settings_view_model
    assert 'LiteRtLmMtpMode-${choice.value}' in settings
    assert 'gemma4MtpDescription()' in settings
    assert 'speculativeDecodingModeFor(context)' in backend_manager
    assert 'inferenceConfigFor(preferred, speculativeDecodingModeFor(context))' in backend_manager
    assert 'speculativeDecodingDecision(context, modelPath, speculativeDecodingMode)' in litert_proxy
    assert 'decideSpeculativeDecoding(' in litert_proxy
    assert 'runtime setting disabled Gemma 4 MTP' in litert_proxy
    assert 'memory guard for Gemma 4 MTP' in litert_proxy
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
    assert 'put("gpu_policy", engineInitResult.gpuPolicy.description)' in litert_proxy
    assert 'put("gpu_attempted", engineInitResult.gpuPolicy.enabled)' in litert_proxy
    assert 'put("gpu_fallback_to_cpu", engineInitResult.gpuPolicy.enabled && engineInitResult.backend != "gpu")' in litert_proxy
    assert 'put("opencl_available", engineInitResult.gpuPolicy.openClAvailable)' in litert_proxy
    assert 'put("hardware_identity", engineInitResult.gpuPolicy.deviceIdentity)' in litert_proxy
    assert 'put("soc_family", engineInitResult.gpuPolicy.socFamily)' in litert_proxy
    assert 'put("gpu_family", engineInitResult.gpuPolicy.gpuFamily)' in litert_proxy
    assert 'put("litert_backend_order", JSONArray(engineInitResult.gpuPolicy.backendOrder))' in litert_proxy
    assert 'HermesAndroidHardwareProfile.classify' in litert_proxy
    assert '"mediatek" -> "MediaTek"' in hardware_profile
    assert '"qualcomm_snapdragon" -> "Qualcomm Snapdragon"' in hardware_profile
    assert '"powervr_img" -> "PowerVR/IMG"' in hardware_profile
    assert 'Adreno, Mali, Immortalis, Xclipse, and PowerVR/IMG' in hardware_profile
    assert 'attempting LiteRT-LM GPU with CPU fallback even though OpenCL probe was not loadable' in litert_proxy
    assert 'libOpenCL.so' in manifest
    assert 'libvndksupport.so' in manifest


def test_android_diagnostics_exposes_agent_environment_report_for_kai_parity():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")
    automation_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesAutomationBridge.kt").read_text(encoding="utf-8")

    assert '"agent_environment_report"' in diagnostics_bridge
    assert '"mcp_tool_server_registry_report"' in diagnostics_bridge
    assert '"agent_objective_coverage_report"' in diagnostics_bridge
    assert '"agent_upgrade_coverage_report"' in diagnostics_bridge
    assert '"hermes_upgrade_coverage_report"' in diagnostics_bridge
    assert '"agent_release_validation_report"' in diagnostics_bridge
    assert '"github_release_readiness_report"' in diagnostics_bridge
    assert '"release_validation_readiness_report"' in diagnostics_bridge
    assert '"agent_capability_upgrade_report"' in diagnostics_bridge
    assert 'mcpToolServerRegistryReportJson(appContext)' in diagnostics_bridge
    assert 'agentObjectiveCoverageReportJson(appContext)' in diagnostics_bridge
    assert 'agentReleaseValidationReportJson(appContext)' in diagnostics_bridge
    assert 'agentCapabilityUpgradeReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentObjectiveCoverageReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'fun agentReleaseValidationReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'fun agentCapabilityUpgradeReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentObjectiveCoverageRows(' in diagnostics_bridge
    assert 'agentObjectiveGapRows(' in diagnostics_bridge
    assert 'agentResearchParityRows()' in diagnostics_bridge
    assert 'agentReleaseValidationRows(' in diagnostics_bridge
    assert 'agentReleaseArtifactGateRows(' in diagnostics_bridge
    assert 'agentFdroidReleaseMetadataRows(' in diagnostics_bridge
    assert 'agentCapabilityUpgradeRows(' in diagnostics_bridge
    assert 'agentCapabilityUpgradeRouteRows()' in diagnostics_bridge
    assert 'fun mcpToolServerRegistryReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'mcpToolServerRegistryRows(' in diagnostics_bridge
    assert 'mcpToolServerRouteRows(' in diagnostics_bridge
    assert 'agentEnvironmentReportJson(appContext)' in diagnostics_bridge
    assert 'agentCapabilityMatrixRows(' in diagnostics_bridge
    assert 'kaiParityMatrixRows(' in diagnostics_bridge
    assert 'kaiOperationsMatrixRows(' in diagnostics_bridge
    assert 'workflowReadinessRows(' in diagnostics_bridge
    assert 'agentToolSandboxRows(' in diagnostics_bridge
    assert '"agent_capability_matrix"' in diagnostics_bridge
    assert '"kai_parity_matrix"' in diagnostics_bridge
    assert '"kai_operations_matrix"' in diagnostics_bridge
    assert '"workflow_readiness_matrix"' in diagnostics_bridge
    assert '"agent_tool_sandbox_matrix"' in diagnostics_bridge
    assert '"mcp_tool_server_registry"' in diagnostics_bridge
    assert '"mcp_tool_server_routes"' in diagnostics_bridge
    assert '"agent_objective_coverage_matrix"' in diagnostics_bridge
    assert '"agent_objective_gap_matrix"' in diagnostics_bridge
    assert '"agent_research_parity_matrix"' in diagnostics_bridge
    assert '"agent_release_validation_matrix"' in diagnostics_bridge
    assert '"agent_release_artifact_gates"' in diagnostics_bridge
    assert '"fdroid_release_metadata_matrix"' in diagnostics_bridge
    assert '"agent_upgrade_objective_matrix"' in diagnostics_bridge
    assert '"agent_upgrade_route_matrix"' in diagnostics_bridge
    assert 'Objective Coverage' in diagnostics_bridge
    assert 'Objective Gaps' in diagnostics_bridge
    assert 'Research Parity Map' in diagnostics_bridge
    assert 'https://github.com/SimonSchubert/Kai' in diagnostics_bridge
    assert 'https://github.com/VREMSoftwareDevelopment/WiFiAnalyzer' in diagnostics_bridge
    assert 'Release and CI proof' in diagnostics_bridge
    assert 'Release Validation' in diagnostics_bridge
    assert 'Release Artifact Gates' in diagnostics_bridge
    assert 'F-Droid Metadata Gates' in diagnostics_bridge
    assert 'android-release.yml' in diagnostics_bridge
    assert 'scripts/android_release_manifest.py' in diagnostics_bridge
    assert 'Fastlane graphics in tagged tree' in diagnostics_bridge
    assert 'Upgrade Objective Matrix' in diagnostics_bridge
    assert 'Upgrade Verification Routes' in diagnostics_bridge
    assert 'Full Hermes upgrade objective audit' in diagnostics_bridge
    assert 'MCP Tool Servers' in diagnostics_bridge
    assert 'MCP Routing Policy' in diagnostics_bridge
    assert 'Streamable HTTP MCP endpoint' in diagnostics_bridge
    assert 'Context7 documentation server' in diagnostics_bridge
    assert 'DeepWiki repository docs server' in diagnostics_bridge
    assert 'Globalping network probe server' in diagnostics_bridge
    assert 'CoinGecko market data server' in diagnostics_bridge
    assert 'Find-A-Domain server' in diagnostics_bridge
    assert 'mcp_streamable_http_supported' in diagnostics_bridge
    assert 'Tool Sandbox Status' in diagnostics_bridge
    assert 'Multi-provider priority and fallback' in diagnostics_bridge
    assert 'Tool and MCP bridge route' in diagnostics_bridge
    assert 'Native diagnostics tool surface' in diagnostics_bridge
    assert 'Terminal/Linux workspace surface' in diagnostics_bridge
    assert 'Privileged Android action surface' in diagnostics_bridge
    assert 'External MCP/server parity surface' in diagnostics_bridge
    assert 'sandbox_scope' in diagnostics_bridge
    assert 'remote_dispatch_capable' in diagnostics_bridge
    assert 'mcp_parity_status' in diagnostics_bridge
    assert 'Encrypted credentials and backup route' in diagnostics_bridge
    assert 'export_app_settings/import_app_settings' in diagnostics_bridge
    assert 'Customizable soul / system prompt' in diagnostics_bridge
    assert 'agent_persona_status' in diagnostics_bridge
    assert 'custom_system_prompt' in diagnostics_bridge
    assert 'TTS and image conversation route' in diagnostics_bridge
    assert 'Scheduled task compatibility route' in diagnostics_bridge
    assert 'schedule_task/list_tasks/cancel_task' in diagnostics_bridge
    assert 'kai_task_compat' in diagnostics_bridge
    assert 'background_ai_prompt_execution' in diagnostics_bridge
    assert 'Route Kai-style tool orchestration' in diagnostics_bridge
    assert 'Use SOC and LiteRT backend policy fields to avoid Snapdragon-only assumptions' in diagnostics_bridge
    assert 'Use hindsight_memory_tool and operator heartbeat/status rows' in diagnostics_bridge
    assert 'agent_environment_report' in chat_client
    assert 'mcp_tool_server_registry_report' in chat_client
    assert 'agent_objective_coverage_report' in chat_client
    assert 'agent_upgrade_coverage_report' in chat_client
    assert 'hermes_upgrade_coverage_report' in chat_client
    assert 'agent_release_validation_report' in chat_client
    assert 'github_release_readiness_report' in chat_client
    assert 'release_validation_readiness_report' in chat_client
    assert 'agent_capability_upgrade_report' in chat_client
    assert 'MCP tool-server registry reports' in chat_client
    assert 'objective coverage/gap and upgrade coverage reports' in chat_client
    assert 'release validation and GitHub release readiness reports for Android CI' in chat_client
    assert 'full upgrade objective audit reports' in chat_client
    assert 'export_app_settings/import_app_settings' in chat_client
    assert 'Kai-style custom agent persona/system prompt' in chat_client
    assert 'schedule_task/list_tasks/cancel_task' in chat_client
    assert 'not unrestricted background AI prompt execution' in chat_client
    assert 'name = "schedule_task"' in chat_client
    assert 'name = "list_tasks"' in chat_client
    assert 'name = "cancel_task"' in chat_client
    assert 'User-configured agent persona' in chat_client
    assert '"agent_capability_matrix"' in chat_client
    assert '"kai_parity_matrix"' in chat_client
    assert '"kai_operations_matrix"' in chat_client
    assert '"workflow_readiness_matrix"' in chat_client
    assert '"agent_tool_sandbox_matrix"' in chat_client
    assert '"mcp_tool_server_registry"' in chat_client
    assert '"mcp_tool_server_routes"' in chat_client
    assert '"agent_objective_coverage_matrix"' in chat_client
    assert '"agent_objective_gap_matrix"' in chat_client
    assert '"agent_research_parity_matrix"' in chat_client
    assert '"agent_release_validation_matrix"' in chat_client
    assert '"agent_release_artifact_gates"' in chat_client
    assert '"fdroid_release_metadata_matrix"' in chat_client
    assert '"agent_upgrade_objective_matrix"' in chat_client
    assert '"agent_upgrade_route_matrix"' in chat_client
    assert '"kai_operations_matrix"' in diagnostic_cards
    assert '"agent_tool_sandbox_matrix"' in diagnostic_cards
    assert '"mcp_tool_server_registry"' in diagnostic_cards
    assert '"mcp_tool_server_routes"' in diagnostic_cards
    assert '"agent_objective_coverage_matrix"' in diagnostic_cards
    assert '"agent_objective_gap_matrix"' in diagnostic_cards
    assert '"agent_research_parity_matrix"' in diagnostic_cards
    assert '"agent_release_validation_matrix"' in diagnostic_cards
    assert '"agent_release_artifact_gates"' in diagnostic_cards
    assert '"fdroid_release_metadata_matrix"' in diagnostic_cards
    assert '"agent_upgrade_objective_matrix"' in diagnostic_cards
    assert '"agent_upgrade_route_matrix"' in diagnostic_cards
    assert 'id = "mcp_registry"' in quick_actions
    assert 'diagnosticAction = "mcp_tool_server_registry_report"' in quick_actions
    assert 'id = "upgrade_audit"' in quick_actions
    assert 'diagnosticAction = "agent_capability_upgrade_report"' in quick_actions
    assert 'id = "objective_coverage"' in quick_actions
    assert 'diagnosticAction = "agent_objective_coverage_report"' in quick_actions
    assert 'id = "release_validation"' in quick_actions
    assert 'diagnosticAction = "agent_release_validation_report"' in quick_actions
    assert '"agent_capability_matrix", "kai_parity_matrix", "agent_workflow_readiness"' in diagnostic_cards
    assert 'capabilityMatrixRow(row)' in diagnostic_cards
    assert '"schedule_task", "kai_schedule_task" -> scheduleTaskJson(context, arguments)' in automation_bridge
    assert '"list_tasks", "kai_list_tasks", "tasks" -> listTasksJson(context)' in automation_bridge
    assert '"cancel_task", "kai_cancel_task" -> cancelTaskJson(context, arguments)' in automation_bridge
    assert '"android_automation_task_not_background_ai_prompt"' in automation_bridge
    assert '"android_automation_notification_task"' in automation_bridge


def test_android_diagnostics_exposes_agent_self_check_report_for_kai_heartbeat_and_signal_routes():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"agent_self_check_report"' in diagnostics_bridge
    assert 'agentSelfCheckReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSelfCheckReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSelfCheckMatrixRows(' in diagnostics_bridge
    assert 'agentSelfCheckRouteRows(' in diagnostics_bridge
    assert '"agent_self_check_matrix"' in diagnostics_bridge
    assert '"agent_self_check_routes"' in diagnostics_bridge
    assert 'Kai-style passive Hermes self-check' in diagnostics_bridge
    assert 'Kai-style heartbeat surface' in diagnostics_bridge
    assert 'Wi-Fi Analyzer card coverage' in diagnostics_bridge
    assert 'Bluetooth proximity card coverage' in diagnostics_bridge
    assert 'Motion sensor workflow coverage' in diagnostics_bridge
    assert 'Radio/RF boundary coverage' in diagnostics_bridge
    assert 'RF coexistence fusion' in diagnostics_bridge
    assert 'MediaTek/non-Adreno backend guard' in diagnostics_bridge
    assert 'Local inference compatibility' in diagnostics_bridge
    assert 'Tool sandbox and Kai operations' in diagnostics_bridge
    assert 'Expandable card manifest' in diagnostics_bridge
    assert 'Treat heartbeat rows as status surfacing' in diagnostics_bridge
    assert 'agent_self_check_report' in chat_client
    assert 'passive agent self-check/heartbeat reports' in chat_client
    assert '"agent_self_check_matrix"' in diagnostic_cards
    assert '"agent_self_check_routes"' in diagnostic_cards
    assert 'id = "agent_self_check"' in quick_actions
    assert 'diagnosticAction = "agent_self_check_report"' in quick_actions
    assert 'action=agent_self_check_report' in quick_actions


def test_android_diagnostics_exposes_agent_observation_dashboard_for_gemma_signal_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"agent_observation_report"' in diagnostics_bridge
    assert 'agentObservationReportJson(appContext)' in diagnostics_bridge
    assert '"agent_card_manifest_report"' in diagnostics_bridge
    assert 'agentCardManifestReportJson(appContext)' in diagnostics_bridge
    assert '"agent_card_priority_report"' in diagnostics_bridge
    assert 'agentCardPriorityReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentObservationReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'fun agentCardManifestReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'fun agentCardPriorityReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentObservationMatrixRows(' in diagnostics_bridge
    assert 'agentObservationRouteRows()' in diagnostics_bridge
    assert 'agentCardManifestRows(' in diagnostics_bridge
    assert 'agentCardManifestSources(' in diagnostics_bridge
    assert 'agentCardPriorityRows(' in diagnostics_bridge
    assert 'agentCardOpenSequenceRows()' in diagnostics_bridge
    assert 'kaiInteractiveScreenParityRows(' in diagnostics_bridge
    assert '"agent_observation_matrix"' in diagnostics_bridge
    assert '"agent_observation_routes"' in diagnostics_bridge
    assert '"agent_card_manifest"' in diagnostics_bridge
    assert '"top_signal_card_priorities"' in diagnostics_bridge
    assert '"agent_card_open_sequence"' in diagnostics_bridge
    assert '"kai_interactive_screen_parity"' in diagnostics_bridge
    assert '"gemma_card_planner_directives"' in diagnostics_bridge
    assert 'Agent Card Manifest' in diagnostics_bridge
    assert 'Top Signal Cards' in diagnostics_bridge
    assert 'Kai Interactive Parity' in diagnostics_bridge
    assert '"gemma_observation_directives"' in diagnostics_bridge
    assert '"accelerator_preflight_observation_summary"' in diagnostics_bridge
    assert '"accelerator_preflight_report"' in diagnostics_bridge
    assert '"accelerator_preflight_matrix"' in diagnostics_bridge
    assert '"non_adreno_backend_advisor_observation_summary"' in diagnostics_bridge
    assert '"non_adreno_backend_advisor_report"' in diagnostics_bridge
    assert '"non_adreno_backend_advisor_matrix"' in diagnostics_bridge
    assert 'Wi-Fi AP metadata and channel graphs' in diagnostics_bridge
    assert 'Bluetooth nearby metadata' in diagnostics_bridge
    assert 'Motion and sensor context' in diagnostics_bridge
    assert 'Radio and RF boundaries' in diagnostics_bridge
    assert 'Radio advisor and bridge cards' in diagnostics_bridge
    assert '"radio_signal_advisor_matrix"' in diagnostics_bridge
    assert '"radio_receiver_candidates"' in diagnostics_bridge
    assert 'radio_signal_advisor_report' in diagnostics_bridge
    assert 'Open non-Adreno backend advisor cards' in diagnostics_bridge
    assert 'Non-Adreno backend advisor route' in diagnostics_bridge
    assert 'Kai operations and interactive routes' in diagnostics_bridge
    assert 'agent_observation_report' in chat_client
    assert 'agent_card_manifest_report' in chat_client
    assert 'agent_card_priority_report' in chat_client
    assert '"agent_observation_matrix"' in chat_client
    assert '"agent_observation_routes"' in chat_client
    assert '"agent_card_manifest"' in chat_client
    assert '"top_signal_card_priorities"' in chat_client
    assert '"agent_card_open_sequence"' in chat_client
    assert '"kai_interactive_screen_parity"' in chat_client
    assert '"gemma_card_planner_directives"' in chat_client
    assert '"agent_card_graph_types"' in chat_client
    assert '"gemma_observation_directives"' in chat_client
    assert '"agent_observation_matrix", "agent_observation_routes"' in diagnostic_cards
    assert '"agent_card_manifest",' in diagnostic_cards
    assert '"agent_card_priority_matrix"' in diagnostic_cards
    assert '"agent_card_open_sequence"' in diagnostic_cards
    assert '"kai_interactive_screen_parity"' in diagnostic_cards
    assert 'id = "agent_observation"' in quick_actions
    assert 'action=agent_observation_report' in quick_actions
    assert 'id = "card_manifest"' in quick_actions
    assert 'action=agent_card_manifest_report' in quick_actions
    assert 'id = "top_cards"' in quick_actions
    assert 'action=agent_card_priority_report' in quick_actions


def test_android_diagnostics_exposes_agent_signal_briefing_for_first_read_top_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"agent_signal_briefing_report"' in diagnostics_bridge
    assert '"signal_briefing_report"' in diagnostics_bridge
    assert '"agent_signal_card_deck_report"' in diagnostics_bridge
    assert '"agent_signal_card_refresh_plan_report"' in diagnostics_bridge
    assert '"agent_signal_card_refresh_status_report"' in diagnostics_bridge
    assert '"expanded_signal_cards"' in diagnostics_bridge
    assert '"top_card_refresh_plan"' in diagnostics_bridge
    assert '"top_card_refresh_status"' in diagnostics_bridge
    assert 'agentSignalBriefingReportJson(appContext)' in diagnostics_bridge
    assert 'agentSignalCardDeckReportJson(appContext)' in diagnostics_bridge
    assert 'agentSignalCardRefreshPlanReportJson(appContext)' in diagnostics_bridge
    assert 'agentSignalCardRefreshStatusReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSignalBriefingReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'fun agentSignalCardDeckReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'fun agentSignalCardRefreshPlanReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'fun agentSignalCardRefreshStatusReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalBriefingRows(' in diagnostics_bridge
    assert 'agentSignalCardDeckRows(' in diagnostics_bridge
    assert 'agentSignalCardRefreshPlanRows(' in diagnostics_bridge
    assert 'agentSignalCardRefreshStatusRows(' in diagnostics_bridge
    assert 'agentTopCardSlotRows(' in diagnostics_bridge
    assert 'agentSignalMetadataKeyRows(' in diagnostics_bridge
    assert 'agentSignalBriefingSourceActions()' in diagnostics_bridge
    assert 'fun agentSignalTimelineReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalTimelineRows(' in diagnostics_bridge
    assert 'agentSignalTimelineRefreshRows()' in diagnostics_bridge
    assert '"agent_signal_briefing_matrix"' in diagnostics_bridge
    assert '"agent_signal_card_deck_manifest"' in diagnostics_bridge
    assert '"agent_signal_card_refresh_plan_matrix"' in diagnostics_bridge
    assert '"agent_signal_card_refresh_status_matrix"' in diagnostics_bridge
    assert '"agent_signal_timeline"' in diagnostics_bridge
    assert '"agent_signal_refresh_routes"' in diagnostics_bridge
    assert '"agent_top_card_slots"' in diagnostics_bridge
    assert '"agent_signal_metadata_keys"' in diagnostics_bridge
    assert '"gemma_signal_briefing_directives"' in diagnostics_bridge
    assert 'Agent Signal Briefing' in diagnostics_bridge
    assert 'Expanded Signal Card Deck' in diagnostics_bridge
    assert 'Signal Card Refresh Plan' in diagnostics_bridge
    assert 'Signal Card Refresh Status' in diagnostics_bridge
    assert 'ready_for_active_refresh' in diagnostics_bridge
    assert 'status_hint' in diagnostics_bridge
    assert 'Bluetooth Nearby Advisor' in diagnostics_bridge
    assert 'Bluetooth Device Candidates' in diagnostics_bridge
    assert 'Motion Sensor Workflow' in diagnostics_bridge
    assert 'Top Card Slots' in diagnostics_bridge
    assert 'Gemma Metadata Keys' in diagnostics_bridge
    assert 'Wi-Fi graph evidence' in diagnostics_bridge
    assert 'Bluetooth metadata evidence' in diagnostics_bridge
    assert 'Motion and sensor evidence' in diagnostics_bridge
    assert 'Radio boundary and bridge evidence' in diagnostics_bridge
    assert 'MediaTek and backend evidence' in diagnostics_bridge
    assert 'accelerator_preflight_report' in diagnostics_bridge
    assert 'accelerator_preflight_matrix' in diagnostics_bridge
    assert 'Refresh accelerator preflight' in diagnostics_bridge
    assert 'delegate preflight' in diagnostics_bridge
    assert 'agent_signal_briefing_report' in chat_client
    assert 'agent_signal_card_deck_report' in chat_client
    assert 'agent_signal_card_refresh_plan_report' in chat_client
    assert 'agent_signal_card_refresh_status_report' in chat_client
    assert 'agent_signal_timeline_report' in chat_client
    assert 'signal_briefing_report' in chat_client
    assert '"agent_signal_briefing_matrix"' in chat_client
    assert '"agent_signal_card_deck_manifest"' in chat_client
    assert '"agent_signal_card_refresh_plan_matrix"' in chat_client
    assert '"agent_signal_card_refresh_status_matrix"' in chat_client
    assert '"agent_signal_timeline"' in diagnostic_cards
    assert '"agent_signal_card_deck_manifest"' in diagnostic_cards
    assert '"agent_signal_card_refresh_plan_matrix"' in diagnostic_cards
    assert '"agent_signal_card_refresh_status_matrix"' in diagnostic_cards
    assert '"agent_signal_refresh_routes"' in diagnostic_cards
    assert '"agent_top_card_slots"' in chat_client
    assert '"agent_signal_metadata_keys"' in chat_client
    assert '"gemma_signal_briefing_directives"' in chat_client
    assert '"agent_signal_briefing_matrix", "agent_signal_timeline", "agent_signal_refresh_routes",' in diagnostic_cards
    assert 'id = "signal_briefing"' in quick_actions
    assert 'id = "signal_card_deck"' in quick_actions
    assert 'id = "card_refresh_plan"' in quick_actions
    assert 'id = "card_refresh_status"' in quick_actions
    assert 'id = "signal_timeline"' in quick_actions
    assert 'action=agent_signal_briefing_report' in quick_actions
    assert 'action=agent_signal_card_deck_report' in quick_actions
    assert 'action=agent_signal_card_refresh_plan_report' in quick_actions
    assert 'action=agent_signal_card_refresh_status_report' in quick_actions
    assert 'action=agent_signal_timeline_report' in quick_actions


def test_android_diagnostics_exposes_signal_evidence_bundle_for_gemma_visible_current_context():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"agent_signal_evidence_report"' in diagnostics_bridge
    assert '"signal_evidence_bundle"' in diagnostics_bridge
    assert 'agentSignalEvidenceReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSignalEvidenceReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalEvidenceRows(' in diagnostics_bridge
    assert 'agentSignalEvidenceRouteRows()' in diagnostics_bridge
    assert 'signalEvidenceSourceActions()' in diagnostics_bridge
    assert 'signalEvidenceGraphTypes()' in diagnostics_bridge
    assert '"signal_evidence_matrix"' in diagnostics_bridge
    assert '"signal_evidence_routes"' in diagnostics_bridge
    assert '"signal_evidence_graph_types"' in diagnostics_bridge
    assert 'Signal Evidence Bundle' in diagnostics_bridge
    assert 'Wi-Fi AP and channel evidence' in diagnostics_bridge
    assert 'Bluetooth proximity evidence' in diagnostics_bridge
    assert 'Motion and sensor evidence' in diagnostics_bridge
    assert 'AM/FM and RF boundary evidence' in diagnostics_bridge
    assert 'Local inference readiness evidence' in diagnostics_bridge
    assert 'Accelerator delegate preflight evidence' in diagnostics_bridge
    assert 'Permission and refresh evidence' in diagnostics_bridge
    assert 'local_inference_compatibility_report' in diagnostics_bridge
    assert 'accelerator_preflight_evidence_summary' in diagnostics_bridge
    assert 'accelerator_preflight_report' in diagnostics_bridge
    assert 'accelerator_preflight_matrix' in diagnostics_bridge
    assert 'agent_signal_evidence_report' in chat_client
    assert 'signal_evidence_bundle' in chat_client
    assert 'extractImplicitSignalEvidenceArguments(userText)' in chat_client
    assert 'extractImplicitAndroidDiagnosticsArguments(userText)' in chat_client
    assert 'fun extractImplicitAndroidDiagnosticsArguments(userText: String): JSONObject?' in chat_client
    assert 'wifiDiagnosticArguments("wifi_channel_rating", userText)' in chat_client
    assert 'wifiDiagnosticArguments("wifi_export", userText)' in chat_client
    assert 'wifiDiagnosticArguments("wifi_ap_details", userText)' in chat_client
    assert 'bluetoothDiagnosticArguments("bluetooth_scan", userText)' in chat_client
    assert '"motion_sensor_history"' in chat_client
    assert '"sensor_types" to "accelerometer,gyroscope,linear_acceleration,rotation_vector"' in chat_client
    assert 'diagnosticArguments("radio_signal_graph")' in chat_client
    assert 'diagnosticArguments("soc_compatibility_report")' in chat_client
    assert 'what Hermes/Gemma can see from nearby signals' in chat_client
    assert 'first call android_device_diagnostics_tool action=agent_signal_evidence_report' in chat_client
    assert '"signal_evidence_matrix"' in chat_client
    assert '"signal_evidence_routes"' in chat_client
    assert '"signal_evidence_graph_types"' in chat_client
    assert '"signal_evidence_matrix", "signal_evidence_routes"' in diagnostic_cards
    assert 'id = "signal_evidence"' in quick_actions
    assert 'action=agent_signal_evidence_report' in quick_actions


def test_android_diagnostics_exposes_signal_replay_export_bundle_for_portable_context():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"agent_signal_replay_export_report"' in diagnostics_bridge
    assert '"signal_replay_export"' in diagnostics_bridge
    assert '"signal_evidence_export"' in diagnostics_bridge
    assert 'agentSignalReplayExportReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSignalReplayExportReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalReplayExportManifestRows(' in diagnostics_bridge
    assert 'agentSignalReplayFrameRows(' in diagnostics_bridge
    assert 'agentSignalReplayMetadataKeyRows()' in diagnostics_bridge
    assert 'agentSignalReplayExportSourceActions()' in diagnostics_bridge
    assert 'signalReplayExportGraphTypes()' in diagnostics_bridge
    assert 'agentSignalReplayExportBundleJson(' in diagnostics_bridge
    assert '"agent_signal_replay_export_manifest"' in diagnostics_bridge
    assert '"agent_signal_replay_frame_index"' in diagnostics_bridge
    assert '"agent_signal_replay_metadata_keys"' in diagnostics_bridge
    assert '"agent_signal_replay_export_bundle"' in diagnostics_bridge
    assert '"gemma_signal_replay_export_directives"' in diagnostics_bridge
    assert 'Signal Replay Export' in diagnostics_bridge
    assert 'Replay Frame Index' in diagnostics_bridge
    assert 'Replay Metadata Keys' in diagnostics_bridge
    assert 'agent_signal_replay_export_report' in chat_client
    assert 'signal replay/export bundles' in chat_client
    assert 'signal replay export' in chat_client
    assert 'signal_evidence_export' in chat_client
    assert 'diagnosticArguments("agent_signal_replay_export_report")' in chat_client
    assert '"agent_signal_replay_export_manifest"' in chat_client
    assert '"agent_signal_replay_frame_index"' in chat_client
    assert '"agent_signal_replay_metadata_keys"' in chat_client
    assert '"agent_signal_replay_export_bundle"' in chat_client
    assert '"gemma_signal_replay_export_directives"' in chat_client
    assert '"bundle_kind"' in chat_client
    assert '"schema_version"' in chat_client
    assert '"export_status"' in chat_client
    assert '"replay_frame"' in chat_client
    assert '"frame_key"' in chat_client
    assert '"claim_scope"' in chat_client
    assert '"proof_status"' in chat_client
    assert '"agent_signal_replay_export_manifest"' in diagnostic_cards
    assert '"agent_signal_replay_frame_index"' in diagnostic_cards
    assert '"agent_signal_replay_metadata_keys"' in diagnostic_cards
    assert 'id = "signal_replay_export"' in quick_actions
    assert 'label = "Replay Export"' in quick_actions
    assert 'action=agent_signal_replay_export_report' in quick_actions


def test_android_diagnostics_exposes_signal_replay_freshness_audit_for_staleness_safe_exports():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"agent_signal_replay_freshness_audit_report"' in diagnostics_bridge
    assert '"signal_replay_freshness"' in diagnostics_bridge
    assert 'agentSignalReplayFreshnessAuditReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSignalReplayFreshnessAuditReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalReplayFreshnessRows(' in diagnostics_bridge
    assert 'agentSignalReplayRefreshRouteRows(' in diagnostics_bridge
    assert 'agentSignalReplayStalenessSummaryRows(' in diagnostics_bridge
    assert 'agentSignalReplayFreshnessSourceActions()' in diagnostics_bridge
    assert 'agentSignalReplayFreshnessGraphTypes()' in diagnostics_bridge
    assert '"agent_signal_replay_freshness_matrix"' in diagnostics_bridge
    assert '"agent_signal_replay_refresh_routes"' in diagnostics_bridge
    assert '"agent_signal_replay_staleness_summary"' in diagnostics_bridge
    assert '"gemma_signal_replay_freshness_directives"' in diagnostics_bridge
    assert 'Replay Freshness Audit' in diagnostics_bridge
    assert 'Replay Refresh Routes' in diagnostics_bridge
    assert 'Replay Staleness Summary' in diagnostics_bridge
    assert 'agent_signal_replay_freshness_audit_report' in chat_client
    assert 'replay freshness/staleness audits' in chat_client
    assert 'freshness_status' in chat_client
    assert 'staleness_risk' in chat_client
    assert 'diagnosticArguments("agent_signal_replay_freshness_audit_report")' in chat_client
    assert '"agent_signal_replay_freshness_matrix"' in chat_client
    assert '"agent_signal_replay_refresh_routes"' in chat_client
    assert '"agent_signal_replay_staleness_summary"' in chat_client
    assert '"gemma_signal_replay_freshness_directives"' in chat_client
    assert '"freshness_status"' in chat_client
    assert '"staleness_risk"' in chat_client
    assert '"route_type"' in chat_client
    assert '"agent_signal_replay_freshness_matrix"' in diagnostic_cards
    assert '"agent_signal_replay_refresh_routes"' in diagnostic_cards
    assert '"agent_signal_replay_staleness_summary"' in diagnostic_cards
    assert 'id = "signal_replay_freshness"' in quick_actions
    assert 'label = "Replay Freshness"' in quick_actions
    assert 'action=agent_signal_replay_freshness_audit_report' in quick_actions


def test_android_diagnostics_exposes_signal_workflow_handoff_for_gemma_next_actions():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"agent_signal_workflow_handoff_report"' in diagnostics_bridge
    assert '"signal_workflow_handoff_report"' in diagnostics_bridge
    assert '"agent_next_signal_action_report"' in diagnostics_bridge
    assert 'agentSignalWorkflowHandoffReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSignalWorkflowHandoffReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalWorkflowHandoffRows(' in diagnostics_bridge
    assert 'agentSignalNextActionRouteRows()' in diagnostics_bridge
    assert 'agentSignalWorkflowHandoffSourceActions()' in diagnostics_bridge
    assert '"agent_signal_workflow_handoff_matrix"' in diagnostics_bridge
    assert '"agent_signal_next_action_routes"' in diagnostics_bridge
    assert '"gemma_signal_workflow_handoff_directives"' in diagnostics_bridge
    assert 'Signal Workflow Handoff' in diagnostics_bridge
    assert 'Next Signal Actions' in diagnostics_bridge
    assert 'Open Wi-Fi Analyzer graph' in diagnostics_bridge
    assert 'Open Bluetooth details and RSSI trends' in diagnostics_bridge
    assert 'Open motion sensor quality before pose' in diagnostics_bridge
    assert 'Open radio receiver advisor' in diagnostics_bridge
    assert 'Open non-Adreno backend advisor' in diagnostics_bridge
    assert 'Open non-Adreno backend launch advisor' in diagnostics_bridge
    assert 'Open Kai/MCP registry' in diagnostics_bridge
    assert 'bridge_required' in diagnostics_bridge
    assert 'physical_device_validation_required' in diagnostics_bridge
    assert 'passive_workflow_handoff' in diagnostics_bridge
    assert 'source_report_permissions_and_hardware_boundaries' in diagnostics_bridge
    assert 'agent_signal_workflow_handoff_report' in chat_client
    assert 'signal workflow handoff and next-action reports' in chat_client
    assert 'diagnosticArguments("agent_signal_workflow_handoff_report")' in chat_client
    assert '"agent_signal_workflow_handoff_matrix"' in chat_client
    assert '"agent_signal_next_action_routes"' in chat_client
    assert '"gemma_signal_workflow_handoff_directives"' in chat_client
    assert '"agent_signal_workflow_handoff_count"' in chat_client
    assert '"agent_signal_next_action_route_count"' in chat_client
    assert '"agent_signal_workflow_handoff_matrix"' in diagnostic_cards
    assert '"agent_signal_next_action_routes"' in diagnostic_cards
    assert 'id = "workflow_handoff"' in quick_actions
    assert 'label = "Workflow Handoff"' in quick_actions
    assert 'action=agent_signal_workflow_handoff_report' in quick_actions


def test_android_diagnostics_exposes_signal_permission_runbook_for_active_refresh_gates():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"agent_signal_permission_runbook_report"' in diagnostics_bridge
    assert '"signal_permission_runbook_report"' in diagnostics_bridge
    assert '"signal_refresh_runbook_report"' in diagnostics_bridge
    assert '"agent_signal_refresh_runbook_report"' in diagnostics_bridge
    assert '"active_signal_refresh_runbook"' in diagnostics_bridge
    assert '"signal_active_refresh_routes"' in diagnostics_bridge
    assert 'agentSignalPermissionRunbookReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSignalPermissionRunbookReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalPermissionRunbookRows(' in diagnostics_bridge
    assert 'agentSignalActiveRefreshRouteRows()' in diagnostics_bridge
    assert 'agentSignalPermissionRunbookSourceActions()' in diagnostics_bridge
    assert '"agent_signal_permission_runbook_matrix"' in diagnostics_bridge
    assert '"agent_signal_active_refresh_routes"' in diagnostics_bridge
    assert '"gemma_signal_permission_runbook_directives"' in diagnostics_bridge
    assert 'Prepare active Wi-Fi scan' in diagnostics_bridge
    assert 'Prepare active Bluetooth scan' in diagnostics_bridge
    assert 'Prepare motion sensor sample' in diagnostics_bridge
    assert 'Prepare AM/FM or SDR bridge samples' in diagnostics_bridge
    assert 'Prepare accelerator proof refresh' in diagnostics_bridge
    assert 'active_refresh_arguments' in diagnostics_bridge
    assert 'passive_fallback_action' in diagnostics_bridge
    assert 'user_consent_required' in diagnostics_bridge
    assert 'settings_actions' in diagnostics_bridge
    assert 'open_app_settings' in diagnostics_bridge
    assert 'open_location_settings' in diagnostics_bridge
    assert 'open_wifi_settings' in diagnostics_bridge
    assert 'open_bluetooth_settings' in diagnostics_bridge
    assert '"agent_signal_permission_runbook_report"' in chat_client
    assert 'signal permission and active-refresh runbooks' in chat_client
    assert 'diagnosticArguments("agent_signal_permission_runbook_report")' in chat_client
    assert '"agent_signal_permission_runbook_matrix"' in chat_client
    assert '"agent_signal_active_refresh_routes"' in chat_client
    assert '"gemma_signal_permission_runbook_directives"' in chat_client
    assert '"agent_signal_permission_runbook_count"' in chat_client
    assert '"agent_signal_active_refresh_route_count"' in chat_client
    assert '"settings_actions"' in chat_client
    assert '"active_refresh_arguments"' in chat_client
    assert '"agent_signal_permission_runbook_matrix"' in diagnostic_cards
    assert '"agent_signal_active_refresh_routes"' in diagnostic_cards
    assert 'id = "permission_runbook"' in quick_actions
    assert 'label = "Permission Runbook"' in quick_actions
    assert 'action=agent_signal_permission_runbook_report' in quick_actions


def test_android_diagnostics_exposes_signal_awareness_report_for_cross_signal_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"signal_awareness_report"' in diagnostics_bridge
    assert 'signalAwarenessReportJson(appContext)' in diagnostics_bridge
    assert 'signalAwarenessRows(' in diagnostics_bridge
    assert 'signalWorkflowRouteRows(' in diagnostics_bridge
    assert 'signalConstraintRows(' in diagnostics_bridge
    assert '"signal_awareness_matrix"' in diagnostics_bridge
    assert '"signal_workflow_routes"' in diagnostics_bridge
    assert '"signal_constraint_matrix"' in diagnostics_bridge
    assert '"radio_signal_feature_matrix"' in diagnostics_bridge
    assert '"radio_signal_workflow_routes"' in diagnostics_bridge
    assert '"radio_signal_constraint_matrix"' in diagnostics_bridge
    assert '"radio_signal_graph"' in diagnostics_bridge
    assert 'radioSignalGraphJson(appContext' in diagnostics_bridge
    assert 'radioSignalGraphRows(' in diagnostics_bridge
    assert '"radio_signal_graph_rows"' in diagnostics_bridge
    assert '"radio_signal_graph_sample_count"' in diagnostics_bridge
    assert '"radio_signal_graph_sample_summary"' in diagnostics_bridge
    assert '"radio_receiver_bridge_schema"' in diagnostics_bridge
    assert '"radio_samples_json"' in diagnostics_bridge
    assert '"radio_bridge_samples"' in diagnostics_bridge
    assert '"sdr_samples_json"' in diagnostics_bridge
    assert '"radio_signal_advisor_report"' in diagnostics_bridge
    assert 'radioSignalAdvisorReportJson(appContext' in diagnostics_bridge
    assert '"radio_signal_decision_packet_report"' in diagnostics_bridge
    assert 'radioSignalDecisionPacketReportJson(appContext' in diagnostics_bridge
    assert 'fun radioSignalDecisionPacketReportJson(context: Context' in diagnostics_bridge
    assert 'radioSignalDecisionPacketRows(' in diagnostics_bridge
    assert 'radioSignalDecisionRouteRows(' in diagnostics_bridge
    assert 'radioSignalDecisionClaimBoundaryRows(' in diagnostics_bridge
    assert 'radioSignalAdvisorRows(' in diagnostics_bridge
    assert 'radioReceiverCandidateRows(' in diagnostics_bridge
    assert '"radio_signal_advisor_matrix"' in diagnostics_bridge
    assert '"radio_receiver_candidates"' in diagnostics_bridge
    assert '"gemma_radio_advisor_directives"' in diagnostics_bridge
    assert '"radio_signal_decision_packet"' in diagnostics_bridge
    assert '"radio_signal_decision_routes"' in diagnostics_bridge
    assert '"radio_signal_claim_boundaries"' in diagnostics_bridge
    assert '"gemma_radio_signal_decision_directives"' in diagnostics_bridge
    assert '"radio_bridge_sample_metadata"' in diagnostics_bridge
    assert 'radioBridgeSampleMetadataRows(' in diagnostics_bridge
    assert 'accepted_radio_bridge_sample_array_keys' in diagnostics_bridge
    assert 'metadata_completeness_score' in diagnostics_bridge
    assert 'radioReceiverBridgeSchemaRows(' in diagnostics_bridge
    assert 'appendRadioSampleRowsFromString(' in diagnostics_bridge
    assert '"AM/FM Signal Graph"' in diagnostics_bridge
    assert '"Radio Bridge Sample Metadata"' in diagnostics_bridge
    assert 'radioBandPlanRows(' in diagnostics_bridge
    assert 'cached_wifi_signal_history' in diagnostics_bridge
    assert 'Route broad RF explanation' in diagnostics_bridge
    assert 'signal_awareness_report' in chat_client
    assert 'radio_analyzer_report' in chat_client
    assert 'radio_signal_graph' in chat_client
    assert 'radio_signal_advisor_report' in chat_client
    assert 'radio_signal_decision_packet_report' in chat_client
    assert 'diagnosticArguments("radio_signal_advisor_report")' in chat_client
    assert 'diagnosticArguments("radio_signal_decision_packet_report")' in chat_client
    assert '"signal_awareness_matrix"' in chat_client
    assert '"signal_workflow_routes"' in chat_client
    assert '"signal_constraint_matrix"' in chat_client
    assert '"radio_signal_feature_matrix"' in chat_client
    assert '"radio_signal_workflow_routes"' in chat_client
    assert '"radio_signal_constraint_matrix"' in chat_client
    assert '"radio_signal_graph_rows"' in chat_client
    assert '"radio_signal_graph_sample_rows"' in chat_client
    assert '"radio_signal_advisor_matrix"' in chat_client
    assert '"radio_receiver_candidates"' in chat_client
    assert '"gemma_radio_advisor_directives"' in chat_client
    assert '"radio_signal_decision_packet"' in chat_client
    assert '"radio_signal_decision_routes"' in chat_client
    assert '"radio_signal_claim_boundaries"' in chat_client
    assert '"gemma_radio_signal_decision_directives"' in chat_client
    assert '"radio_bridge_sample_metadata"' in chat_client
    assert '"radio_receiver_bridge_schema"' in chat_client
    assert '"radio_samples_json"' in chat_client
    assert '"radio_bridge_samples_json"' in chat_client
    assert '"sdr_samples_json"' in chat_client
    assert '"span_hz"' in chat_client
    assert '"sample_rate_hz"' in chat_client
    assert '"frequency_mhz"' in chat_client
    assert '"rds_radio_text"' in chat_client
    assert '"signal_awareness_matrix", "signal_workflow_routes", "signal_constraint_matrix",' in diagnostic_cards
    assert '"radio_signal_feature_matrix", "radio_signal_workflow_routes", "radio_signal_constraint_matrix",' in diagnostic_cards
    assert '"radio_signal_decision_packet", "radio_signal_decision_routes", "radio_signal_claim_boundaries"' in diagnostic_cards
    assert '"radio_signal_graph" -> radioSignalGraphRow(row)' in diagnostic_cards
    assert '"radio_signal_advisor_matrix", "radio_receiver_candidates"' in diagnostic_cards
    assert '"radio_bridge_sample_metadata" -> capabilityMatrixRow(row)' in diagnostic_cards
    assert '"radio_receiver_bridge_schema" -> radioReceiverProfileRow(row)' in diagnostic_cards
    assert 'radioSignalGraphRow(' in diagnostic_cards
    assert 'diagnosticAction = "radio_signal_graph"' in quick_actions
    assert 'diagnosticAction = "radio_signal_advisor_report"' in quick_actions
    assert 'diagnosticAction = "radio_signal_decision_packet_report"' in quick_actions
    assert 'id = "radio_advisor"' in quick_actions
    assert 'id = "radio_decision"' in quick_actions


def test_android_diagnostics_exposes_rf_coexistence_report_for_wifi_bluetooth_radio_context():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"rf_coexistence_report"' in diagnostics_bridge
    assert 'rfCoexistenceReportJson(appContext)' in diagnostics_bridge
    assert 'rfCoexistenceMatrixRows(' in diagnostics_bridge
    assert 'rfCoexistenceRouteRows(' in diagnostics_bridge
    assert '"rf_coexistence_matrix"' in diagnostics_bridge
    assert '"rf_coexistence_routes"' in diagnostics_bridge
    assert '"rf_coexistence_risk_score"' in diagnostics_bridge
    assert '"wifi_channel_utilization"' in diagnostics_bridge
    assert '"bluetooth_signal_history"' in diagnostics_bridge
    assert '"radio_signal_graph_rows"' in diagnostics_bridge
    assert '"mediatek_readiness_matrix"' in diagnostics_bridge
    assert 'rf_coexistence_report' in chat_client
    assert '"rf_coexistence_matrix"' in chat_client
    assert '"rf_coexistence_routes"' in chat_client
    assert '"rf_coexistence_matrix", "rf_coexistence_routes"' in diagnostic_cards
    assert 'id = "rf_coexistence"' in quick_actions
    assert 'diagnosticAction = "rf_coexistence_report"' in quick_actions


def test_android_diagnostics_exposes_soc_compatibility_report_for_backend_policy_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"soc_compatibility_report"' in diagnostics_bridge
    assert 'socCompatibilityReportJson(appContext)' in diagnostics_bridge
    assert '"device_performance_report"' in diagnostics_bridge
    assert 'devicePerformanceReportJson(appContext)' in diagnostics_bridge
    assert 'devicePerformanceProfileJson(appContext)' in diagnostics_bridge
    assert 'devicePerformanceMatrixRows(performanceProfile, socProfile)' in diagnostics_bridge
    assert '"gpu_backend_risk_report"' in diagnostics_bridge
    assert 'gpuBackendRiskReportJson(appContext)' in diagnostics_bridge
    assert '"local_inference_compatibility_report"' in diagnostics_bridge
    assert 'localInferenceCompatibilityReportJson(appContext)' in diagnostics_bridge
    assert 'localInferenceCompatibilityRows(' in diagnostics_bridge
    assert '"local_inference_compatibility_matrix"' in diagnostics_bridge
    assert '"local_inference_compatibility_score"' in diagnostics_bridge
    assert 'MediaTek and non-Adreno fallback policy' in diagnostics_bridge
    assert '"mediatek_readiness_report"' in diagnostics_bridge
    assert 'mediatekReadinessReportJson(appContext)' in diagnostics_bridge
    assert 'fun mediatekReadinessReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'mediatekReadinessRows(' in diagnostics_bridge
    assert '"mediatek_readiness_matrix"' in diagnostics_bridge
    assert '"mediatek_readiness_score"' in diagnostics_bridge
    assert 'MediaTek family detection' in diagnostics_bridge
    assert 'Mali and Immortalis GPU path' in diagnostics_bridge
    assert 'PowerVR/IMG fallback path' in diagnostics_bridge
    assert 'gpuBackendRiskMatrixRows(' in diagnostics_bridge
    assert 'gpuBackendRiskRouteRows(' in diagnostics_bridge
    assert '"gpu_backend_risk_matrix"' in diagnostics_bridge
    assert '"gpu_backend_risk_routes"' in diagnostics_bridge
    assert 'Live accelerator acceptance' in diagnostics_bridge
    assert 'Phone validation scope' in diagnostics_bridge
    assert '"local_backend_runtime_report"' in diagnostics_bridge
    assert 'localBackendRuntimeReportJson(appContext)' in diagnostics_bridge
    assert '"accelerator_preflight_report"' in diagnostics_bridge
    assert 'acceleratorPreflightReportJson(appContext)' in diagnostics_bridge
    assert 'fun acceleratorPreflightReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'acceleratorPreflightRows(' in diagnostics_bridge
    assert '"accelerator_preflight_matrix"' in diagnostics_bridge
    assert '"accelerator_preflight_count"' in diagnostics_bridge
    assert '"non_adreno_backend_advisor_report"' in diagnostics_bridge
    assert 'nonAdrenoBackendAdvisorReportJson(appContext)' in diagnostics_bridge
    assert 'fun nonAdrenoBackendAdvisorReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'nonAdrenoBackendAdvisorRows(' in diagnostics_bridge
    assert '"non_adreno_backend_advisor_matrix"' in diagnostics_bridge
    assert '"non_adreno_backend_launch_sequence"' in diagnostics_bridge
    assert '"gemma_non_adreno_backend_directives"' in diagnostics_bridge
    assert '"mediatek_backend_launch_checklist_report"' in diagnostics_bridge
    assert 'mediatekBackendLaunchChecklistReportJson(appContext)' in diagnostics_bridge
    assert 'fun mediatekBackendLaunchChecklistReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'mediatekBackendLaunchChecklistRows(' in diagnostics_bridge
    assert '"mediatek_backend_launch_checklist_matrix"' in diagnostics_bridge
    assert '"gemma_mediatek_launch_directives"' in diagnostics_bridge
    assert 'MediaTek Launch Checklist' in diagnostics_bridge
    assert 'Verify GPU proof or name CPU fallback' in diagnostics_bridge
    assert 'launch_gate_status' in diagnostics_bridge
    assert 'live_runtime_proof' in diagnostics_bridge
    assert 'cpu_fallback_explicit' in diagnostics_bridge
    assert 'Classify device family before launch' in diagnostics_bridge
    assert 'Choose artifact lane without Qualcomm bias' in diagnostics_bridge
    assert 'Prove live accelerator state' in diagnostics_bridge
    assert 'ABI and package lane' in diagnostics_bridge
    assert 'OpenCL library visibility' in diagnostics_bridge
    assert 'opencl_probe_loads_library' in diagnostics_bridge
    assert 'Non-Adreno GPU policy' in diagnostics_bridge
    assert 'runtimeBackendMatrixRows(' in diagnostics_bridge
    assert 'LiteRtLmOpenAiProxy.currentHealthJson()' in diagnostics_bridge
    assert 'socBackendMatrixRows(' in diagnostics_bridge
    assert 'socBackendRouteRows(' in diagnostics_bridge
    assert 'socBackendConstraintRows(' in diagnostics_bridge
    assert '"soc_backend_matrix"' in diagnostics_bridge
    assert '"soc_backend_policy_routes"' in diagnostics_bridge
    assert '"soc_backend_constraint_matrix"' in diagnostics_bridge
    assert '"runtime_backend_matrix"' in diagnostics_bridge
    assert '"runtime_stability_matrix"' in diagnostics_bridge
    assert 'Thermal & Memory Guardrails' in diagnostics_bridge
    assert 'MediaTek/non-Adreno stability guardrail' in diagnostics_bridge
    assert 'PowerManager.currentThermalStatus' in diagnostics_bridge
    assert 'Build.VERSION.MEDIA_PERFORMANCE_CLASS' in diagnostics_bridge
    assert 'LiteRT-LM /health accelerator' in diagnostics_bridge
    assert 'MediaTek/Mali/PowerVR coverage' in diagnostics_bridge
    assert 'Avoid Adreno-only assumptions' in diagnostics_bridge
    assert 'local_backend_runtime_report' in chat_client
    assert 'accelerator_preflight_report' in chat_client
    assert 'non_adreno_backend_advisor_report' in chat_client
    assert 'non_adreno_backend_advisor_matrix' in chat_client
    assert 'non_adreno_backend_launch_sequence' in chat_client
    assert 'mediatek_backend_launch_checklist_report' in chat_client
    assert '"mediatek_backend_launch_checklist_matrix"' in chat_client
    assert 'gemma_mediatek_launch_directives' in chat_client
    assert 'soc_compatibility_report' in chat_client
    assert 'gpu_backend_risk_report' in chat_client
    assert 'local_inference_compatibility_report' in chat_client
    assert 'mediatek_readiness_report' in chat_client
    assert '"mediatek_readiness_matrix"' in chat_client
    assert 'device_performance_report' in chat_client
    assert '"runtime_backend_matrix"' in chat_client
    assert '"runtime_stability_matrix"' in chat_client
    assert '"soc_backend_matrix"' in chat_client
    assert '"soc_backend_policy_routes"' in chat_client
    assert '"soc_backend_constraint_matrix"' in chat_client
    assert '"gpu_backend_risk_matrix"' in chat_client
    assert '"gpu_backend_risk_routes"' in chat_client
    assert '"local_inference_compatibility_matrix"' in chat_client
    assert '"accelerator_preflight_matrix"' in chat_client
    assert '"gpu_backend_risk_matrix", "gpu_backend_risk_routes",' in diagnostic_cards
    assert '"non_adreno_backend_advisor_matrix"' in diagnostic_cards
    assert '"mediatek_backend_launch_checklist_matrix"' in diagnostic_cards
    assert '"accelerator_preflight_matrix",' in diagnostic_cards
    assert '"mediatek_readiness_matrix",' in diagnostic_cards
    assert '"local_inference_compatibility_matrix",' in diagnostic_cards
    assert '"runtime_backend_matrix", "runtime_stability_matrix" -> capabilityMatrixRow(row)' in diagnostic_cards
    assert 'id = "runtime_backend"' in quick_actions
    assert 'id = "runtime_stability"' in quick_actions
    assert 'id = "soc_compatibility"' in quick_actions
    assert 'id = "mediatek_readiness"' in quick_actions
    assert 'id = "accelerator_preflight"' in quick_actions
    assert 'id = "non_adreno_backend_advisor"' in quick_actions
    assert 'id = "mediatek_launch_checklist"' in quick_actions
    assert 'id = "backend_risk"' in quick_actions
    assert 'id = "inference_compatibility"' in quick_actions
    assert 'action=local_inference_compatibility_report' in quick_actions


def test_litert_proxy_exposes_in_process_current_health_for_runtime_cards():
    litert_proxy = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/backend/LiteRtLmOpenAiProxy.kt").read_text(encoding="utf-8")

    assert 'internal fun currentHealthJson()' in litert_proxy
    assert 'fun healthJson(): JSONObject' in litert_proxy
    assert 'session.method == Method.GET && session.uri == "/health" -> jsonResponse(healthJson())' in litert_proxy
    assert 'put("accelerator", runtimeBackendLabel)' in litert_proxy
    assert 'put("gpu_policy", engineInitResult.gpuPolicy.description)' in litert_proxy
    assert 'put("gpu_fallback_to_cpu", engineInitResult.gpuPolicy.enabled && engineInitResult.backend != "gpu")' in litert_proxy


def test_android_diagnostics_exposes_wifi_analyzer_report_for_readiness_and_scan_policy_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github/workflows/android.yml").read_text(encoding="utf-8")

    assert '"wifi_analyzer_report"' in diagnostics_bridge
    assert '"wifi_signal_advisor_report"' in diagnostics_bridge
    assert '"wifi_channel_decision_packet_report"' in diagnostics_bridge
    assert '"wifi_connection_link"' in diagnostics_bridge
    assert 'wifiSignalAdvisorReportJson(appContext' in diagnostics_bridge
    assert 'fun wifiSignalAdvisorReportJson(context: Context' in diagnostics_bridge
    assert 'wifiChannelDecisionPacketReportJson(appContext' in diagnostics_bridge
    assert 'fun wifiChannelDecisionPacketReportJson(context: Context' in diagnostics_bridge
    assert 'wifiChannelDecisionPacketRows(' in diagnostics_bridge
    assert 'wifiChannelDecisionRouteRows(' in diagnostics_bridge
    assert 'wifiChannelDecisionClaimBoundaryRows(' in diagnostics_bridge
    assert 'wifiSignalAdvisorRows(' in diagnostics_bridge
    assert 'wifiRoamingCandidateRows(' in diagnostics_bridge
    assert 'wifiConnectionLinkReportJson(appContext)' in diagnostics_bridge
    assert 'fun wifiConnectionLinkReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert '"wifi_channel_graph"' in diagnostics_bridge
    assert '"wifi_channel_utilization"' in diagnostics_bridge
    assert '"wifi_signal_advisor_matrix"' in diagnostics_bridge
    assert '"wifi_roaming_candidates"' in diagnostics_bridge
    assert '"wifi_channel_decision_packet"' in diagnostics_bridge
    assert '"wifi_channel_decision_routes"' in diagnostics_bridge
    assert '"wifi_channel_decision_claim_boundaries"' in diagnostics_bridge
    assert '"gemma_wifi_advisor_directives"' in diagnostics_bridge
    assert '"gemma_wifi_channel_decision_directives"' in diagnostics_bridge
    assert 'wifiAnalyzerReportJson(appContext' in diagnostics_bridge
    assert 'wifiFilteredNetworkRows(' in diagnostics_bridge
    assert '"wifi_filtered_scan"' in diagnostics_bridge
    assert '"applied_wifi_filters"' in diagnostics_bridge
    assert '"wifi_filter_application"' in diagnostics_bridge
    assert 'wifiConnectionStatusJson(appContext, wifiManager, permissionStatus' in diagnostics_bridge
    assert 'wifiConnectionLinkRows(connectionStatus' in diagnostics_bridge
    assert 'Current connection link telemetry' in diagnostics_bridge
    assert 'Current Wi-Fi association' in diagnostics_bridge
    assert 'Link speed telemetry' in diagnostics_bridge
    assert '"filtered_wifi_analyzer_filters"' in diagnostics_bridge
    assert 'wifiAnalyzerFeatureRows(' in diagnostics_bridge
    assert 'wifiChannelGraphRows(' in diagnostics_bridge
    assert '"Wi-Fi Channel Graph"' in diagnostics_bridge
    assert 'wifiChannelUtilizationRowsForNetworks(' in diagnostics_bridge
    assert 'wifiAccessPointSemanticRows(' in diagnostics_bridge
    assert 'wifiBandCoverageRows(' in diagnostics_bridge
    assert 'wifiAnalyzerWorkflowRows(' in diagnostics_bridge
    assert 'wifiScanPolicyRows(' in diagnostics_bridge
    assert '"wifi_analyzer_feature_matrix"' in diagnostics_bridge
    assert '"wifi_analyzer_workflow_routes"' in diagnostics_bridge
    assert '"wifi_scan_policy_matrix"' in diagnostics_bridge
    assert '"wifi_scan_control"' in diagnostics_bridge
    assert '"wifi_channel_graph_count"' in diagnostics_bridge
    assert '"wifi_connection_link_count"' in diagnostics_bridge
    assert 'scan_mode=paused' in diagnostics_bridge
    assert 'scan_mode=resumed' in diagnostics_bridge
    assert 'Pause/resume scan control' in diagnostics_bridge
    assert 'Route pause or resume scan mode' in diagnostics_bridge
    assert '"wifi_channel_utilization"' in diagnostics_bridge
    assert '"wifi_access_point_semantics"' in diagnostics_bridge
    assert '"wifi_band_coverage"' in diagnostics_bridge
    assert 'WiFiAnalyzer-style readiness' in diagnostics_bridge
    assert 'WiFiAnalyzer-style decision support' in diagnostics_bridge
    assert 'Gemma-visible Wi-Fi channel decision packet' in diagnostics_bridge
    assert 'Current link decision' in diagnostics_bridge
    assert 'Roaming candidate decision' in diagnostics_bridge
    assert 'Permission and refresh decision' in diagnostics_bridge
    assert 'Channel signal graph' in diagnostics_bridge
    assert 'Channel utilization occupancy' in diagnostics_bridge
    assert 'Agent AP semantic and risk labels' in diagnostics_bridge
    assert 'Band coverage and 2.4/5/6GHz visibility' in diagnostics_bridge
    assert 'wifi_analyzer_report' in chat_client
    assert 'wifi_signal_advisor_report' in chat_client
    assert 'wifi_channel_decision_packet_report' in chat_client
    assert 'wifi_connection_link' in chat_client
    assert '"wifi_connection_link"' in chat_client
    assert '"wifi_signal_advisor_matrix"' in chat_client
    assert '"wifi_roaming_candidates"' in chat_client
    assert '"wifi_channel_decision_packet"' in chat_client
    assert '"wifi_channel_decision_routes"' in chat_client
    assert '"wifi_channel_decision_claim_boundaries"' in chat_client
    assert '"gemma_wifi_channel_decision_directives"' in chat_client
    assert 'wifi_filtered_scan' in chat_client
    assert 'wifi_channel_graph' in chat_client
    assert '"filter_band"' in chat_client
    assert '"filter_security"' in chat_client
    assert '"filter_signal"' in chat_client
    assert '"filter_ssid"' in chat_client
    assert '"filter_vendor"' in chat_client
    assert '"export_format"' in chat_client
    assert 'fun wifiDiagnosticArguments(action: String, userText: String): JSONObject' in chat_client
    assert 'wifiTextFilter(userText, "ssid")' in chat_client
    assert 'wifiTextFilter(userText, "vendor")' in chat_client
    assert 'filter_band" to bandFilters.joinToString(",")' in chat_client
    assert '"min_rssi_dbm"' in chat_client
    assert '"applied_wifi_filters"' in chat_client
    assert '"wifi_filter_application"' in chat_client
    assert '"wifi_connection_status"' in chat_client
    assert 'wifi_channel_utilization' in chat_client
    assert '"wifi_analyzer_feature_matrix"' in chat_client
    assert '"wifi_channel_utilization"' in chat_client
    assert '"wifi_channel_graph"' in chat_client
    assert '"wifi_access_point_semantics"' in chat_client
    assert '"wifi_band_coverage"' in chat_client
    assert '"wifi_analyzer_workflow_routes"' in chat_client
    assert '"wifi_scan_policy_matrix"' in chat_client
    assert '"wifi_signal_advisor_count"' in chat_client
    assert '"wifi_roaming_candidate_count"' in chat_client
    assert '"wifi_scan_control"' in chat_client
    assert '"scan_mode"' in chat_client
    assert 'Wi-Fi or Bluetooth scan mode for direct signal actions' in chat_client
    assert '"wifi_channel_graph" -> wifiChannelGraphRow(row)' in diagnostic_cards
    assert '"wifi_channel_utilization" -> wifiChannelUtilizationRow(row)' in diagnostic_cards
    assert '"wifi_channel_decision_packet", "wifi_channel_decision_routes", "wifi_channel_decision_claim_boundaries"' in diagnostic_cards
    assert '"wifi_access_point_semantics" -> wifiAccessPointSemanticRow(row)' in diagnostic_cards
    assert '"wifi_band_coverage" -> wifiBandCoverageRow(row)' in diagnostic_cards
    assert '"wifi_analyzer_feature_matrix", "wifi_analyzer_workflow_routes", "wifi_scan_policy_matrix"' in diagnostic_cards
    assert '"wifi_connection_link",' in diagnostic_cards
    assert '"wifi_signal_advisor_matrix", "wifi_roaming_candidates"' in diagnostic_cards
    assert '"wifi_filter_application"' in diagnostic_cards
    assert 'capabilityMatrixRow(row)' in diagnostic_cards
    assert 'id = "wifi_advisor"' in quick_actions
    assert 'id = "wifi_channel_decision"' in quick_actions
    assert 'diagnosticAction = "wifi_signal_advisor_report"' in quick_actions
    assert 'diagnosticAction = "wifi_channel_decision_packet_report"' in quick_actions
    assert 'diagnosticAction = "wifi_connection_link"' in quick_actions
    assert ':app:compileDebugAndroidTestKotlin' in workflow


def test_android_diagnostics_exposes_bluetooth_analyzer_report_for_readiness_and_scan_policy_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")

    assert '"bluetooth_analyzer_report"' in diagnostics_bridge
    assert '"bluetooth_signal_history"' in diagnostics_bridge
    assert '"bluetooth_device_details"' in diagnostics_bridge
    assert '"bluetooth_export"' in diagnostics_bridge
    assert '"bluetooth_signal_advisor_report"' in diagnostics_bridge
    assert '"bluetooth_nearby_decision_packet_report"' in diagnostics_bridge
    assert 'bluetoothSignalAdvisorReportJson(appContext' in diagnostics_bridge
    assert 'bluetoothNearbyDecisionPacketReportJson(appContext' in diagnostics_bridge
    assert 'fun bluetoothNearbyDecisionPacketReportJson(context: Context' in diagnostics_bridge
    assert 'bluetoothNearbyDecisionPacketRows(' in diagnostics_bridge
    assert 'bluetoothNearbyDecisionRouteRows(' in diagnostics_bridge
    assert 'bluetoothNearbyDecisionClaimBoundaryRows(' in diagnostics_bridge
    assert 'bluetoothSignalAdvisorRows(' in diagnostics_bridge
    assert 'bluetoothDeviceCandidateRows(' in diagnostics_bridge
    assert '"bluetooth_signal_advisor_matrix"' in diagnostics_bridge
    assert '"bluetooth_device_candidates"' in diagnostics_bridge
    assert '"gemma_bluetooth_advisor_directives"' in diagnostics_bridge
    assert '"bluetooth_nearby_decision_packet"' in diagnostics_bridge
    assert '"bluetooth_nearby_decision_routes"' in diagnostics_bridge
    assert '"bluetooth_nearby_claim_boundaries"' in diagnostics_bridge
    assert '"gemma_bluetooth_nearby_directives"' in diagnostics_bridge
    assert 'mergeBluetoothSignalHistory(' in diagnostics_bridge
    assert 'bluetoothSignalHistoryRowsFromStore(' in diagnostics_bridge
    assert 'bluetoothAnalyzerReportJson(appContext' in diagnostics_bridge
    assert 'bluetoothDeviceDetailsJson(appContext' in diagnostics_bridge
    assert 'bluetoothDeviceDetailRows(' in diagnostics_bridge
    assert 'bluetoothDeviceExportJson(' in diagnostics_bridge
    assert 'bluetoothAnalyzerFeatureRows(' in diagnostics_bridge
    assert 'bluetoothAnalyzerWorkflowRows(' in diagnostics_bridge
    assert 'bluetoothScanPolicyRows(' in diagnostics_bridge
    assert 'bluetoothScanFilterSpec(' in diagnostics_bridge
    assert 'bluetoothFilteredDeviceRowsForSpec(' in diagnostics_bridge
    assert '"applied_bluetooth_filters"' in diagnostics_bridge
    assert '"bluetooth_filter_application"' in diagnostics_bridge
    assert '"available_bluetooth_analyzer_filters"' in diagnostics_bridge
    assert 'bluetoothServiceUuidLabel(' in diagnostics_bridge
    assert 'bluetoothManufacturerIdLabel(' in diagnostics_bridge
    assert '"service_labels"' in diagnostics_bridge
    assert '"manufacturer_names"' in diagnostics_bridge
    assert 'Bluetooth SIG service labels' in diagnostics_bridge
    assert '"bluetooth_signal_history"' in diagnostics_bridge
    assert '"bluetooth_device_detail"' in diagnostics_bridge
    assert 'Open Bluetooth device detail evidence' in diagnostics_bridge
    assert 'Bluetooth device detail route' in diagnostics_bridge
    assert '"bluetooth_analyzer_feature_matrix"' in diagnostics_bridge
    assert '"bluetooth_analyzer_workflow_routes"' in diagnostics_bridge
    assert '"bluetooth_scan_policy_matrix"' in diagnostics_bridge
    assert '"bluetooth_scan_control"' in diagnostics_bridge
    assert 'Pause/resume BLE scan control' in diagnostics_bridge
    assert 'Route pause or resume BLE scan mode' in diagnostics_bridge
    assert 'Device detail and export rows' in diagnostics_bridge
    assert 'Route Bluetooth device details/export' in diagnostics_bridge
    assert 'scan_mode=paused' in diagnostics_bridge
    assert 'scan_mode=resumed' in diagnostics_bridge
    assert 'Bluetooth Analyzer readiness' in diagnostics_bridge
    assert 'Gemma-visible Bluetooth nearby decision packet' in diagnostics_bridge
    assert 'bluetooth_analyzer_report' in chat_client
    assert 'bluetooth_signal_advisor_report' in chat_client
    assert 'bluetooth_nearby_decision_packet_report' in chat_client
    assert 'bluetooth_signal_history' in chat_client
    assert 'bluetooth_device_details' in chat_client
    assert 'bluetooth_export' in chat_client
    assert '"bluetooth_device_detail_count"' in chat_client
    assert '"bluetooth_device_details"' in chat_client
    assert '"bluetooth_service_label_count"' in chat_client
    assert '"bluetooth_manufacturer_name_count"' in chat_client
    assert '"service_labels"' in chat_client
    assert '"manufacturer_names"' in chat_client
    assert '"bluetooth_signal_history"' in chat_client
    assert '"bluetooth_analyzer_feature_matrix"' in chat_client
    assert '"bluetooth_analyzer_workflow_routes"' in chat_client
    assert '"bluetooth_scan_policy_matrix"' in chat_client
    assert '"bluetooth_signal_advisor_matrix"' in chat_client
    assert '"bluetooth_device_candidates"' in chat_client
    assert '"bluetooth_nearby_decision_packet"' in chat_client
    assert '"bluetooth_nearby_decision_routes"' in chat_client
    assert '"bluetooth_nearby_claim_boundaries"' in chat_client
    assert '"gemma_bluetooth_nearby_directives"' in chat_client
    assert '"bluetooth_device_candidate_count"' in chat_client
    assert '"bluetooth_scan_control"' in chat_client
    assert 'bluetoothDiagnosticArguments("bluetooth_signal_advisor_report", userText)' in chat_client
    assert 'bluetoothDiagnosticArguments("bluetooth_nearby_decision_packet_report", userText)' in chat_client
    assert 'bluetoothDiagnosticArguments("bluetooth_scan", userText)' in chat_client
    assert 'bluetoothDiagnosticArguments("bluetooth_device_details", userText)' in chat_client
    assert 'bluetoothDiagnosticArguments("bluetooth_export", userText)' in chat_client
    assert '"filter_bluetooth_service"' in chat_client
    assert '"filter_bluetooth_manufacturer"' in chat_client
    assert '"filter_bluetooth_proximity"' in chat_client
    assert 'Wi-Fi or Bluetooth scan mode for direct signal actions' in chat_client
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")
    assert 'diagnosticAction = "bluetooth_signal_advisor_report"' in quick_actions
    assert 'diagnosticAction = "bluetooth_nearby_decision_packet_report"' in quick_actions
    assert 'diagnosticAction = "bluetooth_device_details"' in quick_actions
    assert '"bluetooth_signal_history" -> bluetoothSignalHistoryRow(row)' in diagnostic_cards
    assert '"bluetooth_device_detail" -> bluetoothRow(row)' in diagnostic_cards
    assert '"bluetooth_analyzer_feature_matrix", "bluetooth_analyzer_workflow_routes", "bluetooth_scan_policy_matrix"' in diagnostic_cards
    assert '"bluetooth_nearby_decision_packet", "bluetooth_nearby_decision_routes", "bluetooth_nearby_claim_boundaries"' in diagnostic_cards
    assert '"bluetooth_signal_advisor_matrix", "bluetooth_device_candidates"' in diagnostic_cards
    assert '"bluetooth_filter_application"' in diagnostic_cards
    assert 'capabilityMatrixRow(row)' in diagnostic_cards


def test_android_diagnostics_exposes_sensor_analyzer_report_for_motion_and_sampling_policy_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")

    assert '"sensor_analyzer_report"' in diagnostics_bridge
    assert '"sensor_workflow_advisor_report"' in diagnostics_bridge
    assert '"motion_sensor_decision_packet_report"' in diagnostics_bridge
    assert '"motion_sensor_history"' in diagnostics_bridge
    assert 'mergeMotionSensorHistory(' in diagnostics_bridge
    assert 'motionSensorHistoryRowsFromStore(' in diagnostics_bridge
    assert 'sensorAnalyzerReportJson(appContext' in diagnostics_bridge
    assert 'sensorWorkflowAdvisorReportJson(appContext' in diagnostics_bridge
    assert 'motionSensorDecisionPacketReportJson(appContext' in diagnostics_bridge
    assert 'fun motionSensorDecisionPacketReportJson(context: Context' in diagnostics_bridge
    assert 'motionSensorDecisionPacketRows(' in diagnostics_bridge
    assert 'motionSensorDecisionRouteRows(' in diagnostics_bridge
    assert 'motionSensorDecisionClaimBoundaryRows(' in diagnostics_bridge
    assert 'sensorWorkflowAdvisorRows(' in diagnostics_bridge
    assert 'sensorWorkflowCandidateRows(' in diagnostics_bridge
    assert 'sensorAnalyzerFeatureRows(' in diagnostics_bridge
    assert 'sensorAnalyzerWorkflowRows(' in diagnostics_bridge
    assert 'sensorSamplingPolicyRows(' in diagnostics_bridge
    assert 'motionPoseEstimateRows(' in diagnostics_bridge
    assert '"sensor_analyzer_feature_matrix"' in diagnostics_bridge
    assert '"sensor_analyzer_workflow_routes"' in diagnostics_bridge
    assert '"sensor_sampling_policy_matrix"' in diagnostics_bridge
    assert '"sensor_workflow_advisor_matrix"' in diagnostics_bridge
    assert '"sensor_workflow_candidates"' in diagnostics_bridge
    assert '"gemma_sensor_workflow_directives"' in diagnostics_bridge
    assert '"motion_sensor_decision_packet"' in diagnostics_bridge
    assert '"motion_sensor_decision_routes"' in diagnostics_bridge
    assert '"motion_sensor_claim_boundaries"' in diagnostics_bridge
    assert '"gemma_motion_sensor_decision_directives"' in diagnostics_bridge
    assert '"motion_pose_estimates"' in diagnostics_bridge
    assert '"motion_pose_estimate"' in diagnostics_bridge
    assert '"motion_sensor_quality"' in diagnostics_bridge
    assert 'motionSensorQualityJson(appContext' in diagnostics_bridge
    assert 'motionSensorQualityRows(' in diagnostics_bridge
    assert '"motion_pose"' in diagnostics_bridge
    assert 'Sensor Analyzer readiness' in diagnostics_bridge
    assert 'accelerometer' in diagnostics_bridge
    assert 'gyroscope' in diagnostics_bridge
    assert 'sensor_analyzer_report' in chat_client
    assert 'sensor_workflow_advisor_report' in chat_client
    assert 'motion_sensor_decision_packet_report' in chat_client
    assert 'motion_sensor_quality' in chat_client
    assert 'motion_sensor_history' in chat_client
    assert 'motion_pose' in chat_client
    assert '"motion_sensor_quality"' in chat_client
    assert '"motion_pose_estimates"' in chat_client
    assert '"sensor_workflow_advisor_matrix"' in chat_client
    assert '"sensor_workflow_candidates"' in chat_client
    assert '"sensor_workflow_candidate_count"' in chat_client
    assert '"motion_sensor_decision_packet"' in chat_client
    assert '"motion_sensor_decision_routes"' in chat_client
    assert '"motion_sensor_claim_boundaries"' in chat_client
    assert '"gemma_motion_sensor_decision_directives"' in chat_client
    assert 'diagnosticArguments("sensor_workflow_advisor_report", "include_snapshot" to false)' in chat_client
    assert 'diagnosticArguments("motion_sensor_decision_packet_report", "include_snapshot" to false)' in chat_client
    assert '"motion_sensor_decision_packet"' in diagnostic_cards
    assert '"motion_sensor_decision_routes"' in diagnostic_cards
    assert '"motion_sensor_claim_boundaries"' in diagnostic_cards
    assert '"motion_sensor_quality"' in diagnostic_cards
    assert '"sensor_workflow_advisor_matrix"' in diagnostic_cards
    assert '"sensor_workflow_candidates"' in diagnostic_cards
    assert '"sensor_analyzer_feature_matrix"' in chat_client
    assert '"sensor_analyzer_workflow_routes"' in chat_client
    assert '"sensor_sampling_policy_matrix"' in chat_client
    assert '"sensor_analyzer_feature_matrix", "sensor_analyzer_workflow_routes", "sensor_sampling_policy_matrix"' in diagnostic_cards
    assert '"sensor_workflow_advisor_matrix", "sensor_workflow_candidates"' in diagnostic_cards
    assert '"motion_sensor_decision_packet", "motion_sensor_decision_routes", "motion_sensor_claim_boundaries"' in diagnostic_cards
    assert '"motion_sensor_quality" -> capabilityMatrixRow(row)' in diagnostic_cards
    assert '"motion_sensor_history" -> motionSensorHistoryRow(row)' in diagnostic_cards
    assert '"motion_pose_estimate" -> motionPoseEstimateRow(row)' in diagnostic_cards
    assert 'capabilityMatrixRow(row)' in diagnostic_cards
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")
    assert 'diagnosticAction = "sensor_workflow_advisor_report"' in quick_actions
    assert 'diagnosticAction = "motion_sensor_decision_packet_report"' in quick_actions
    assert 'id = "motion_decision"' in quick_actions


def test_android_linux_subsystem_reapplies_executable_bits_before_reusing_cached_prefix():
    bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesLinuxSubsystemBridge.kt").read_text(encoding="utf-8")
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
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesIntentBridge.kt"
    ).read_text(encoding="utf-8")
    automation_test = (
        REPO_ROOT / "android/app/src/androidTest/java/com/mobilefork/hermesagent/HermesAutomationInstrumentedTest.kt"
    ).read_text(encoding="utf-8")

    assert 'androidx.core.content.FileProvider' in manifest
    assert 'android:authorities="${applicationId}.files"' in manifest
    assert 'android:resource="@xml/hermes_file_paths"' in manifest
    assert 'name="hermes_home"' in paths
    assert 'path="hermes-home/"' in paths
    assert 'FileProvider.getUriForFile(' in intent_bridge
    assert 'Intent.FLAG_GRANT_READ_URI_PERMISSION' in intent_bridge
    assert '"html", "htm" -> "text/html"' in intent_bridge
    assert 'shouldAddBrowsableCategory(resolvedOpenUri?.uri ?: intent.data)' in intent_bridge
    assert 'BROWSABLE_URI_SCHEMES = setOf("http", "https")' in intent_bridge
    assert 'Browser.EXTRA_APPLICATION_ID' in intent_bridge
    assert 'preferBrowserPackage' in intent_bridge
    assert 'selectPreferredBrowserPackage' in intent_bridge
    assert '<queries>' in manifest
    assert 'android:scheme="https"' in manifest
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
        / "android/app/src/androidTest/java/com/mobilefork/hermesagent/NativeAgentRuntimeSmokeTest.kt"
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
    downloads_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt").read_text(encoding="utf-8")
    download_manager = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/models/HermesModelDownloadManager.kt").read_text(encoding="utf-8")

    assert 'Dispatchers.IO' in downloads_view_model
    assert 'withContext(Dispatchers.IO)' in downloads_view_model
    assert 'selectRepoFileForDownload' in download_manager
    assert 'findCompatibleRepoFile' in download_manager
    assert 'findFallbackRepoFile' in download_manager
    assert 'api/models/' in download_manager
    assert 'Unable to infer a downloadable model artifact' in download_manager
    assert 'huggingface.co/' in download_manager


def test_chat_composer_matches_round_ui_spec():
    chat = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")

    assert 'RoundedCornerShape(28.dp)' in chat
    assert 'shape = RoundedCornerShape(28.dp)' in chat


def test_overlay_scene_uses_screen_aware_window_bounds():
    overlay = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesOverlaySceneBridge.kt").read_text(encoding="utf-8")
    automation_test = (
        REPO_ROOT / "android/app/src/test/java/com/mobilefork/hermesagent/device/HermesAutomationStoreTest.kt"
    ).read_text(encoding="utf-8")

    assert "val layoutMetrics = resolvedLayoutMetrics(appContext, payload)" in overlay
    assert "currentWindowMetrics" in overlay
    assert "windowInsets.getInsetsIgnoringVisibility" in overlay
    assert "safeInsetLeftPx" in overlay
    assert "usableWidthPx" in overlay
    assert "screen_aspect_ratio" in overlay
    assert "width_mode" in overlay
    assert "WIDTH_MODE_FRACTION" in overlay
    assert "availableWidthPx" in overlay
    assert "resolvedWidthPx" in overlay
    assert "OVERLAY_EDGE_MARGIN_DP" in overlay
    assert "FLAG_LAYOUT_NO_LIMITS" not in overlay
    assert "maxLines = layoutMetrics.textMaxLines" in overlay
    assert "maxHeight = layoutMetrics.textMaxHeightPx" in overlay
    assert "TextUtils.TruncateAt.END" in overlay
    assert "layoutMetrics.toJson()" in overlay
    assert "layout.resolvedWidthPx <= layout.availableWidthPx" in automation_test
    assert "overlaySceneLayoutHandlesPercentPixelAndNarrowScreens" in automation_test


def test_provider_setup_webview_errors_show_browser_copy_fallback():
    activity = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesProviderSetupWebActivity.kt"
    ).read_text(encoding="utf-8")
    activity_test = (
        REPO_ROOT / "android/app/src/androidTest/java/com/mobilefork/hermesagent/ProviderSetupWebActivityInstrumentedTest.kt"
    ).read_text(encoding="utf-8")

    assert "onReceivedError" in activity
    assert "onReceivedHttpError" in activity
    assert "request.isForMainFrame" in activity
    assert "showLoadFailureFallback" in activity
    assert "showFallback(setupPageTitle" in activity
    assert "Setup page failed to load; URL copied." in activity
    assert 'toolbarButton("Open in browser")' in activity
    assert 'toolbarButton("Copy URL")' in activity
    assert 'listOf("openrouter", "alibaba", "alibaba-coding-plan", "qwen-oauth", "zai", "zai-coding-plan")' in activity_test
    assert 'providerSetupOpenUsesExternalBrowserForQwenCloudWhenAvailable' in activity_test
    assert 'HermesExternalBrowserLauncher.createBrowserIntent' in activity_test
    assert 'Intents.init()' in activity_test
    assert 'provider setup chooser for ' in activity_test
    assert 'qwenDocsOpened.get()' in activity_test


def test_device_backend_exposes_deeper_radio_control_actions_and_status():
    bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesSystemControlBridge.kt").read_text(encoding="utf-8")
    device = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/device/DeviceScreen.kt").read_text(encoding="utf-8")
    state_writer = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/DeviceStateWriter.kt").read_text(encoding="utf-8")

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
    assert 'android_device_performance_profile' in state_writer


def test_android_automation_exposes_operator_standby_history_for_remote_dispatch():
    bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesAutomationBridge.kt").read_text(encoding="utf-8")
    store = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesAutomationStore.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/device/DeviceViewModel.kt").read_text(encoding="utf-8")
    device = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/device/DeviceScreen.kt").read_text(encoding="utf-8")

    assert 'HermesAutomationRunEvent' in store
    assert 'KEY_RUN_EVENTS' in store
    assert 'addRunEvent' in store
    assert 'listRunEvents' in store
    assert 'operator_standby_status' in bridge
    assert 'operator_devices' in bridge
    assert 'OpenGUI devices' in bridge
    assert 'compatible_device_queries' in bridge
    assert 'operator_execution_status' in bridge
    assert 'role_routing_supported' in bridge
    assert 'standby_dispatch_supported' in bridge
    assert 'long_running_task_state_supported' in bridge
    assert 'execution_review_supported' in bridge
    assert 'structured_results_supported' in bridge
    assert 'structured_result_schema' in bridge
    assert '"planner"' in bridge
    assert '"executor_vlm"' in bridge
    assert '"summarizer"' in bridge
    assert 'execution_state_strategy' in bridge
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


def test_android_ui_tool_has_opengui_style_coordinate_gesture_parity():
    controller = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesAccessibilityController.kt"
    ).read_text(encoding="utf-8")
    ui_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesAccessibilityUiBridge.kt"
    ).read_text(encoding="utf-8")
    chat_client = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt"
    ).read_text(encoding="utf-8")
    opengui_parser = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/OpenGuiActionCompat.kt"
    ).read_text(encoding="utf-8")
    app_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesAppControlBridge.kt"
    ).read_text(encoding="utf-8")
    accessibility_config = (
        REPO_ROOT / "android/app/src/main/res/xml/hermes_accessibility_service.xml"
    ).read_text(encoding="utf-8")

    assert 'GestureDescription' in controller
    assert 'dispatchGesture' in controller
    assert 'fun performTap(' in controller
    assert 'fun performSwipe(' in controller
    assert 'HermesScreenMetrics' in controller

    assert 'fun performCoordinateGestureJson(' in ui_bridge
    assert 'fun captureScreenshotJson(' in ui_bridge
    assert 'takeScreenshot(' in ui_bridge
    assert 'Bitmap.wrapHardwareBuffer' in ui_bridge
    assert 'hermes-screenshots' in ui_bridge
    assert 'image_sha256' in ui_bridge
    assert 'screenshot_hash_kind' in ui_bridge
    assert 'fun performScrollGestureJson(' in ui_bridge
    assert 'fun performTextInputJson(' in ui_bridge
    assert 'NORMALIZED_COORDINATE_SPACES' in ui_bridge
    assert 'PERCENT_COORDINATE_SPACES' in ui_bridge
    assert 'defaultScrollStartPoint' in ui_bridge
    assert 'resolvedScrollDistance' in ui_bridge
    assert 'resolved_coordinates' in ui_bridge
    assert 'screen_width' in ui_bridge
    assert 'screen_height' in ui_bridge
    assert 'current_app_name' in ui_bridge
    assert 'scale_factor' in ui_bridge
    assert 'normalized_coordinate_support' in ui_bridge
    assert 'className: String' in ui_bridge
    assert 'node.className?.toString().orEmpty().contains(className' in ui_bridge

    for action in [
        '"sense"',
        '"opengui_sense"',
        '"perception_status"',
        '"a11y_tree"',
        '"tap"',
        '"long_press"',
        '"swipe"',
        '"drag"',
        '"screenshot"',
        '"visual_snapshot"',
        '"capture_screenshot"',
        '"opengui_history"',
        '"clear_opengui_history"',
        '"open_app"',
        '"launch_app"',
        '"coordinate_tap"',
        '"coordinate_click"',
        '"coordinate_swipe"',
        '"scroll"',
        '"scroll_up"',
        '"scroll_down"',
        '"scroll_left"',
        '"scroll_right"',
        '"type"',
        '"type_text"',
        '"parse_opengui_action"',
        '"opengui_action"',
        '"press_home"',
        '"press_back"',
    ]:
        assert action in chat_client

    for argument in [
        '"raw_action"',
        '"x"',
        '"y"',
        '"x1"',
        '"y1"',
        '"x2"',
        '"y2"',
        '"coordinate_space"',
        '"duration_ms"',
        '"direction"',
        '"distance_px"',
        '"save_file"',
        '"include_base64"',
        '"max_image_edge_px"',
        '"include_snapshot"',
        '"include_screenshot"',
        '"class_name"',
        '"app_name"',
    ]:
        assert argument in chat_client

    assert 'executeAndroidCoordinateGesture' in chat_client
    assert 'executeAndroidScrollGesture' in chat_client
    assert 'hasCoordinateGestureArguments' in chat_client
    assert 'executeAndroidSelectorAction' in chat_client
    assert 'coordinate_arguments' in chat_client
    assert 'opengui_action_arguments' in chat_client
    assert 'screenshot_capture_supported' in chat_client
    assert 'OpenGuiSenseStatus.build' in chat_client
    assert 'opengui_sense_supported' in chat_client
    assert 'opengui_sense_actions' in chat_client
    assert 'executeAndroidScreenshotTool' in chat_client
    assert 'optionalBooleanArgument' in chat_client
    assert 'normalized_coordinate_support' in chat_client
    assert 'screen_width' in chat_client
    assert 'HermesAppControlBridge.launchApp' in chat_client
    assert 'fun launchApp(context: Context, packageName: String, appName: String)' in app_bridge
    assert 'queryIntentActivities(launcherIntent, 0)' in app_bridge
    assert 'launch_app app_name matched multiple launcher apps; pass package_name' in app_bridge
    assert 'object OpenGuiActionCompat' in opengui_parser
    assert 'click(start_box=' in chat_client
    assert 'need_login' in opengui_parser
    assert 'asset_risk' in opengui_parser
    assert 'delete_confirm' in opengui_parser
    assert 'downgrade_to_a11y' in opengui_parser
    assert '"start_coords"' in opengui_parser
    assert '"end_coords"' in opengui_parser
    assert '<bbox>' in opengui_parser
    assert '<point>' in opengui_parser
    assert 'extractPredictionMetadata' in opengui_parser
    assert '"action_summary"' in opengui_parser
    assert '"reflection"' in opengui_parser
    assert 'OpenGuiActionHistory()' in chat_client
    assert 'openGuiActionHistory.snapshotJson()' in chat_client
    assert 'openGuiActionHistory.clearJson()' in chat_client
    assert 'deterministic OpenGUI action history' in chat_client
    assert 'update_working_memory' in opengui_parser
    assert 'hermes_opengui_working_memory' in chat_client
    assert 'executeParsedOpenGuiWorkingMemoryUpdate' in chat_client
    assert 'MAX_OPEN_GUI_WORKING_MEMORY_CHARS' in chat_client
    assert 'OpenGuiExecutionReview.review' in chat_client
    assert 'requires_replan' in chat_client
    assert 'repeated-action and screen-state review guards' in chat_client
    assert 'OpenGUI-style sense/perception routing' in chat_client
    assert 'ui_state_hash' in chat_client
    assert 'screen_hash' in chat_client
    assert 'android:canTakeScreenshot="true"' in accessibility_config
    assert 'capture("09-compact-floating-icon")' in (
        REPO_ROOT / "android/app/src/androidTest/java/com/mobilefork/hermesagent/DeepAppUiVisualInstrumentedTest.kt"
    ).read_text(encoding="utf-8")


def test_chat_endpoint_url_normalization_and_floating_icon_are_guarded():
    endpoint_url = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/api/HermesEndpointUrl.kt"
    ).read_text(encoding="utf-8")
    api_client = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/api/HermesApiClient.kt"
    ).read_text(encoding="utf-8")
    sse_client = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/api/HermesSseClient.kt"
    ).read_text(encoding="utf-8")
    chat = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt"
    ).read_text(encoding="utf-8")
    endpoint_test = (
        REPO_ROOT / "android/app/src/test/java/com/mobilefork/hermesagent/api/HermesEndpointUrlTest.kt"
    ).read_text(encoding="utf-8")
    provider_presets = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/data/ProviderPresets.kt"
    ).read_text(encoding="utf-8")
    provider_presets_test = (
        REPO_ROOT / "android/app/src/test/java/com/mobilefork/hermesagent/ui/settings/ProviderPresetsTest.kt"
    ).read_text(encoding="utf-8")
    settings = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsScreen.kt"
    ).read_text(encoding="utf-8")
    strings = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/i18n/HermesStrings.kt"
    ).read_text(encoding="utf-8")

    assert 'object HermesEndpointUrl' in endpoint_url
    assert '"/v1/chat/completions"' in endpoint_url
    assert '"/chat/completions"' in endpoint_url
    assert 'candidate.startsWith("wss://", ignoreCase = true)' in endpoint_url
    assert 'defaultSchemeFor(candidate)' in endpoint_url
    assert 'fun openAiRuntimeBaseUrl(baseUrl: String)' in endpoint_url
    assert 'HermesEndpointUrl.normalizeBaseUrl(baseUrl)' in api_client
    assert 'HermesEndpointUrl.chatCompletionsUrl(normalizedBaseUrl)' in sse_client
    assert 'HermesEndpointUrl.openAiRuntimeBaseUrl(trimmed)' in provider_presets
    assert 'customRuntimeConfigNormalizesPastedFullEndpointForOpenAiSdk' in provider_presets_test
    assert 'HermesEndpointUrl.chatCompletionsUrl(baseUrl)' in settings
    assert '.testTag("HermesEndpointDebugPreview")' in settings
    assert 'fun customEndpointPreview(url: String)' in strings
    assert 'Hermes will try: $url' in strings
    assert 'Hermes normalizes raw hosts, /v1 URLs, and /v1/chat/completions URLs' in (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatViewModel.kt"
    ).read_text(encoding="utf-8")
    assert 'private fun HermesFloatingActionIcon(' in chat
    assert 'Brush.linearGradient' in chat
    floating_mark = (
        REPO_ROOT / "android/app/src/main/res/drawable/ic_hermes_floating_mark.xml"
    ).read_text(encoding="utf-8")
    assert 'R.drawable.ic_hermes_floating_mark' in chat
    assert 'xmlns:aapt="http://schemas.android.com/aapt"' in floating_mark
    assert 'android:type="radial"' in floating_mark
    assert 'android:type="linear"' in floating_mark
    assert '#FF7EF4FF' in floating_mark
    assert '#FFCBB7FF' in floating_mark
    assert '#FFFFD77A' in floating_mark
    assert '.testTag("HermesFloatingActionButton")' in chat
    assert 'val actionMenuScrollState = rememberScrollState()' in chat
    assert 'val ultraNarrowActionMenu = maxWidth < 220.dp' in chat
    assert '.heightIn(max = if (ultraNarrowActionMenu) 64.dp else 220.dp)' in chat
    assert '.verticalScroll(actionMenuScrollState)' in chat
    assert 'compact = true' in chat
    assert 'onActionMenuExpandedChange = { composerActionMenuOpen = it }' in chat
    assert '&& !composerActionMenuOpen' in chat
    assert 'containerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.14f)' in chat
    assert 'val narrowHeader = maxWidth < 360.dp' in chat
    assert 'val tinyVerticalViewport = maxHeight < 360.dp' in chat
    assert 'if (!tinyVerticalViewport)' in chat
    assert '.testTag("HermesChatComposerUltraNarrowControls")' in chat
    assert 'private fun ChatHeaderDisplayModeButton(' in chat
    assert '.testTag("HermesChatHistoryButton")' in chat
    assert '.testTag("HermesChatPageActionsButton")' in chat
    assert '.testTag("HermesChatMoreInputActionsButton")' in chat
    assert '.testTag("HermesChatMicButton")' in chat
    visual_test = (
        REPO_ROOT / "android/app/src/androidTest/java/com/mobilefork/hermesagent/DeepAppUiVisualInstrumentedTest.kt"
    ).read_text(encoding="utf-8")
    assert 'onNodeWithTag("HermesFloatingActionButton").assertIsDisplayed()' in visual_test
    assert 'onNodeWithTag("HermesChatMoreInputActionsButton").performClick()' in visual_test
    assert 'onNodeWithTag("HermesChatAttachImageButton").assertIsDisplayed()' in visual_test
    assert 'capture("10-compact-action-tray")' in visual_test
    assert 'compactControlsRemainReachableOnNarrowScreens' in visual_test
    assert 'capture("11-narrow-controls")' in visual_test
    assert 'customEndpointDebugPreviewNormalizesPastedUrlInSettings' in visual_test
    assert 'capture("12-custom-endpoint-debug-preview")' in visual_test
    assert 'chatInputAcceptsHumanLikeTypingWithoutLosingComposerControls' in visual_test
    assert 'promptChunks.forEach' in visual_test
    assert 'Thread.sleep(45L)' in visual_test
    assert 'capture("13-human-like-typing")' in visual_test
    assert 'ultraNarrowComposerControlsRemainReachableOnTinyScreens' in visual_test
    assert 'screenWidthDp < 220' in visual_test
    assert 'onNodeWithTag("HermesChatComposerUltraNarrowControls").assertIsDisplayed()' in visual_test
    assert 'capture("14-ultra-narrow-controls")' in visual_test
    assert 'capture("15-ultra-narrow-keyboard")' in visual_test
    assert 'screenshotHasVisibleContent(bitmap)' in visual_test
    assert 'onNodeWithText("Conversation history").assertIsDisplayed()' in visual_test
    assert 'normalizeBaseUrl_acceptsRawHttpsHostWithOpenAiPath' in endpoint_test
    assert 'normalizeBaseUrl_usesHttpForLoopbackAndLanWithoutScheme' in endpoint_test
    assert 'openAiRuntimeBaseUrl_preservesProxyPrefixAndKeepsV1ForSdkCalls' in endpoint_test


def test_android_ui_tool_reviews_repeated_opengui_actions_before_execution():
    review = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/OpenGuiExecutionReview.kt"
    ).read_text(encoding="utf-8")
    review_test = (
        REPO_ROOT / "android/app/src/test/java/com/mobilefork/hermesagent/ui/chat/OpenGuiExecutionReviewTest.kt"
    ).read_text(encoding="utf-8")

    assert 'ACTION_REPETITION_THRESHOLD = 5' in review
    assert 'CYCLE_MIN_REPETITIONS = 3' in review
    assert 'CONSECUTIVE_SCROLL_EXIT_THRESHOLD = 8' in review
    assert 'action_repetition' in review
    assert 'action_cycle' in review
    assert 'scroll_loop' in review
    assert 'screen_no_progress' in review
    assert 'screen_cycle' in review
    assert 'blockedActionJson' in review
    assert 'requires_replan' in review
    assert 'detectsRepeatedCoordinateActionsBeforeExecutingFifthAction' in review_test
    assert 'detectsTwoStepActionCycles' in review_test
    assert 'detectsLongScrollRunsWithoutCoordinates' in review_test
    assert 'detectsUnchangedScreenSnapshotsForActiveActions' in review_test
    assert 'detectsAlternatingScreenStateCycle' in review_test

    automation_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesAutomationBridge.kt"
    ).read_text(encoding="utf-8")
    assert '"class_name"' in automation_bridge
    assert '"className"' in automation_bridge
    assert '"widget_class"' in automation_bridge


def test_android_diagnostics_exposes_agent_signal_session_snapshot_for_fused_context():
    diagnostics_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt"
    ).read_text(encoding="utf-8")
    chat_client = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt"
    ).read_text(encoding="utf-8")
    diagnostic_cards = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt"
    ).read_text(encoding="utf-8")
    quick_actions = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt"
    ).read_text(encoding="utf-8")

    assert '"agent_signal_session_snapshot_report"' in diagnostics_bridge
    assert '"signal_session_snapshot_report"' in diagnostics_bridge
    assert 'agentSignalSessionSnapshotReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSignalSessionSnapshotReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalSessionSnapshotRows(' in diagnostics_bridge
    assert 'agentSignalSessionDomainRows(' in diagnostics_bridge
    assert 'agentSignalSessionActionRouteRows()' in diagnostics_bridge
    assert 'agentSignalSessionSnapshotSourceActions()' in diagnostics_bridge
    assert '"agent_signal_session_snapshot_matrix"' in diagnostics_bridge
    assert '"agent_signal_session_domain_matrix"' in diagnostics_bridge
    assert '"agent_signal_session_action_routes"' in diagnostics_bridge
    assert '"gemma_signal_session_snapshot_directives"' in diagnostics_bridge
    assert 'Agent Signal Session Snapshot' in diagnostics_bridge
    assert 'Session Domain Coverage' in diagnostics_bridge
    assert 'MediaTek launch session gate' in diagnostics_bridge
    assert 'RF coexistence session risk' in diagnostics_bridge
    assert 'agent_signal_session_snapshot_report' in chat_client
    assert '"agent_signal_session_snapshot_matrix"' in chat_client
    assert '"agent_signal_session_domain_matrix"' in chat_client
    assert '"agent_signal_session_action_routes"' in chat_client
    assert '"agent_signal_session_snapshot_matrix"' in diagnostic_cards
    assert '"agent_signal_session_domain_matrix"' in diagnostic_cards
    assert '"agent_signal_session_action_routes"' in diagnostic_cards
    assert 'id = "signal_session_snapshot"' in quick_actions
    assert 'action=agent_signal_session_snapshot_report' in quick_actions


def test_android_diagnostics_exposes_agent_signal_proof_audit_for_claim_boundaries():
    diagnostics_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt"
    ).read_text(encoding="utf-8")
    chat_client = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt"
    ).read_text(encoding="utf-8")
    diagnostic_cards = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt"
    ).read_text(encoding="utf-8")
    quick_actions = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt"
    ).read_text(encoding="utf-8")

    assert '"agent_signal_proof_audit_report"' in diagnostics_bridge
    assert '"signal_proof_audit_report"' in diagnostics_bridge
    assert 'agentSignalProofAuditReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSignalProofAuditReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalProofAuditRows(' in diagnostics_bridge
    assert 'agentSignalClaimBoundaryRows(' in diagnostics_bridge
    assert 'agentSignalProofAuditSourceActions()' in diagnostics_bridge
    assert '"agent_signal_proof_audit_matrix"' in diagnostics_bridge
    assert '"agent_signal_claim_boundary_matrix"' in diagnostics_bridge
    assert '"gemma_signal_proof_audit_directives"' in diagnostics_bridge
    assert 'Signal Proof Audit' in diagnostics_bridge
    assert 'Signal Claim Boundaries' in diagnostics_bridge
    assert 'active_evidence_present' in diagnostics_bridge
    assert 'passive_fallback_action' in diagnostics_bridge
    assert 'bridge_required' in diagnostics_bridge
    assert 'physical_device_validation_required' in diagnostics_bridge
    assert 'release_validation_required' in diagnostics_bridge
    assert 'agent_signal_proof_audit_report' in chat_client
    assert '"agent_signal_proof_audit_matrix"' in chat_client
    assert '"agent_signal_claim_boundary_matrix"' in chat_client
    assert 'signal proof audit' in chat_client
    assert '"agent_signal_proof_audit_matrix"' in diagnostic_cards
    assert '"agent_signal_claim_boundary_matrix"' in diagnostic_cards
    assert 'id = "signal_proof_audit"' in quick_actions
    assert 'action=agent_signal_proof_audit_report' in quick_actions


def test_android_diagnostics_exposes_mediatek_signal_stack_for_non_adreno_signal_context():
    diagnostics_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt"
    ).read_text(encoding="utf-8")
    chat_client = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt"
    ).read_text(encoding="utf-8")
    diagnostic_cards = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt"
    ).read_text(encoding="utf-8")
    quick_actions = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt"
    ).read_text(encoding="utf-8")

    assert '"mediatek_signal_stack_report"' in diagnostics_bridge
    assert 'mediatekSignalStackReportJson(appContext)' in diagnostics_bridge
    assert 'fun mediatekSignalStackReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'mediatekSignalStackRows(' in diagnostics_bridge
    assert 'mediatekSignalStackRouteRows(' in diagnostics_bridge
    assert 'mediatekSignalClaimBoundaryRows(' in diagnostics_bridge
    assert 'mediatekSignalStackSourceActions()' in diagnostics_bridge
    assert 'mediatekSignalStackGraphTypes()' in diagnostics_bridge
    assert '"mediatek_signal_stack_matrix"' in diagnostics_bridge
    assert '"mediatek_signal_refresh_routes"' in diagnostics_bridge
    assert '"mediatek_signal_claim_boundaries"' in diagnostics_bridge
    assert '"gemma_mediatek_signal_directives"' in diagnostics_bridge
    assert 'MediaTek Signal Stack' in diagnostics_bridge
    assert 'MediaTek Signal Routes' in diagnostics_bridge
    assert 'MediaTek Claim Boundaries' in diagnostics_bridge
    assert 'Backend policy is not a live signal' in diagnostics_bridge
    assert 'AM/FM and broad RF require bridge proof' in diagnostics_bridge
    assert 'Physical MediaTek device proof remains separate' in diagnostics_bridge
    assert 'mediatek_signal_stack_report' in chat_client
    assert 'MediaTek signal-stack reports' in chat_client
    assert 'diagnosticArguments("mediatek_signal_stack_report")' in chat_client
    assert '"mediatek_signal_stack_matrix"' in chat_client
    assert '"mediatek_signal_refresh_routes"' in chat_client
    assert '"mediatek_signal_claim_boundaries"' in chat_client
    assert '"source_graph_type"' in chat_client
    assert '"proof_action"' in chat_client
    assert '"backend_risk_action"' in chat_client
    assert '"claim_scope"' in chat_client
    assert '"mediatek_signal_stack_matrix"' in diagnostic_cards
    assert '"mediatek_signal_refresh_routes"' in diagnostic_cards
    assert '"mediatek_signal_claim_boundaries"' in diagnostic_cards
    assert 'id = "mediatek_signal_stack"' in quick_actions
    assert 'label = "MTK Signals"' in quick_actions
    assert 'action=mediatek_signal_stack_report' in quick_actions


def test_android_diagnostics_exposes_mediatek_device_validation_for_physical_phone_proof():
    diagnostics_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt"
    ).read_text(encoding="utf-8")
    chat_client = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt"
    ).read_text(encoding="utf-8")
    diagnostic_cards = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt"
    ).read_text(encoding="utf-8")
    quick_actions = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt"
    ).read_text(encoding="utf-8")

    assert '"mediatek_device_validation_report"' in diagnostics_bridge
    assert '"physical_mediatek_validation_report"' in diagnostics_bridge
    assert '"non_adreno_device_validation_report"' in diagnostics_bridge
    assert 'mediatekDeviceValidationReportJson(appContext)' in diagnostics_bridge
    assert 'fun mediatekDeviceValidationReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'mediatekDeviceValidationRows(' in diagnostics_bridge
    assert 'mediatekDeviceValidationRouteRows(' in diagnostics_bridge
    assert 'mediatekDeviceReleaseProofGateRows(' in diagnostics_bridge
    assert 'mediatekDeviceValidationSourceActions()' in diagnostics_bridge
    assert 'mediatekDeviceValidationGraphTypes()' in diagnostics_bridge
    assert '"mediatek_device_validation_matrix"' in diagnostics_bridge
    assert '"live_signal_validation_routes"' in diagnostics_bridge
    assert '"release_device_proof_gates"' in diagnostics_bridge
    assert '"gemma_device_validation_directives"' in diagnostics_bridge
    assert 'Physical MediaTek/non-Adreno identity' in diagnostics_bridge
    assert 'Live Wi-Fi analyzer proof' in diagnostics_bridge
    assert 'GitHub release APK proof' in diagnostics_bridge
    assert 'physical_device_validation_required' in diagnostics_bridge
    assert 'release_validation_required' in diagnostics_bridge
    assert 'bridge_required' in diagnostics_bridge
    assert 'claim_scope' in diagnostics_bridge
    assert 'mediatek_device_validation_report' in chat_client
    assert 'physical MediaTek/non-Adreno device validation reports' in chat_client
    assert 'diagnosticArguments("mediatek_device_validation_report")' in chat_client
    assert '"mediatek_device_validation_matrix"' in chat_client
    assert '"live_signal_validation_routes"' in chat_client
    assert '"release_device_proof_gates"' in chat_client
    assert '"gemma_device_validation_directives"' in chat_client
    assert '"mediatek_device_validation_matrix"' in diagnostic_cards
    assert '"live_signal_validation_routes"' in diagnostic_cards
    assert '"release_device_proof_gates"' in diagnostic_cards
    assert 'id = "mediatek_device_validation"' in quick_actions
    assert 'label = "Device Proof"' in quick_actions
    assert 'action=mediatek_device_validation_report' in quick_actions


def test_android_diagnostics_exposes_device_validation_evidence_export_for_phone_release_proof():
    diagnostics_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt"
    ).read_text(encoding="utf-8")
    chat_client = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt"
    ).read_text(encoding="utf-8")
    diagnostic_cards = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt"
    ).read_text(encoding="utf-8")
    quick_actions = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt"
    ).read_text(encoding="utf-8")

    assert '"device_validation_evidence_export_report"' in diagnostics_bridge
    assert '"phone_validation_evidence_export"' in diagnostics_bridge
    assert 'deviceValidationEvidenceExportReportJson(appContext)' in diagnostics_bridge
    assert 'fun deviceValidationEvidenceExportReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'deviceValidationEvidenceManifestRows(' in diagnostics_bridge
    assert 'deviceValidationRequiredArtifactRows(' in diagnostics_bridge
    assert 'deviceValidationPhoneCommandRouteRows(' in diagnostics_bridge
    assert 'deviceValidationGithubReleaseRouteRows(' in diagnostics_bridge
    assert 'deviceValidationFdroidRouteRows(' in diagnostics_bridge
    assert '"device_validation_evidence_manifest"' in diagnostics_bridge
    assert '"device_validation_required_artifacts"' in diagnostics_bridge
    assert '"phone_validation_command_routes"' in diagnostics_bridge
    assert '"github_release_evidence_routes"' in diagnostics_bridge
    assert '"fdroid_evidence_routes"' in diagnostics_bridge
    assert '"device_validation_evidence_export_bundle"' in diagnostics_bridge
    assert '"gemma_device_validation_export_directives"' in diagnostics_bridge
    assert 'Physical phone identity capture' in diagnostics_bridge
    assert 'LiteRT /health backend proof' in diagnostics_bridge
    assert 'GitHub release asset checksum proof' in diagnostics_bridge
    assert 'F-Droid metadata/Fastlane proof' in diagnostics_bridge
    assert 'adb shell getprop' in diagnostics_bridge
    assert 'capture_command' in diagnostics_bridge
    assert 'phone_validation_evidence_export' in chat_client
    assert 'device-validation evidence export bundles' in chat_client
    assert 'action=device_validation_evidence_export_report' in chat_client
    assert 'diagnosticArguments("device_validation_evidence_export_report")' in chat_client
    assert '"device_validation_evidence_manifest"' in chat_client
    assert '"device_validation_required_artifacts"' in chat_client
    assert '"phone_validation_command_routes"' in chat_client
    assert '"github_release_evidence_routes"' in chat_client
    assert '"fdroid_evidence_routes"' in chat_client
    assert '"gemma_device_validation_export_directives"' in chat_client
    assert '"device_validation_evidence_manifest"' in diagnostic_cards
    assert '"device_validation_required_artifacts"' in diagnostic_cards
    assert '"phone_validation_command_routes"' in diagnostic_cards
    assert '"github_release_evidence_routes"' in diagnostic_cards
    assert '"fdroid_evidence_routes"' in diagnostic_cards
    assert 'id = "device_evidence_export"' in quick_actions
    assert 'label = "Proof Export"' in quick_actions
    assert 'action=device_validation_evidence_export_report' in quick_actions


def test_android_diagnostics_exposes_signal_observation_packet_for_gemma_visible_top_cards():
    diagnostics_bridge = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt"
    ).read_text(encoding="utf-8")
    chat_client = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt"
    ).read_text(encoding="utf-8")
    diagnostic_cards = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt"
    ).read_text(encoding="utf-8")
    quick_actions = (
        REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt"
    ).read_text(encoding="utf-8")

    assert '"agent_signal_observation_packet_report"' in diagnostics_bridge
    assert '"gemma_signal_observation_packet"' in diagnostics_bridge
    assert 'agentSignalObservationPacketReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentSignalObservationPacketReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentSignalObservationPacketRows(' in diagnostics_bridge
    assert 'agentSignalObservationVisualSlotRows(' in diagnostics_bridge
    assert 'agentSignalObservationGraphRouteRows(' in diagnostics_bridge
    assert 'agentSignalObservationClaimBoundaryRows(' in diagnostics_bridge
    assert '"agent_signal_observation_packet"' in diagnostics_bridge
    assert '"agent_signal_observation_visual_slots"' in diagnostics_bridge
    assert '"agent_signal_observation_graph_routes"' in diagnostics_bridge
    assert '"agent_signal_observation_claim_boundaries"' in diagnostics_bridge
    assert '"agent_signal_observation_packet_bundle"' in diagnostics_bridge
    assert '"gemma_signal_observation_packet_directives"' in diagnostics_bridge
    assert 'Wi-Fi Analyzer observation packet' in diagnostics_bridge
    assert 'Bluetooth proximity observation packet' in diagnostics_bridge
    assert 'AM/FM and SDR radio observation packet' in diagnostics_bridge
    assert 'wifi_analyzer_parity_keys' in diagnostics_bridge
    assert 'kai_parity_keys' in diagnostics_bridge
    assert 'agent_signal_observation_packet_report' in chat_client
    assert 'compact signal observation packets' in chat_client
    assert 'diagnosticArguments("agent_signal_observation_packet_report")' in chat_client
    assert '"agent_signal_observation_packet"' in chat_client
    assert '"agent_signal_observation_visual_slots"' in chat_client
    assert '"agent_signal_observation_graph_routes"' in chat_client
    assert '"agent_signal_observation_claim_boundaries"' in chat_client
    assert '"gemma_signal_observation_packet_directives"' in chat_client
    assert '"agent_signal_observation_packet"' in diagnostic_cards
    assert '"agent_signal_observation_visual_slots"' in diagnostic_cards
    assert '"agent_signal_observation_graph_routes"' in diagnostic_cards
    assert '"agent_signal_observation_claim_boundaries"' in diagnostic_cards
    assert 'id = "signal_observation_packet"' in quick_actions
    assert 'label = "Sight Packet"' in quick_actions
    assert 'action=agent_signal_observation_packet_report' in quick_actions
