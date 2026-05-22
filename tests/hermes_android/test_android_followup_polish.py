from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_localization_layer_covers_visible_chat_auth_portal_device_and_settings_copy():
    strings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")
    chat = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
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
    assert 'com.google.ai.edge.litertlm:litertlm-android:0.11.0' in gradle
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
    automation_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesAutomationBridge.kt").read_text(encoding="utf-8")

    assert '"agent_environment_report"' in diagnostics_bridge
    assert 'agentEnvironmentReportJson(appContext)' in diagnostics_bridge
    assert 'agentCapabilityMatrixRows(' in diagnostics_bridge
    assert 'kaiParityMatrixRows(' in diagnostics_bridge
    assert 'kaiOperationsMatrixRows(' in diagnostics_bridge
    assert 'workflowReadinessRows(' in diagnostics_bridge
    assert '"agent_capability_matrix"' in diagnostics_bridge
    assert '"kai_parity_matrix"' in diagnostics_bridge
    assert '"kai_operations_matrix"' in diagnostics_bridge
    assert '"workflow_readiness_matrix"' in diagnostics_bridge
    assert 'Multi-provider priority and fallback' in diagnostics_bridge
    assert 'Tool and MCP bridge route' in diagnostics_bridge
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
    assert '"kai_operations_matrix"' in diagnostic_cards
    assert '"agent_capability_matrix", "kai_parity_matrix", "agent_workflow_readiness"' in diagnostic_cards
    assert 'capabilityMatrixRow(row)' in diagnostic_cards
    assert '"schedule_task", "kai_schedule_task" -> scheduleTaskJson(context, arguments)' in automation_bridge
    assert '"list_tasks", "kai_list_tasks", "tasks" -> listTasksJson(context)' in automation_bridge
    assert '"cancel_task", "kai_cancel_task" -> cancelTaskJson(context, arguments)' in automation_bridge
    assert '"android_automation_task_not_background_ai_prompt"' in automation_bridge
    assert '"android_automation_notification_task"' in automation_bridge


def test_android_diagnostics_exposes_agent_observation_dashboard_for_gemma_signal_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")
    quick_actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")

    assert '"agent_observation_report"' in diagnostics_bridge
    assert 'agentObservationReportJson(appContext)' in diagnostics_bridge
    assert '"agent_card_manifest_report"' in diagnostics_bridge
    assert 'agentCardManifestReportJson(appContext)' in diagnostics_bridge
    assert 'fun agentObservationReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'fun agentCardManifestReportJson(context: Context): JSONObject' in diagnostics_bridge
    assert 'agentObservationMatrixRows(' in diagnostics_bridge
    assert 'agentObservationRouteRows()' in diagnostics_bridge
    assert 'agentCardManifestRows(' in diagnostics_bridge
    assert 'agentCardManifestSources(' in diagnostics_bridge
    assert '"agent_observation_matrix"' in diagnostics_bridge
    assert '"agent_observation_routes"' in diagnostics_bridge
    assert '"agent_card_manifest"' in diagnostics_bridge
    assert 'Agent Card Manifest' in diagnostics_bridge
    assert '"gemma_observation_directives"' in diagnostics_bridge
    assert 'Wi-Fi AP metadata and channel graphs' in diagnostics_bridge
    assert 'Bluetooth nearby metadata' in diagnostics_bridge
    assert 'Motion and sensor context' in diagnostics_bridge
    assert 'Radio and RF boundaries' in diagnostics_bridge
    assert 'Kai operations and interactive routes' in diagnostics_bridge
    assert 'agent_observation_report' in chat_client
    assert 'agent_card_manifest_report' in chat_client
    assert '"agent_observation_matrix"' in chat_client
    assert '"agent_observation_routes"' in chat_client
    assert '"agent_card_manifest"' in chat_client
    assert '"agent_card_graph_types"' in chat_client
    assert '"gemma_observation_directives"' in chat_client
    assert '"agent_observation_matrix", "agent_observation_routes"' in diagnostic_cards
    assert '"agent_card_manifest",' in diagnostic_cards
    assert 'id = "agent_observation"' in quick_actions
    assert 'action=agent_observation_report' in quick_actions
    assert 'id = "card_manifest"' in quick_actions
    assert 'action=agent_card_manifest_report' in quick_actions


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
    assert '"AM/FM Signal Graph"' in diagnostics_bridge
    assert 'radioBandPlanRows(' in diagnostics_bridge
    assert 'cached_wifi_signal_history' in diagnostics_bridge
    assert 'Route broad RF explanation' in diagnostics_bridge
    assert 'signal_awareness_report' in chat_client
    assert 'radio_analyzer_report' in chat_client
    assert 'radio_signal_graph' in chat_client
    assert '"signal_awareness_matrix"' in chat_client
    assert '"signal_workflow_routes"' in chat_client
    assert '"signal_constraint_matrix"' in chat_client
    assert '"radio_signal_feature_matrix"' in chat_client
    assert '"radio_signal_workflow_routes"' in chat_client
    assert '"radio_signal_constraint_matrix"' in chat_client
    assert '"radio_signal_graph_rows"' in chat_client
    assert '"radio_signal_graph_sample_rows"' in chat_client
    assert '"signal_awareness_matrix", "signal_workflow_routes", "signal_constraint_matrix",' in diagnostic_cards
    assert '"radio_signal_feature_matrix", "radio_signal_workflow_routes", "radio_signal_constraint_matrix",' in diagnostic_cards
    assert '"radio_signal_graph" -> radioSignalGraphRow(row)' in diagnostic_cards
    assert 'radioSignalGraphRow(' in diagnostic_cards
    assert 'diagnosticAction = "radio_signal_graph"' in quick_actions


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
    assert 'gpuBackendRiskMatrixRows(' in diagnostics_bridge
    assert 'gpuBackendRiskRouteRows(' in diagnostics_bridge
    assert '"gpu_backend_risk_matrix"' in diagnostics_bridge
    assert '"gpu_backend_risk_routes"' in diagnostics_bridge
    assert 'Live accelerator acceptance' in diagnostics_bridge
    assert 'Phone validation scope' in diagnostics_bridge
    assert '"local_backend_runtime_report"' in diagnostics_bridge
    assert 'localBackendRuntimeReportJson(appContext)' in diagnostics_bridge
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
    assert 'soc_compatibility_report' in chat_client
    assert 'gpu_backend_risk_report' in chat_client
    assert 'device_performance_report' in chat_client
    assert '"runtime_backend_matrix"' in chat_client
    assert '"runtime_stability_matrix"' in chat_client
    assert '"soc_backend_matrix"' in chat_client
    assert '"soc_backend_policy_routes"' in chat_client
    assert '"soc_backend_constraint_matrix"' in chat_client
    assert '"gpu_backend_risk_matrix"' in chat_client
    assert '"gpu_backend_risk_routes"' in chat_client
    assert '"gpu_backend_risk_matrix", "gpu_backend_risk_routes",' in diagnostic_cards
    assert '"runtime_backend_matrix", "runtime_stability_matrix" -> capabilityMatrixRow(row)' in diagnostic_cards
    assert 'id = "runtime_backend"' in quick_actions
    assert 'id = "runtime_stability"' in quick_actions
    assert 'id = "soc_compatibility"' in quick_actions
    assert 'id = "backend_risk"' in quick_actions


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
    workflow = (REPO_ROOT / ".github/workflows/android.yml").read_text(encoding="utf-8")

    assert '"wifi_analyzer_report"' in diagnostics_bridge
    assert '"wifi_channel_graph"' in diagnostics_bridge
    assert '"wifi_channel_utilization"' in diagnostics_bridge
    assert 'wifiAnalyzerReportJson(appContext' in diagnostics_bridge
    assert 'wifiFilteredNetworkRows(' in diagnostics_bridge
    assert '"wifi_filtered_scan"' in diagnostics_bridge
    assert '"applied_wifi_filters"' in diagnostics_bridge
    assert '"wifi_filter_application"' in diagnostics_bridge
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
    assert 'scan_mode=paused' in diagnostics_bridge
    assert 'scan_mode=resumed' in diagnostics_bridge
    assert 'Pause/resume scan control' in diagnostics_bridge
    assert 'Route pause or resume scan mode' in diagnostics_bridge
    assert '"wifi_channel_utilization"' in diagnostics_bridge
    assert '"wifi_access_point_semantics"' in diagnostics_bridge
    assert '"wifi_band_coverage"' in diagnostics_bridge
    assert 'WiFiAnalyzer-style readiness' in diagnostics_bridge
    assert 'Channel signal graph' in diagnostics_bridge
    assert 'Channel utilization occupancy' in diagnostics_bridge
    assert 'Agent AP semantic and risk labels' in diagnostics_bridge
    assert 'Band coverage and 2.4/5/6GHz visibility' in diagnostics_bridge
    assert 'wifi_analyzer_report' in chat_client
    assert 'wifi_filtered_scan' in chat_client
    assert 'wifi_channel_graph' in chat_client
    assert '"filter_band"' in chat_client
    assert '"filter_security"' in chat_client
    assert '"filter_signal"' in chat_client
    assert '"filter_ssid"' in chat_client
    assert '"min_rssi_dbm"' in chat_client
    assert '"applied_wifi_filters"' in chat_client
    assert '"wifi_filter_application"' in chat_client
    assert 'wifi_channel_utilization' in chat_client
    assert '"wifi_analyzer_feature_matrix"' in chat_client
    assert '"wifi_channel_utilization"' in chat_client
    assert '"wifi_channel_graph"' in chat_client
    assert '"wifi_access_point_semantics"' in chat_client
    assert '"wifi_band_coverage"' in chat_client
    assert '"wifi_analyzer_workflow_routes"' in chat_client
    assert '"wifi_scan_policy_matrix"' in chat_client
    assert '"wifi_scan_control"' in chat_client
    assert '"scan_mode"' in chat_client
    assert 'Wi-Fi or Bluetooth scan mode for direct signal actions' in chat_client
    assert '"wifi_channel_graph" -> wifiChannelGraphRow(row)' in diagnostic_cards
    assert '"wifi_channel_utilization" -> wifiChannelUtilizationRow(row)' in diagnostic_cards
    assert '"wifi_access_point_semantics" -> wifiAccessPointSemanticRow(row)' in diagnostic_cards
    assert '"wifi_band_coverage" -> wifiBandCoverageRow(row)' in diagnostic_cards
    assert '"wifi_analyzer_feature_matrix", "wifi_analyzer_workflow_routes", "wifi_scan_policy_matrix"' in diagnostic_cards
    assert '"wifi_filter_application"' in diagnostic_cards
    assert 'capabilityMatrixRow(row)' in diagnostic_cards
    assert ':app:compileDebugAndroidTestKotlin' in workflow


def test_android_diagnostics_exposes_bluetooth_analyzer_report_for_readiness_and_scan_policy_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")

    assert '"bluetooth_analyzer_report"' in diagnostics_bridge
    assert '"bluetooth_signal_history"' in diagnostics_bridge
    assert 'mergeBluetoothSignalHistory(' in diagnostics_bridge
    assert 'bluetoothSignalHistoryRowsFromStore(' in diagnostics_bridge
    assert 'bluetoothAnalyzerReportJson(appContext' in diagnostics_bridge
    assert 'bluetoothAnalyzerFeatureRows(' in diagnostics_bridge
    assert 'bluetoothAnalyzerWorkflowRows(' in diagnostics_bridge
    assert 'bluetoothScanPolicyRows(' in diagnostics_bridge
    assert 'bluetoothServiceUuidLabel(' in diagnostics_bridge
    assert 'bluetoothManufacturerIdLabel(' in diagnostics_bridge
    assert '"service_labels"' in diagnostics_bridge
    assert '"manufacturer_names"' in diagnostics_bridge
    assert 'Bluetooth SIG service labels' in diagnostics_bridge
    assert '"bluetooth_signal_history"' in diagnostics_bridge
    assert '"bluetooth_analyzer_feature_matrix"' in diagnostics_bridge
    assert '"bluetooth_analyzer_workflow_routes"' in diagnostics_bridge
    assert '"bluetooth_scan_policy_matrix"' in diagnostics_bridge
    assert '"bluetooth_scan_control"' in diagnostics_bridge
    assert 'Pause/resume BLE scan control' in diagnostics_bridge
    assert 'Route pause or resume BLE scan mode' in diagnostics_bridge
    assert 'scan_mode=paused' in diagnostics_bridge
    assert 'scan_mode=resumed' in diagnostics_bridge
    assert 'Bluetooth Analyzer readiness' in diagnostics_bridge
    assert 'bluetooth_analyzer_report' in chat_client
    assert 'bluetooth_signal_history' in chat_client
    assert '"bluetooth_service_label_count"' in chat_client
    assert '"bluetooth_manufacturer_name_count"' in chat_client
    assert '"service_labels"' in chat_client
    assert '"manufacturer_names"' in chat_client
    assert '"bluetooth_signal_history"' in chat_client
    assert '"bluetooth_analyzer_feature_matrix"' in chat_client
    assert '"bluetooth_analyzer_workflow_routes"' in chat_client
    assert '"bluetooth_scan_policy_matrix"' in chat_client
    assert '"bluetooth_scan_control"' in chat_client
    assert 'Wi-Fi or Bluetooth scan mode for direct signal actions' in chat_client
    assert '"bluetooth_signal_history" -> bluetoothSignalHistoryRow(row)' in diagnostic_cards
    assert '"bluetooth_analyzer_feature_matrix", "bluetooth_analyzer_workflow_routes", "bluetooth_scan_policy_matrix"' in diagnostic_cards
    assert 'capabilityMatrixRow(row)' in diagnostic_cards


def test_android_diagnostics_exposes_sensor_analyzer_report_for_motion_and_sampling_policy_cards():
    diagnostics_bridge = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/device/HermesDeviceDiagnosticsBridge.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")

    assert '"sensor_analyzer_report"' in diagnostics_bridge
    assert '"motion_sensor_history"' in diagnostics_bridge
    assert 'mergeMotionSensorHistory(' in diagnostics_bridge
    assert 'motionSensorHistoryRowsFromStore(' in diagnostics_bridge
    assert 'sensorAnalyzerReportJson(appContext' in diagnostics_bridge
    assert 'sensorAnalyzerFeatureRows(' in diagnostics_bridge
    assert 'sensorAnalyzerWorkflowRows(' in diagnostics_bridge
    assert 'sensorSamplingPolicyRows(' in diagnostics_bridge
    assert 'motionPoseEstimateRows(' in diagnostics_bridge
    assert '"sensor_analyzer_feature_matrix"' in diagnostics_bridge
    assert '"sensor_analyzer_workflow_routes"' in diagnostics_bridge
    assert '"sensor_sampling_policy_matrix"' in diagnostics_bridge
    assert '"motion_pose_estimates"' in diagnostics_bridge
    assert '"motion_pose_estimate"' in diagnostics_bridge
    assert '"motion_pose"' in diagnostics_bridge
    assert 'Sensor Analyzer readiness' in diagnostics_bridge
    assert 'accelerometer' in diagnostics_bridge
    assert 'gyroscope' in diagnostics_bridge
    assert 'sensor_analyzer_report' in chat_client
    assert 'motion_sensor_history' in chat_client
    assert 'motion_pose' in chat_client
    assert '"motion_pose_estimates"' in chat_client
    assert '"sensor_analyzer_feature_matrix"' in chat_client
    assert '"sensor_analyzer_workflow_routes"' in chat_client
    assert '"sensor_sampling_policy_matrix"' in chat_client
    assert '"sensor_analyzer_feature_matrix", "sensor_analyzer_workflow_routes", "sensor_sampling_policy_matrix"' in diagnostic_cards
    assert '"motion_sensor_history" -> motionSensorHistoryRow(row)' in diagnostic_cards
    assert '"motion_pose_estimate" -> motionPoseEstimateRow(row)' in diagnostic_cards
    assert 'capabilityMatrixRow(row)' in diagnostic_cards


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
