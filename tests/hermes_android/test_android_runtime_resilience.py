from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_android_boot_and_chat_paths_guard_local_backend_failures_instead_of_crashing():
    application = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/HermesApplication.kt").read_text(encoding="utf-8")
    runtime_service = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/backend/HermesRuntimeService.kt").read_text(encoding="utf-8")
    boot_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/boot/BootViewModel.kt").read_text(encoding="utf-8")
    boot_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/boot/BootScreen.kt").read_text(encoding="utf-8")
    chat_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatViewModel.kt").read_text(encoding="utf-8")
    native_tool_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    sse_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/api/HermesSseClient.kt").read_text(encoding="utf-8")

    assert "class HermesApplication : Application()" in application
    assert "instance = this" in application
    assert "BACKGROUND_RUNTIME_STARTUP_DELAY_MS" not in application
    assert "STARTUP_BACKGROUND_WORK_DELAY_MS" not in application
    assert "DeviceStateWriter.write" not in application
    assert "HermesRuntimeManager.ensureStarted" not in application

    assert "CoroutineScope(SupervisorJob() + Dispatchers.IO)" in runtime_service
    assert "promoteToForeground(runtime = null)" in runtime_service
    assert "serviceScope.launch {" in runtime_service
    assert "HermesRuntimeManager.ensureStarted(applicationContext)" in runtime_service
    assert "context.startService(intent)" in runtime_service
    assert "ContextCompat.startForegroundService(context, intent)" in runtime_service

    assert 'startupDelayMillis = if (firstRefresh) FIRST_SHELL_REFRESH_DELAY_MS else 0L' in boot_view_model
    assert 'private const val FIRST_SHELL_REFRESH_DELAY_MS = 150L' in boot_view_model
    assert 'delay(startupDelayMillis)' in boot_view_model
    assert 'BootUiState(status = "Hermes shell ready", ready = true)' in boot_view_model
    assert 'checkHealth(' not in boot_view_model
    assert 'HermesRuntimeManager.ensureStarted' not in boot_view_model
    assert 'init {' not in boot_view_model
    assert 'LaunchedEffect(Unit)' in boot_screen
    assert 'withFrameNanos { }' in boot_screen
    assert 'viewModel.refresh()' in boot_screen

    assert 'runCatching {' in chat_view_model
    assert 'client.streamChatCompletion(' in chat_view_model
    assert 'error.message ?: error.javaClass.simpleName' in chat_view_model

    assert "postChatCompletionWithContextRecovery(" in native_tool_client
    assert "toolSpecs = activeToolSpecs" in native_tool_client
    assert "recoverMessagesAfterContextOverflow(messages)" in native_tool_client
    assert "recoverToolSpecsAfterContextOverflow(toolSpecs)" in native_tool_client
    assert "followUp.content.ifBlank { toolCompletionReply(latestToolResult) }" in native_tool_client
    assert "nativeVisibleReplyContent(rawContent, latestToolResult)" in native_tool_client
    assert "isCompressionOnlyToolEcho(content: String)" in native_tool_client
    assert 'removePrefix("tool call completed:")' in native_tool_client
    assert '"available_system_actions"' in native_tool_client
    assert "androidSystemStatusReply(parsed)?.let { return it }" in native_tool_client
    assert '"available_privileged_actions"' in native_tool_client

    assert 'internal fun parseStream(' in sse_client
    assert 'parseStream(source, onDelta, onComplete, onError, onStatus)' in sse_client
    assert 'runCatching { extractStreamEvent(payload) }' in sse_client
    assert 'catch (error: Exception)' in sse_client


def test_android_python_runtime_smoke_resets_local_backend_selection_before_remote_runtime_probe():
    runtime_smoke = (REPO_ROOT / "android/app/src/androidTest/java/com/mobilefork/hermesagent/NativeAgentRuntimeSmokeTest.kt").read_text(encoding="utf-8")

    assert 'AppSettingsStore(context).let { store ->' in runtime_smoke
    assert 'onDeviceBackend = BackendKind.NONE.persistedValue' in runtime_smoke


def test_main_activity_keeps_device_state_refresh_off_the_startup_thread():
    main_activity = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/MainActivity.kt").read_text(encoding="utf-8")

    assert "HermesCrashLogStore.install(applicationContext)" in main_activity
    assert "private fun handleShortcutIntent(intent: Intent?)" in main_activity
    assert "HermesLauncherShortcutBridge.handleShortcutIntentJson(applicationContext, intent)" in main_activity
    assert "writeDeviceStateAsync" not in main_activity
    assert "STARTUP_DEVICE_STATE_DELAY_MS" not in main_activity
    assert "DeviceStateWriter.write(applicationContext)" not in main_activity


def test_android_chat_ui_and_native_tool_prompt_stay_compact_on_large_font_phone_screens():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")
    native_tool_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")

    assert 'placeholder = {' in chat_screen
    assert 'label = { Text(strings.messageHermes' not in chat_screen
    assert '.fillMaxWidth()\n                    .testTag("HermesChatSendButton")' in chat_screen
    assert 'TextOverflow.Ellipsis' in chat_screen
    assert 'modifier = Modifier.size(22.dp)' in app_shell
    assert 'style = MaterialTheme.typography.bodyLarge' in app_shell
    assert 'compactToolSpecsFor(userText)' in native_tool_client
    assert '.ifEmpty { inferredToolNames(userText) }' in native_tool_client
    assert 'return JSONArray()' in native_tool_client
    assert 'toolsEnabled = activeToolSpecs.length() > 0' in native_tool_client
    assert 'relevantMemoryContext = relevantMemoryContext' in native_tool_client
    assert 'compactCustomSystemPrompt' in native_tool_client
    assert 'Keep replies brief and direct.' in native_tool_client
    assert 'inferredToolNames(userText: String)' in native_tool_client
    assert '"launch browser"' in native_tool_client
    assert '"browse url"' in native_tool_client
    assert '"browser" in lower' in native_tool_client
    assert 'create_intent_task: start_activity, open_uri, or send_broadcast' in native_tool_client
    assert 'explicitlyRequestedToolNames(userText)' in native_tool_client
    assert 'formatNativeChatError' in native_tool_client
    assert 'The local model ran out of context' in native_tool_client
    assert 'Native chat request failed: ${response.code} $body' not in native_tool_client


def test_chat_streaming_stays_pinned_to_latest_message_bottom_anchor():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")

    assert "latestMessageFingerprint" in chat_screen
    assert "${message.id}:${message.role}:${message.content.length}:${uiState.messages.size}" in chat_screen
    assert "LaunchedEffect(latestMessageFingerprint, uiState.isSending, uiState.isShowingHistory, chatDisplayMode)" in chat_screen
    assert "listState.scrollToItem(targetIndex)" in chat_screen
    assert "listState.animateScrollToItem(targetIndex)" in chat_screen
    assert 'item(key = "HermesChatBottomAnchor")' in chat_screen
    assert 'testTag("HermesChatBottomAnchor")' in chat_screen
