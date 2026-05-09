from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_android_boot_and_chat_paths_guard_local_backend_failures_instead_of_crashing():
    application = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/HermesApplication.kt").read_text(encoding="utf-8")
    runtime_service = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/HermesRuntimeService.kt").read_text(encoding="utf-8")
    boot_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/boot/BootViewModel.kt").read_text(encoding="utf-8")
    chat_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatViewModel.kt").read_text(encoding="utf-8")
    native_tool_client = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")
    sse_client = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/api/HermesSseClient.kt").read_text(encoding="utf-8")

    assert "BACKGROUND_RUNTIME_STARTUP_DELAY_MS = 1500L" in application
    assert "DeviceStateWriter.write(this@HermesApplication)" in application
    assert "delay(BACKGROUND_RUNTIME_STARTUP_DELAY_MS)" in application

    assert "CoroutineScope(SupervisorJob() + Dispatchers.IO)" in runtime_service
    assert "promoteToForeground(runtime = null)" in runtime_service
    assert "serviceScope.launch {" in runtime_service
    assert "HermesRuntimeManager.ensureStarted(applicationContext)" in runtime_service
    assert "context.startService(intent)" in runtime_service
    assert "ContextCompat.startForegroundService(context, intent)" in runtime_service

    assert 'runCatching {' in boot_view_model
    assert 'startupDelayMillis = if (firstRefresh) 1000L else 0L' in boot_view_model
    assert 'delay(startupDelayMillis)' in boot_view_model
    assert 'checkHealth(runtime.baseUrl, runtime.apiKey)' in boot_view_model
    assert 'Hermes backend health check failed' in boot_view_model

    assert 'runCatching {' in chat_view_model
    assert 'client.streamChatCompletion(' in chat_view_model
    assert 'error.message ?: error.javaClass.simpleName' in chat_view_model

    assert "toolSpecs = null" in native_tool_client
    assert "followUp.content.ifBlank { toolCompletionReply(latestToolResult) }" in native_tool_client

    assert 'internal fun parseStream(' in sse_client
    assert 'parseStream(source, onDelta, onComplete, onError)' in sse_client
    assert 'runCatching { extractDelta(payload) }' in sse_client
    assert 'catch (error: Exception)' in sse_client


def test_android_chat_ui_and_native_tool_prompt_stay_compact_on_large_font_phone_screens():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")
    native_tool_client = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")

    assert 'placeholder = {' in chat_screen
    assert 'label = { Text(strings.messageHermes' not in chat_screen
    assert '.fillMaxWidth()\n                    .testTag("HermesChatSendButton")' in chat_screen
    assert 'TextOverflow.Ellipsis' in chat_screen
    assert 'softWrap = false' in app_shell
    assert 'style = MaterialTheme.typography.labelSmall' in app_shell
    assert 'compactToolSpecsFor(userText)' in native_tool_client
    assert 'explicitlyRequestedToolNames(userText)' in native_tool_client
    assert 'formatNativeChatError' in native_tool_client
    assert 'The local model ran out of context' in native_tool_client
    assert 'Native chat request failed: ${response.code} $body' not in native_tool_client
