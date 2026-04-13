from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_android_boot_and_chat_paths_guard_local_backend_failures_instead_of_crashing():
    boot_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/boot/BootViewModel.kt").read_text(encoding="utf-8")
    chat_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatViewModel.kt").read_text(encoding="utf-8")
    sse_client = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/api/HermesSseClient.kt").read_text(encoding="utf-8")

    assert 'runCatching {' in boot_view_model
    assert 'checkHealth(runtime.baseUrl, runtime.apiKey)' in boot_view_model
    assert 'Hermes backend health check failed' in boot_view_model

    assert 'runCatching {' in chat_view_model
    assert 'client.streamChatCompletion(' in chat_view_model
    assert 'error.message ?: error.javaClass.simpleName' in chat_view_model

    assert 'internal fun parseStream(' in sse_client
    assert 'parseStream(source, onDelta, onComplete, onError)' in sse_client
    assert 'runCatching { extractDelta(payload) }' in sse_client
    assert 'catch (error: Exception)' in sse_client
