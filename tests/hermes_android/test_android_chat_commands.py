from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chat_command_router_supports_native_navigation_and_auth_commands():
    router = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatCommandRouter.kt").read_text(encoding="utf-8")

    for command in [
        '/help',
        '/new',
        '/history',
        '/clear',
        '/accounts',
        '/settings',
        '/device',
        '/portal',
        '/provider',
        '/model',
        '/signin',
        '/speak',
    ]:
        assert command in router
    assert 'chatgpt|claude|gemini|google|email|phone' in router
    assert 'applyProvider' in router
    assert 'applyModel' in router
    assert 'startAuthMethod' in router
    assert 'speakLastReply' in router
