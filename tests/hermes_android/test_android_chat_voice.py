from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_manifest_and_chat_use_android_voice_input():
    manifest = (REPO_ROOT / "android/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    speech_controller = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/SpeechInputController.kt").read_text(encoding="utf-8")

    assert 'android.permission.RECORD_AUDIO' in manifest
    assert 'ActivityResultContracts.RequestPermission()' in chat_screen
    assert 'SpeechInputController.buildIntent()' in chat_screen
    assert 'Voice recognition is not available on this device' in chat_screen
    assert 'RecognizerIntent.ACTION_RECOGNIZE_SPEECH' in speech_controller


def test_chat_supports_tts_playback_for_assistant_replies():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    tts_controller = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/HermesTtsController.kt").read_text(encoding="utf-8")

    assert 'HermesTtsController' in chat_screen
    assert 'Speak reply' in chat_screen
    assert 'TextToSpeech' in tts_controller
    assert 'fun speak(text: String): Boolean' in tts_controller
    assert 'textToSpeech.speak' in tts_controller
