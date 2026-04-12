from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chat_screen_has_bubbles_history_and_action_icons():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")

    assert 'ConversationHistoryList(' in chat_screen
    assert 'ChatBubble(' in chat_screen
    assert 'R.drawable.ic_action_history' in chat_screen
    assert 'R.drawable.ic_action_mic' in chat_screen
    assert 'R.drawable.ic_action_speaker' in chat_screen
    assert 'R.drawable.ic_action_cog' in chat_screen
    assert 'onOpenContextActions' in chat_screen
    assert 'Message Hermes' in chat_screen
    assert 'Speak last reply' in chat_screen
    assert 'Available app commands:' in (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatCommandRouter.kt").read_text(encoding="utf-8")


def test_conversation_store_tracks_multiple_sessions_and_messages():
    conversation_store = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/ConversationStore.kt").read_text(encoding="utf-8")

    assert 'data class StoredConversationMessage' in conversation_store
    assert 'data class ConversationSummary' in conversation_store
    assert 'fun listConversationSummaries()' in conversation_store
    assert 'fun createNewConversation(' in conversation_store
    assert 'fun upsertMessage(' in conversation_store
    assert 'fun updateMessageContent(' in conversation_store
    assert 'conversations_json' in conversation_store


def test_chat_view_model_persists_history_and_supports_native_command_feedback():
    chat_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatViewModel.kt").read_text(encoding="utf-8")

    assert 'fun showHistory()' in chat_view_model
    assert 'fun openConversation(' in chat_view_model
    assert 'fun startNewConversation()' in chat_view_model
    assert 'fun consumeCommandResult(' in chat_view_model
    assert 'Voice input captured' in chat_view_model
    assert 'Speaking the latest Hermes reply' not in chat_view_model  # UI handles TTS feedback



def test_empty_chat_layout_centers_welcome_state_instead_of_leaving_blank_gap():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")

    assert 'Box(' in chat_screen
    assert 'contentAlignment = Alignment.Center' in chat_screen
