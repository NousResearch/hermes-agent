from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chat_screen_has_bubbles_history_and_action_icons():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")

    assert 'ConversationHistoryList(' in chat_screen
    assert 'ChatBubble(' in chat_screen
    assert 'R.drawable.ic_action_history' in chat_screen
    assert 'R.drawable.ic_action_mic' in chat_screen
    assert 'R.drawable.ic_action_image' in chat_screen
    assert 'HermesChatAttachImageButton' in chat_screen
    assert 'HermesChatAttachments' in chat_screen
    assert 'R.drawable.ic_action_speaker' in chat_screen
    assert 'R.drawable.ic_action_cog' in chat_screen
    assert 'onOpenContextActions' in chat_screen
    assert 'remember(strings.language' in chat_screen
    assert 'onContextActionsChanged(shellActions)' in chat_screen
    assert 'SignalIntelligenceQuickActionGrid(' in chat_screen
    assert 'HermesSignalQuickActions' in chat_screen
    strings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")

    assert 'strings.messageHermes' in chat_screen
    assert 'Message Hermes Fork' in strings
    assert 'Speak last reply' in chat_screen
    assert 'Available app commands:' in (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatCommandRouter.kt").read_text(encoding="utf-8")


def test_conversation_store_tracks_multiple_sessions_and_messages():
    conversation_store = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/data/ConversationStore.kt").read_text(encoding="utf-8")

    assert 'data class StoredConversationMessage' in conversation_store
    assert 'data class ConversationSummary' in conversation_store
    assert 'fun listConversationSummaries()' in conversation_store
    assert 'fun createNewConversation(' in conversation_store
    assert 'fun upsertMessage(' in conversation_store
    assert 'fun updateMessageContent(' in conversation_store
    assert 'conversations_json' in conversation_store


def test_chat_view_model_persists_history_and_supports_native_command_feedback():
    chat_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatViewModel.kt").read_text(encoding="utf-8")

    assert 'fun showHistory()' in chat_view_model
    assert 'fun openConversation(' in chat_view_model
    assert 'fun startNewConversation()' in chat_view_model
    assert 'fun consumeCommandResult(' in chat_view_model
    assert 'Voice input captured' in chat_view_model
    assert 'Speaking the latest Hermes reply' not in chat_view_model  # UI handles TTS feedback



def test_empty_chat_layout_scrolls_welcome_state_on_small_or_large_font_screens():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")

    assert 'LazyColumn(' in chat_screen
    assert 'EmptyChatHint(' in chat_screen
    assert 'contentPadding = PaddingValues(vertical = 8.dp)' in chat_screen


def test_signal_intelligence_quick_actions_launch_direct_diagnostic_cards():
    actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    chat_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatViewModel.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")

    for action in [
        "signal_awareness_report",
        "agent_signal_evidence_report",
        "agent_observation_report",
        "agent_environment_report",
        "soc_compatibility_report",
        "gpu_backend_risk_report",
        "local_inference_compatibility_report",
        "wifi_analyzer_report",
        "wifi_scan",
        "wifi_channel_utilization",
        "bluetooth_analyzer_report",
        "bluetooth_signal_history",
        "sensor_analyzer_report",
        "motion_sensor_history",
        "radio_signal_graph",
    ]:
        assert f"action={action}" in actions
        assert f'"{action}"' in chat_client

    assert "sendQuickPrompt" in chat_screen
    assert "fun sendQuickPrompt" in chat_view_model
    assert "extractExplicitAndroidDiagnosticsArguments(userText)" in chat_client
    assert 'testTag("HermesSignalQuickAction_${action.id}")' in chat_screen
    assert 'id = "wifi_analyzer"' in actions
    assert 'id = "signal_evidence"' in actions
    assert 'id = "agent_observation"' in actions
    assert 'id = "soc_compatibility"' in actions
    assert 'id = "backend_risk"' in actions
    assert 'id = "inference_compatibility"' in actions
    assert 'id = "wifi_occupancy"' in actions
    assert 'id = "bluetooth_analyzer"' in actions
    assert 'id = "bluetooth_history"' in actions
    assert 'id = "sensor_analyzer"' in actions
    assert 'id = "motion_history"' in actions


def test_expanded_activity_rows_show_every_agent_visible_diagnostic_card():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")

    assert "COLLAPSED_ACTIVITY_DIAGNOSTIC_CARD_LIMIT = 3" in diagnostic_cards
    assert "if (expanded) return cards" in diagnostic_cards
    assert "diagnosticCardPreviewPriority" in diagnostic_cards
    assert ".sortedWith(" in diagnostic_cards
    assert '"wifi_channel_graph"' in diagnostic_cards
    assert '"bluetooth_rssi"' in diagnostic_cards
    assert '"radio_signal_graph"' in diagnostic_cards
    assert '"motion_sensor_history"' in diagnostic_cards
    assert '"soc_backend_matrix"' in diagnostic_cards
    assert "hiddenDiagnosticCardCountForActivityPreview(" in diagnostic_cards
    assert "diagnosticCards.take(3)" not in chat_screen
    assert "visibleDiagnosticCards.forEach" in chat_screen
    assert '"+$hiddenDiagnosticCardCount more cards"' in chat_screen
