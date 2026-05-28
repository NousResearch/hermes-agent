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
    chat_command_router = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatCommandRouter.kt").read_text(encoding="utf-8")
    assert 'strings.chatCommandHelp()' in chat_command_router
    assert 'Available app commands:' in strings


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
    assert 'endpoint.streamingStatus(attachments.isNotEmpty())' in chat_view_model
    assert 'fun ChatEndpoint.failureMessage(message: String)' in chat_view_model
    assert 'Hermes normalizes raw hosts, /v1 URLs, and /v1/chat/completions URLs' in chat_view_model
    assert 'streaming endpoints must stay open until [DONE]' in chat_view_model
    assert 'buildChatRequestMessages(' in chat_view_model
    assert 'User-configured agent persona/system instructions' in chat_view_model
    assert 'isEndpointStatusText(displayText)' in (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    assert 'Speaking the latest Hermes reply' not in chat_view_model  # UI handles TTS feedback


def test_sse_client_surfaces_custom_endpoint_disconnects_and_uses_longer_stream_timeout():
    sse_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/api/HermesSseClient.kt").read_text(encoding="utf-8")
    sse_test = (REPO_ROOT / "android/app/src/test/java/com/mobilefork/hermesagent/api/HermesSseClientTest.kt").read_text(encoding="utf-8")

    assert 'DEFAULT_HTTP_CLIENT' in sse_client
    assert '.readTimeout(120, TimeUnit.SECONDS)' in sse_client
    assert '.header("Accept", "text/event-stream")' in sse_client
    assert 'Custom endpoint stream closed before the endpoint sent [DONE]' in sse_client
    assert 'endpointTransportErrorMessage(error)' in sse_client
    assert 'sseDataPayload(line)' in sse_client
    assert 'line.startsWith("data:")' in sse_client
    assert 'sawFinishReason' in sse_client
    assert 'streamChatCompletion_reports_endpoint_hint_when_sse_stream_closes_before_done' in sse_test
    assert 'streamChatCompletion_accepts_finishReasonAsCompletionWhenDoneFrameIsMissing' in sse_test
    assert 'streamChatCompletion_accepts_dataFramesWithoutSpaceAndKeepAliveLines' in sse_test



def test_empty_chat_layout_scrolls_welcome_state_on_small_or_large_font_screens():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")

    assert 'LazyColumn(' in chat_screen
    assert 'EmptyChatHint(' in chat_screen
    assert 'contentPadding = PaddingValues(top = 24.dp, bottom = 8.dp)' in chat_screen


def test_signal_intelligence_quick_actions_launch_direct_diagnostic_cards():
    actions = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/SignalIntelligenceQuickActions.kt").read_text(encoding="utf-8")
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    chat_view_model = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatViewModel.kt").read_text(encoding="utf-8")
    chat_client = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/NativeToolCallingChatClient.kt").read_text(encoding="utf-8")

    for action in [
        "signal_awareness_report",
        "agent_signal_briefing_report",
        "agent_signal_card_deck_report",
        "agent_signal_card_refresh_plan_report",
        "agent_signal_card_refresh_status_report",
        "agent_signal_session_snapshot_report",
        "agent_signal_proof_audit_report",
        "agent_signal_replay_export_report",
        "agent_signal_replay_freshness_audit_report",
        "agent_signal_timeline_report",
        "agent_signal_evidence_report",
        "agent_signal_workflow_handoff_report",
        "agent_signal_permission_runbook_report",
        "agent_observation_report",
        "agent_card_priority_report",
        "agent_environment_report",
        "agent_self_check_report",
        "mcp_tool_server_registry_report",
        "agent_objective_coverage_report",
        "agent_release_validation_report",
        "agent_capability_upgrade_report",
        "soc_compatibility_report",
        "mediatek_readiness_report",
        "mediatek_signal_stack_report",
        "accelerator_preflight_report",
        "non_adreno_backend_advisor_report",
        "mediatek_backend_launch_checklist_report",
        "gpu_backend_risk_report",
        "local_inference_compatibility_report",
        "wifi_analyzer_report",
        "wifi_signal_advisor_report",
        "wifi_channel_decision_packet_report",
        "wifi_connection_link",
        "wifi_scan",
        "wifi_channel_utilization",
        "bluetooth_analyzer_report",
        "bluetooth_signal_advisor_report",
        "bluetooth_nearby_decision_packet_report",
        "bluetooth_signal_history",
        "bluetooth_device_details",
        "sensor_analyzer_report",
        "sensor_workflow_advisor_report",
        "motion_sensor_decision_packet_report",
        "motion_sensor_quality",
        "motion_sensor_history",
        "radio_signal_graph",
        "radio_signal_advisor_report",
        "radio_signal_decision_packet_report",
        "sdr_bridge_samples",
        "rf_coexistence_report",
    ]:
        if action != "sdr_bridge_samples":
            assert f"action={action}" in actions
        assert f'"{action}"' in chat_client

    assert "sendQuickPrompt" in chat_screen
    assert "fun sendQuickPrompt" in chat_view_model
    assert "extractExplicitAndroidDiagnosticsArguments(userText)" in chat_client
    assert 'testTag("HermesSignalQuickAction_${action.id}")' in chat_screen
    assert 'id = "wifi_analyzer"' in actions
    assert 'id = "wifi_channel_decision"' in actions
    assert 'id = "signal_briefing"' in actions
    assert 'id = "signal_card_deck"' in actions
    assert 'id = "card_refresh_plan"' in actions
    assert 'id = "card_refresh_status"' in actions
    assert 'id = "signal_session_snapshot"' in actions
    assert 'id = "signal_proof_audit"' in actions
    assert 'id = "signal_replay_export"' in actions
    assert 'id = "signal_replay_freshness"' in actions
    assert 'id = "signal_timeline"' in actions
    assert 'id = "signal_evidence"' in actions
    assert 'id = "workflow_handoff"' in actions
    assert 'id = "permission_runbook"' in actions
    assert 'id = "rf_coexistence"' in actions
    assert 'id = "agent_self_check"' in actions
    assert 'id = "agent_observation"' in actions
    assert 'id = "top_cards"' in actions
    assert 'id = "mcp_registry"' in actions
    assert 'id = "upgrade_audit"' in actions
    assert 'id = "soc_compatibility"' in actions
    assert 'id = "mediatek_readiness"' in actions
    assert 'id = "mediatek_signal_stack"' in actions
    assert 'id = "accelerator_preflight"' in actions
    assert 'id = "non_adreno_backend_advisor"' in actions
    assert 'id = "mediatek_launch_checklist"' in actions
    assert 'id = "backend_risk"' in actions
    assert 'id = "inference_compatibility"' in actions
    assert 'id = "wifi_advisor"' in actions
    assert 'id = "wifi_link"' in actions
    assert 'id = "wifi_occupancy"' in actions
    assert 'id = "bluetooth_analyzer"' in actions
    assert 'id = "bluetooth_advisor"' in actions
    assert 'id = "bluetooth_decision"' in actions
    assert 'id = "bluetooth_history"' in actions
    assert 'id = "bluetooth_details"' in actions
    assert 'id = "sensor_analyzer"' in actions
    assert 'id = "sensor_advisor"' in actions
    assert 'id = "motion_decision"' in actions
    assert 'id = "motion_quality"' in actions
    assert 'id = "motion_history"' in actions
    assert 'id = "radio_advisor"' in actions
    assert 'id = "radio_decision"' in actions


def test_expanded_activity_rows_show_every_agent_visible_diagnostic_card():
    chat_screen = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/ChatScreen.kt").read_text(encoding="utf-8")
    diagnostic_cards = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/chat/DiagnosticCards.kt").read_text(encoding="utf-8")

    assert "COLLAPSED_ACTIVITY_DIAGNOSTIC_CARD_LIMIT = 3" in diagnostic_cards
    assert "if (expanded) return cards" in diagnostic_cards
    assert "diagnosticCardPreviewPriority" in diagnostic_cards
    assert ".sortedWith(" in diagnostic_cards
    assert '"wifi_channel_graph"' in diagnostic_cards
    assert '"bluetooth_rssi"' in diagnostic_cards
    assert '"bluetooth_device_detail"' in diagnostic_cards
    assert '"bluetooth_nearby_decision_packet"' in diagnostic_cards
    assert '"bluetooth_nearby_decision_routes"' in diagnostic_cards
    assert '"bluetooth_nearby_claim_boundaries"' in diagnostic_cards
    assert '"bluetooth_signal_advisor_matrix"' in diagnostic_cards
    assert '"bluetooth_device_candidates"' in diagnostic_cards
    assert '"radio_signal_graph"' in diagnostic_cards
    assert '"radio_signal_decision_packet"' in diagnostic_cards
    assert '"radio_signal_decision_routes"' in diagnostic_cards
    assert '"radio_signal_claim_boundaries"' in diagnostic_cards
    assert '"radio_bridge_sample_metadata"' in diagnostic_cards
    assert '"radio_receiver_bridge_schema"' in diagnostic_cards
    assert '"agent_card_priority_matrix"' in diagnostic_cards
    assert '"agent_signal_briefing_matrix"' in diagnostic_cards
    assert '"agent_signal_timeline"' in diagnostic_cards
    assert '"agent_signal_refresh_routes"' in diagnostic_cards
    assert '"agent_signal_workflow_handoff_matrix"' in diagnostic_cards
    assert '"agent_signal_next_action_routes"' in diagnostic_cards
    assert '"agent_signal_permission_runbook_matrix"' in diagnostic_cards
    assert '"agent_signal_active_refresh_routes"' in diagnostic_cards
    assert '"agent_signal_card_deck_manifest"' in diagnostic_cards
    assert '"agent_signal_card_refresh_plan_matrix"' in diagnostic_cards
    assert '"agent_signal_card_refresh_status_matrix"' in diagnostic_cards
    assert '"agent_signal_session_snapshot_matrix"' in diagnostic_cards
    assert '"agent_signal_session_domain_matrix"' in diagnostic_cards
    assert '"agent_signal_session_action_routes"' in diagnostic_cards
    assert '"agent_signal_proof_audit_matrix"' in diagnostic_cards
    assert '"agent_signal_claim_boundary_matrix"' in diagnostic_cards
    assert '"agent_signal_replay_export_manifest"' in diagnostic_cards
    assert '"agent_signal_replay_frame_index"' in diagnostic_cards
    assert '"agent_signal_replay_metadata_keys"' in diagnostic_cards
    assert '"agent_signal_replay_freshness_matrix"' in diagnostic_cards
    assert '"agent_signal_replay_refresh_routes"' in diagnostic_cards
    assert '"agent_signal_replay_staleness_summary"' in diagnostic_cards
    assert '"agent_top_card_slots"' in diagnostic_cards
    assert '"agent_signal_metadata_keys"' in diagnostic_cards
    assert '"agent_card_open_sequence"' in diagnostic_cards
    assert '"kai_interactive_screen_parity"' in diagnostic_cards
    assert '"mcp_tool_server_registry"' in diagnostic_cards
    assert '"mcp_tool_server_routes"' in diagnostic_cards
    assert '"agent_upgrade_objective_matrix"' in diagnostic_cards
    assert '"agent_upgrade_route_matrix"' in diagnostic_cards
    assert '"motion_sensor_history"' in diagnostic_cards
    assert '"motion_sensor_decision_packet"' in diagnostic_cards
    assert '"motion_sensor_decision_routes"' in diagnostic_cards
    assert '"motion_sensor_claim_boundaries"' in diagnostic_cards
    assert '"motion_sensor_quality"' in diagnostic_cards
    assert '"sensor_workflow_advisor_matrix"' in diagnostic_cards
    assert '"sensor_workflow_candidates"' in diagnostic_cards
    assert '"radio_signal_advisor_matrix"' in diagnostic_cards
    assert '"radio_receiver_candidates"' in diagnostic_cards
    assert '"soc_backend_matrix"' in diagnostic_cards
    assert '"accelerator_preflight_matrix"' in diagnostic_cards
    assert '"non_adreno_backend_advisor_matrix"' in diagnostic_cards
    assert '"mediatek_backend_launch_checklist_matrix"' in diagnostic_cards
    assert '"mediatek_signal_stack_matrix"' in diagnostic_cards
    assert '"mediatek_signal_refresh_routes"' in diagnostic_cards
    assert '"mediatek_signal_claim_boundaries"' in diagnostic_cards
    assert "hiddenDiagnosticCardCountForActivityPreview(" in diagnostic_cards
    assert "diagnosticCards.take(3)" not in chat_screen
    assert "visibleDiagnosticCards.forEach" in chat_screen
    assert "strings.moreCards(hiddenDiagnosticCardCount)" in chat_screen
