package com.mobilefork.hermesagent.ui.chat

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.InterruptedIOException

class NativeToolCallingChatClientTest {
    @Test
    fun systemPromptIncludesBoundedCustomPersonaBeforePromotedMemory() {
        val content = NativeToolCallingChatClient.buildSystemPromptContent(
            toolsEnabled = true,
            customSystemPrompt = "Stay concise and use Wi-Fi analyzer cards when signal context matters.",
            promotedMemoryContext = "User prefers local models.",
        )

        assertTrue(content.contains("Kai-style custom agent persona/system prompt"))
        assertTrue(content.contains("schedule_task/list_tasks/cancel_task"))
        assertTrue(content.contains("not unrestricted background AI prompt execution"))
        assertTrue(content.contains("action=agent_signal_evidence_report"))
        assertTrue(content.contains("action=agent_signal_replay_export_report"))
        assertTrue(content.contains("action=agent_signal_replay_freshness_audit_report"))
        assertTrue(content.contains("action=agent_signal_observation_packet_report"))
        assertTrue(content.contains("action=agent_signal_proof_audit_report"))
        assertTrue(content.contains("action=agent_signal_workflow_handoff_report"))
        assertTrue(content.contains("action=agent_signal_permission_runbook_report"))
        assertTrue(content.contains("action=mediatek_signal_stack_report"))
        assertTrue(content.contains("action=device_validation_evidence_export_report"))
        assertTrue(content.contains("action=wifi_channel_decision_packet_report"))
        assertTrue(content.contains("action=bluetooth_nearby_decision_packet_report"))
        assertTrue(content.contains("action=motion_sensor_decision_packet_report"))
        assertTrue(content.contains("action=radio_signal_decision_packet_report"))
        assertTrue(content.contains("signal proof audits"))
        assertTrue(content.contains("signal replay/export bundles"))
        assertTrue(content.contains("device-validation evidence export bundles"))
        assertTrue(content.contains("replay freshness/staleness audits"))
        assertTrue(content.contains("compact signal observation packets"))
        assertTrue(content.contains("visual slots, graph routes"))
        assertTrue(content.contains("signal workflow handoff and next-action reports"))
        assertTrue(content.contains("signal permission and active-refresh runbooks"))
        assertTrue(content.contains("MediaTek signal-stack reports"))
        assertTrue(content.contains("MCP tool-server registry reports"))
        assertTrue(content.contains("full upgrade objective audit reports"))
        assertTrue(content.contains("what Hermes/Gemma can see from nearby signals"))
        assertTrue(content.contains("User-configured agent persona"))
        assertTrue(content.contains("Stay concise and use Wi-Fi analyzer cards"))
        assertTrue(content.contains("Promoted local memory context"))
        assertTrue(content.indexOf("User-configured agent persona") < content.indexOf("Promoted local memory context"))
    }

    @Test
    fun systemPromptCompressesLongCustomPersonaAndMemoryContext() {
        val content = NativeToolCallingChatClient.buildSystemPromptContent(
            toolsEnabled = false,
            customSystemPrompt = "persona ".repeat(400),
            promotedMemoryContext = "memory ".repeat(500),
        )

        assertTrue(content.contains("User-configured agent persona"))
        assertTrue(content.contains("Promoted local memory context"))
        assertTrue(content.contains("hermes context compressed"))
        assertFalse(content.contains("persona ".repeat(250)))
        assertFalse(content.contains("memory ".repeat(300)))
    }

    @Test
    fun contextRecoveryCompactsInitialSystemAndUserMessages() {
        val messages = JSONArray()
            .put(JSONObject().put("role", "system").put("content", "system ".repeat(1_500)))
            .put(JSONObject().put("role", "user").put("content", "user ".repeat(1_800)))

        val recovered = NativeToolContextCompressor.recoverMessagesAfterContextOverflow(messages)

        assertEquals(2, recovered.length())
        assertTrue(recovered.toString().length < messages.toString().length)
        assertTrue(recovered.getJSONObject(0).getString("content").contains("hermes context compressed"))
        assertTrue(recovered.getJSONObject(1).getString("content").contains("hermes context compressed"))
    }

    @Test
    fun contextRecoveryCompactsToolSchemaDescriptions() {
        val toolSpecs = JSONArray()
            .put(
                JSONObject()
                    .put("type", "function")
                    .put(
                        "function",
                        JSONObject()
                            .put("name", "android_device_diagnostics_tool")
                            .put("description", "diagnostics ".repeat(200))
                            .put(
                                "parameters",
                                JSONObject()
                                    .put("type", "object")
                                    .put(
                                        "properties",
                                        JSONObject()
                                            .put(
                                                "action",
                                                JSONObject()
                                                    .put("type", "string")
                                                    .put("description", "status wifi bluetooth sensors ".repeat(200)),
                                            ),
                                    ),
                            ),
                    ),
            )

        val recovered = NativeToolContextCompressor.recoverToolSpecsAfterContextOverflow(toolSpecs)

        requireNotNull(recovered)
        val function = recovered.getJSONObject(0).getJSONObject("function")
        val action = function.getJSONObject("parameters").getJSONObject("properties").getJSONObject("action")
        assertTrue(function.getString("description").length < 320)
        assertTrue(action.getString("description").length < 320)
        assertTrue(recovered.toString().length < toolSpecs.toString().length)
    }

    @Test
    fun contextWindowErrorsAreRecognizedForRecovery() {
        assertTrue(
            NativeToolCallingChatClient.isContextWindowError(
                IllegalStateException("exceed_context_size: prompt exceeds the available context size"),
            ),
        )
        assertTrue(
            NativeToolCallingChatClient.isContextWindowError(
                IllegalStateException("The local model ran out of context."),
            ),
        )
        assertFalse(
            NativeToolCallingChatClient.isContextWindowError(
                IllegalStateException("HTTP 500 backend unavailable"),
            ),
        )
    }

    @Test
    fun skipsLocalFollowUpAfterExternalActivityHandoff() {
        val result = JSONObject()
            .put("success", true)
            .put("action", "open_uri")
            .put("external_activity_handoff", true)
            .put("message", "Started Android intent")

        assertTrue(NativeToolCallingChatClient.shouldSkipNativeFollowUpAfterToolResult(result.toString()))
    }

    @Test
    fun continuesLocalFollowUpAfterOrdinaryToolResult() {
        val result = JSONObject()
            .put("success", true)
            .put("path", "hermes-output.txt")

        assertFalse(NativeToolCallingChatClient.shouldSkipNativeFollowUpAfterToolResult(result.toString()))
    }

    @Test
    fun recoversHtmlGenerationTimeoutsWithFallbackDocument() {
        assertTrue(
            NativeToolCallingChatClient.isRecoverableHtmlGenerationFailure(
                IllegalStateException("LiteRT-LM generation timed out after 45 seconds before producing a response"),
            )
        )
        assertTrue(
            NativeToolCallingChatClient.isRecoverableHtmlGenerationFailure(
                InterruptedIOException("timeout"),
            )
        )
    }

    @Test
    fun doesNotHideNonTimeoutHtmlGenerationErrors() {
        assertFalse(
            NativeToolCallingChatClient.isRecoverableHtmlGenerationFailure(
                IllegalStateException("image input is unavailable"),
            )
        )
    }

    @Test
    fun extractsExplicitAndroidDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=wifi_analyzer_report refresh=false max_results=12",
        )

        requireNotNull(parsed)
        assertEquals("wifi_analyzer_report", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
        assertEquals(12, parsed.getInt("max_results"))
    }

    @Test
    fun extractsExplicitWifiConnectionLinkDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=wifi_connection_link",
        )

        requireNotNull(parsed)
        assertEquals("wifi_connection_link", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitWifiSignalAdvisorDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=wifi_signal_advisor_report refresh=false",
        )

        requireNotNull(parsed)
        assertEquals("wifi_signal_advisor_report", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
    }

    @Test
    fun extractsExplicitWifiChannelDecisionPacketDiagnosticQuickActionArguments() {
        val canonical = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=wifi_channel_decision_packet_report refresh=false",
        )
        val alias = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=mediatek_wifi_channel_decision refresh=false",
        )

        requireNotNull(canonical)
        requireNotNull(alias)
        assertEquals("wifi_channel_decision_packet_report", canonical.getString("action"))
        assertFalse(canonical.getBoolean("refresh"))
        assertEquals("mediatek_wifi_channel_decision", alias.getString("action"))
        assertFalse(alias.getBoolean("refresh"))
    }

    @Test
    fun extractsExplicitBluetoothNearbyDecisionPacketDiagnosticQuickActionArguments() {
        val canonical = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=bluetooth_nearby_decision_packet_report refresh=false",
        )
        val alias = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=mediatek_bluetooth_decision_packet refresh=false",
        )

        requireNotNull(canonical)
        requireNotNull(alias)
        assertEquals("bluetooth_nearby_decision_packet_report", canonical.getString("action"))
        assertFalse(canonical.getBoolean("refresh"))
        assertEquals("mediatek_bluetooth_decision_packet", alias.getString("action"))
        assertFalse(alias.getBoolean("refresh"))
    }

    @Test
    fun extractsExplicitMotionSensorDecisionPacketDiagnosticQuickActionArguments() {
        val canonical = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=motion_sensor_decision_packet_report include_snapshot=false",
        )
        val alias = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=mediatek_motion_decision_packet include_snapshot=false",
        )

        requireNotNull(canonical)
        requireNotNull(alias)
        assertEquals("motion_sensor_decision_packet_report", canonical.getString("action"))
        assertFalse(canonical.getBoolean("include_snapshot"))
        assertEquals("mediatek_motion_decision_packet", alias.getString("action"))
        assertFalse(alias.getBoolean("include_snapshot"))
    }

    @Test
    fun extractsExplicitSocCompatibilityDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=soc_compatibility_report",
        )

        requireNotNull(parsed)
        assertEquals("soc_compatibility_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitMediatekReadinessDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=mediatek_readiness_report",
        )

        requireNotNull(parsed)
        assertEquals("mediatek_readiness_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitAcceleratorPreflightDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=accelerator_preflight_report",
        )

        requireNotNull(parsed)
        assertEquals("accelerator_preflight_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitNonAdrenoBackendAdvisorDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=non_adreno_backend_advisor_report",
        )

        requireNotNull(parsed)
        assertEquals("non_adreno_backend_advisor_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitGpuBackendRiskDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=gpu_backend_risk_report",
        )

        requireNotNull(parsed)
        assertEquals("gpu_backend_risk_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitLocalInferenceCompatibilityDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=local_inference_compatibility_report",
        )

        requireNotNull(parsed)
        assertEquals("local_inference_compatibility_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitSignalEvidenceDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_signal_evidence_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_signal_evidence_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitSignalReplayExportDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_signal_replay_export_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_signal_replay_export_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitSignalReplayFreshnessDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_signal_replay_freshness_audit_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_signal_replay_freshness_audit_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitSignalSessionSnapshotDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_signal_session_snapshot_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_signal_session_snapshot_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitSignalProofAuditDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_signal_proof_audit_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_signal_proof_audit_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitSignalPermissionRunbookDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_signal_permission_runbook_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_signal_permission_runbook_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitSignalTimelineDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_signal_timeline_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_signal_timeline_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitUpgradeCoverageAndReleaseReadinessDiagnosticAliases() {
        val coverage = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_upgrade_coverage_report",
        )
        val release = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=github_release_readiness_report",
        )

        requireNotNull(coverage)
        requireNotNull(release)
        assertEquals("agent_upgrade_coverage_report", coverage.getString("action"))
        assertEquals("github_release_readiness_report", release.getString("action"))
    }

    @Test
    fun extractsExplicitMediatekDeviceValidationDiagnosticAliases() {
        val canonical = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=mediatek_device_validation_report",
        )
        val phoneAlias = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=phone_signal_validation_report",
        )

        requireNotNull(canonical)
        requireNotNull(phoneAlias)
        assertEquals("mediatek_device_validation_report", canonical.getString("action"))
        assertEquals("phone_signal_validation_report", phoneAlias.getString("action"))
    }

    @Test
    fun extractsExplicitDeviceValidationEvidenceExportDiagnosticAliases() {
        val canonical = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=device_validation_evidence_export_report",
        )
        val phoneAlias = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=phone_validation_evidence_export",
        )
        val fdroidAlias = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=fdroid_device_evidence_export",
        )

        requireNotNull(canonical)
        requireNotNull(phoneAlias)
        requireNotNull(fdroidAlias)
        assertEquals("device_validation_evidence_export_report", canonical.getString("action"))
        assertEquals("phone_validation_evidence_export", phoneAlias.getString("action"))
        assertEquals("fdroid_device_evidence_export", fdroidAlias.getString("action"))
    }

    @Test
    fun extractsExplicitSignalObservationPacketDiagnosticAliases() {
        val canonical = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_signal_observation_packet_report",
        )
        val gemmaAlias = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=gemma_signal_observation_packet",
        )
        val contextAlias = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=signal_context_packet_report",
        )

        requireNotNull(canonical)
        requireNotNull(gemmaAlias)
        requireNotNull(contextAlias)
        assertEquals("agent_signal_observation_packet_report", canonical.getString("action"))
        assertEquals("gemma_signal_observation_packet", gemmaAlias.getString("action"))
        assertEquals("signal_context_packet_report", contextAlias.getString("action"))
    }

    @Test
    fun extractsExplicitMcpRegistryDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=mcp_tool_server_registry_report",
        )

        requireNotNull(parsed)
        assertEquals("mcp_tool_server_registry_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitUpgradeAuditDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_capability_upgrade_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_capability_upgrade_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitSdrBridgeSampleDiagnosticArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=sdr_bridge_samples sdr_samples_json=noaa_sample span_hz=200000 sample_rate_hz=240000 receiver_id=external_sdr_bridge",
        )

        requireNotNull(parsed)
        assertEquals("sdr_bridge_samples", parsed.getString("action"))
        assertEquals("noaa_sample", parsed.getString("sdr_samples_json"))
        assertEquals("200000", parsed.getString("span_hz"))
        assertEquals("240000", parsed.getString("sample_rate_hz"))
        assertEquals("external_sdr_bridge", parsed.getString("receiver_id"))
    }

    @Test
    fun extractsImplicitSignalEvidenceForNearbySignalQuestionsOnly() {
        val parsed = NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments(
            "What can Hermes see from nearby Wi-Fi, Bluetooth, and radio signals right now?",
        )

        requireNotNull(parsed)
        assertEquals("agent_signal_evidence_report", parsed.getString("action"))
        assertEquals(
            "agent_signal_evidence_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Show the current signal evidence bundle.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_timeline_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Show the agent signal timeline for what Gemma recently saw.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_replay_export_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Export a portable signal replay bundle for Gemma.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_replay_freshness_audit_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Show the replay freshness audit before treating the exported signal rows as current.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_observation_packet_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Build a Gemma signal observation packet for the top card signal context.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_card_deck_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Show the expanded signal card deck for Gemma.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_card_refresh_plan_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Show the signal card refresh plan before refreshing an expanded top card.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_card_refresh_status_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Which signal cards can refresh right now?")?.getString("action"),
        )
        assertEquals(
            "agent_signal_proof_audit_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Show the current signal proof audit before claiming live evidence.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_session_snapshot_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Show the current signal session snapshot.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_workflow_handoff_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Show the signal workflow handoff for the next evidence route.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_permission_runbook_report",
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("Show the signal permission runbook before a live scan.")?.getString("action"),
        )
        assertNull(
            NativeToolCallingChatClient.extractImplicitSignalEvidenceArguments("What can you see on the screen?"),
        )
    }

    @Test
    fun extractsImplicitDomainDiagnosticsForSignalHardwareQuestions() {
        assertEquals(
            "wifi_signal_advisor_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show Wi-Fi advisor recommendations and roaming candidates.")?.getString("action"),
        )
        assertEquals(
            "mcp_tool_server_registry_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the Kai MCP server registry and Context7 parity.")?.getString("action"),
        )
        assertEquals(
            "agent_capability_upgrade_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the full upgrade objective audit and what remains incomplete.")?.getString("action"),
        )
        assertEquals(
            "agent_objective_coverage_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the objective coverage gaps and Kai Wi-Fi Analyzer parity map.")?.getString("action"),
        )
        assertEquals(
            "agent_objective_coverage_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the Hermes upgrade coverage report for the full objective.")?.getString("action"),
        )
        assertEquals(
            "agent_release_validation_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show release validation for GitHub release artifacts and F-Droid metadata.")?.getString("action"),
        )
        assertEquals(
            "agent_release_validation_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show GitHub release readiness before I publish the Android APK.")?.getString("action"),
        )
        assertEquals(
            "device_validation_evidence_export_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Build a release proof package for physical phone evidence.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_observation_packet_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Build a nearby signal context packet showing what Gemma can see from Wi-Fi, Bluetooth, sensors, and radio.")?.getString("action"),
        )
        assertEquals(
            "mediatek_device_validation_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show physical MediaTek validation and device proof gates for the phone.")?.getString("action"),
        )
        assertEquals(
            "mediatek_device_validation_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show phone signal validation before claiming live signal proof.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_card_deck_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the top signal card deck with Wi-Fi, Bluetooth, radio, sensors, backend, and release cards.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_card_refresh_plan_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the per-card refresh plan for the expanded signal cards.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_card_refresh_status_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show top card refresh status and which signal cards can refresh.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_proof_audit_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the signal proof audit so Gemma knows what live evidence can be claimed.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_session_snapshot_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the current signal session snapshot before picking a top card.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_replay_export_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the signal replay export for nearby RF evidence.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_replay_freshness_audit_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show signal replay staleness and freshness for the export bundle.")?.getString("action"),
        )
        assertEquals(
            "wifi_channel_decision_packet_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Which Wi-Fi channel should my router use with MediaTek RF coexistence?")?.getString("action"),
        )
        assertEquals(
            "bluetooth_nearby_decision_packet_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the nearby Bluetooth decision packet with MediaTek RF coexistence context.")?.getString("action"),
        )
        assertEquals(
            "wifi_channel_rating",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Rate the best Wi-Fi channel for the nearby APs.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_timeline_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the recent signal timeline for what the agent recently saw.")?.getString("action"),
        )
        assertEquals(
            "agent_signal_workflow_handoff_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("What should Gemma check next for nearby signals?")?.getString("action"),
        )
        assertEquals(
            "agent_signal_permission_runbook_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the active signal refresh routes before requesting permissions.")?.getString("action"),
        )
        assertEquals(
            "wifi_connection_link",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show my current Wi-Fi connection link quality.")?.getString("action"),
        )
        assertEquals(
            "bluetooth_scan",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show nearby Bluetooth devices and BLE beacons.")?.getString("action"),
        )
        assertEquals(
            "bluetooth_signal_advisor_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the Bluetooth advisor recommendation for nearby devices.")?.getString("action"),
        )
        assertEquals(
            "bluetooth_device_details",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show Bluetooth device details for nearby BLE devices.")?.getString("action"),
        )
        assertEquals(
            "bluetooth_export",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Export Bluetooth device metadata as CSV.")?.getString("action"),
        )
        assertEquals(
            "motion_sensor_decision_packet_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the IMU decision packet with accelerometer and gyroscope claim boundaries.")?.getString("action"),
        )
        assertEquals(
            "motion_sensor_history",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show accelerometer history and gyroscope trends.")?.getString("action"),
        )
        assertEquals(
            "motion_sensor_quality",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Check IMU quality and sensor calibration before orientation automation.")?.getString("action"),
        )
        assertEquals(
            "sensor_workflow_advisor_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the sensor workflow advisor for accelerometer and gyroscope tasks.")?.getString("action"),
        )
        assertEquals(
            "radio_signal_decision_packet_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the radio decision packet for AM/FM and SDR bridge evidence with MediaTek context.")?.getString("action"),
        )
        assertEquals(
            "radio_signal_advisor_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the radio advisor recommendation for AM/FM and SDR receiver choices.")?.getString("action"),
        )
        assertEquals(
            "radio_signal_graph",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the AM/FM radio graph.")?.getString("action"),
        )
        assertEquals(
            "radio_signal_graph",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show external SDR bridge sample readiness.")?.getString("action"),
        )
        assertEquals(
            "mediatek_signal_stack_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show MediaTek signal stack for Wi-Fi Bluetooth radio sensors.")?.getString("action"),
        )
        assertEquals(
            "mediatek_signal_stack_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show non-Adreno signal stack claim boundaries.")?.getString("action"),
        )
        assertEquals(
            "mediatek_readiness_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show MediaTek Dimensity readiness for Mali GPU fallback.")?.getString("action"),
        )
        assertEquals(
            "accelerator_preflight_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Run an OpenCL preflight before starting the MediaTek GPU delegate.")?.getString("action"),
        )
        assertEquals(
            "mediatek_backend_launch_checklist_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Before starting local inference on a MediaTek Mali phone, show the non-Adreno backend advisor launch plan.")?.getString("action"),
        )
        assertEquals(
            "mediatek_backend_launch_checklist_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Show the MediaTek launch checklist before starting local inference.")?.getString("action"),
        )
        assertEquals(
            "soc_compatibility_report",
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("Will this MediaTek Dimensity phone work without Snapdragon assumptions?")?.getString("action"),
        )
        assertNull(
            NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments("What can you see on the screen?"),
        )
    }

    @Test
    fun extractsImplicitBluetoothAnalyzerFiltersAndPauseResumeArguments() {
        val parsed = NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments(
            "Show nearby Bluetooth devices service Heart Rate with manufacturer Apple and near proximity using a fresh scan.",
        )

        requireNotNull(parsed)
        assertEquals("bluetooth_scan", parsed.getString("action"))
        assertTrue(parsed.getBoolean("refresh"))
        assertEquals("resumed", parsed.getString("scan_mode"))
        assertEquals("Heart Rate", parsed.getString("filter_bluetooth_service"))
        assertEquals("Apple", parsed.getString("filter_bluetooth_manufacturer"))
        assertEquals("near", parsed.getString("filter_bluetooth_proximity"))

        val export = NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments(
            "Export Bluetooth device details service Heart Rate with manufacturer Apple as CSV while paused.",
        )

        requireNotNull(export)
        assertEquals("bluetooth_export", export.getString("action"))
        assertFalse(export.getBoolean("refresh"))
        assertEquals("paused", export.getString("scan_mode"))
        assertEquals("Heart Rate", export.getString("filter_bluetooth_service"))
        assertEquals("Apple", export.getString("filter_bluetooth_manufacturer"))
        assertEquals("csv", export.getString("export_format"))
    }

    @Test
    fun extractsImplicitWifiAnalyzerFiltersExportAndPauseResumeArguments() {
        val export = NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments(
            "Export WPA3 5GHz Wi-Fi access point details as CSV while paused.",
        )

        requireNotNull(export)
        assertEquals("wifi_export", export.getString("action"))
        assertFalse(export.getBoolean("refresh"))
        assertEquals("paused", export.getString("scan_mode"))
        assertEquals("5GHz", export.getString("filter_band"))
        assertEquals("WPA3", export.getString("filter_security"))
        assertEquals("csv", export.getString("export_format"))

        val hidden = NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments(
            "Show only hidden weak Wi-Fi networks from cached scan results.",
        )

        requireNotNull(hidden)
        assertEquals("wifi_scan", hidden.getString("action"))
        assertEquals("weak", hidden.getString("filter_signal"))
        assertTrue(hidden.getBoolean("hidden_only"))
        assertEquals("paused", hidden.getString("scan_mode"))

        val fresh = NativeToolCallingChatClient.extractImplicitAndroidDiagnosticsArguments(
            "Show complete Wi-Fi AP details for SSID HermesLab with a fresh scan.",
        )

        requireNotNull(fresh)
        assertEquals("wifi_ap_details", fresh.getString("action"))
        assertTrue(fresh.getBoolean("refresh"))
        assertEquals("resumed", fresh.getString("scan_mode"))
        assertEquals("HermesLab", fresh.getString("filter_ssid"))
    }

    @Test
    fun extractsExplicitAgentCardManifestDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_card_manifest_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_card_manifest_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitAgentCardPriorityDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_card_priority_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_card_priority_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitSignalWorkflowHandoffDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=agent_signal_workflow_handoff_report",
        )

        requireNotNull(parsed)
        assertEquals("agent_signal_workflow_handoff_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitLocalBackendRuntimeDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=local_backend_runtime_report",
        )

        requireNotNull(parsed)
        assertEquals("local_backend_runtime_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitWifiChannelUtilizationDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=wifi_channel_utilization refresh=false scan_mode=paused",
        )

        requireNotNull(parsed)
        assertEquals("wifi_channel_utilization", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
        assertEquals("paused", parsed.getString("scan_mode"))
    }

    @Test
    fun extractsExplicitWifiChannelGraphDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=wifi_channel_graph refresh=false scan_mode=paused",
        )

        requireNotNull(parsed)
        assertEquals("wifi_channel_graph", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
        assertEquals("paused", parsed.getString("scan_mode"))
    }

    @Test
    fun extractsExplicitBluetoothAdvisorDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=bluetooth_signal_advisor_report refresh=false scan_mode=paused",
        )

        requireNotNull(parsed)
        assertEquals("bluetooth_signal_advisor_report", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
        assertEquals("paused", parsed.getString("scan_mode"))
    }

    @Test
    fun extractsExplicitBluetoothSignalHistoryDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=bluetooth_signal_history refresh=false scan_mode=paused",
        )

        requireNotNull(parsed)
        assertEquals("bluetooth_signal_history", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
        assertEquals("paused", parsed.getString("scan_mode"))
    }

    @Test
    fun extractsExplicitBluetoothDeviceDetailsDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=bluetooth_device_details refresh=false scan_mode=paused detail_limit=6",
        )
        val export = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=bluetooth_export refresh=false export_format=both",
        )

        requireNotNull(parsed)
        assertEquals("bluetooth_device_details", parsed.getString("action"))
        assertFalse(parsed.getBoolean("refresh"))
        assertEquals("paused", parsed.getString("scan_mode"))
        assertEquals(6, parsed.getInt("detail_limit"))
        requireNotNull(export)
        assertEquals("bluetooth_export", export.getString("action"))
        assertEquals("both", export.getString("export_format"))
    }

    @Test
    fun extractsExplicitSensorWorkflowAdvisorDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=sensor_workflow_advisor_report include_snapshot=false sensor_types=accelerometer,gyroscope",
        )

        requireNotNull(parsed)
        assertEquals("sensor_workflow_advisor_report", parsed.getString("action"))
        assertFalse(parsed.getBoolean("include_snapshot"))
        assertEquals("accelerometer,gyroscope", parsed.getString("sensor_types"))
    }

    @Test
    fun extractsExplicitMotionSensorHistoryDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=motion_sensor_history sample=true sensor_types=accelerometer,gyroscope",
        )

        requireNotNull(parsed)
        assertEquals("motion_sensor_history", parsed.getString("action"))
        assertTrue(parsed.getBoolean("sample"))
        assertEquals("accelerometer,gyroscope", parsed.getString("sensor_types"))
    }

    @Test
    fun extractsExplicitMotionSensorQualityDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=motion_sensor_quality include_snapshot=false sensor_types=accelerometer,gyroscope,rotation_vector",
        )

        requireNotNull(parsed)
        assertEquals("motion_sensor_quality", parsed.getString("action"))
        assertFalse(parsed.getBoolean("include_snapshot"))
        assertEquals("accelerometer,gyroscope,rotation_vector", parsed.getString("sensor_types"))
    }

    @Test
    fun extractsExplicitMotionPoseDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=motion_pose sensor_types=accelerometer,magnetic_field,gyroscope",
        )

        requireNotNull(parsed)
        assertEquals("motion_pose", parsed.getString("action"))
        assertEquals("accelerometer,magnetic_field,gyroscope", parsed.getString("sensor_types"))
    }

    @Test
    fun extractsExplicitRadioAnalyzerDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=radio_analyzer_report",
        )

        requireNotNull(parsed)
        assertEquals("radio_analyzer_report", parsed.getString("action"))
    }

    @Test
    fun extractsExplicitRadioSignalGraphDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=radio_signal_graph sample_source=vendor_fm_bridge receiver_id=fm_vendor_or_sdr station_label=\"Hermes FM\" frequency_mhz=99.5 rssi_dbm=-58 snr_db=31 modulation=fm rds_program_service=HERMES",
        )

        requireNotNull(parsed)
        assertEquals("radio_signal_graph", parsed.getString("action"))
        assertEquals("vendor_fm_bridge", parsed.getString("sample_source"))
        assertEquals("fm_vendor_or_sdr", parsed.getString("receiver_id"))
        assertEquals("Hermes FM", parsed.getString("station_label"))
        assertEquals("99.5", parsed.getString("frequency_mhz"))
        assertEquals("-58", parsed.getString("rssi_dbm"))
        assertEquals("31", parsed.getString("snr_db"))
        assertEquals("fm", parsed.getString("modulation"))
        assertEquals("HERMES", parsed.getString("rds_program_service"))
    }

    @Test
    fun extractsExplicitRadioSignalAdvisorDiagnosticQuickActionArguments() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=radio_signal_advisor_report sample_source=vendor_fm_bridge receiver_id=fm_vendor_or_sdr",
        )

        requireNotNull(parsed)
        assertEquals("radio_signal_advisor_report", parsed.getString("action"))
        assertEquals("vendor_fm_bridge", parsed.getString("sample_source"))
        assertEquals("fm_vendor_or_sdr", parsed.getString("receiver_id"))
    }

    @Test
    fun extractsExplicitRadioSignalDecisionPacketDiagnosticQuickActionArguments() {
        val canonical = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=radio_signal_decision_packet_report sample_source=vendor_fm_bridge receiver_id=fm_vendor_or_sdr",
        )
        val alias = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=mediatek_radio_decision_packet",
        )

        requireNotNull(canonical)
        requireNotNull(alias)
        assertEquals("radio_signal_decision_packet_report", canonical.getString("action"))
        assertEquals("vendor_fm_bridge", canonical.getString("sample_source"))
        assertEquals("fm_vendor_or_sdr", canonical.getString("receiver_id"))
        assertEquals("mediatek_radio_decision_packet", alias.getString("action"))
    }

    @Test
    fun ignoresUnknownExplicitAndroidDiagnosticActions() {
        val parsed = NativeToolCallingChatClient.extractExplicitAndroidDiagnosticsArguments(
            "Run android_device_diagnostics_tool action=network_intrusion",
        )

        assertNull(parsed)
    }

    @Test
    fun compactsLargeJsonToolResultForLocalModelContext() {
        val largeOutput = "scroll-item\n".repeat(700)
        val result = JSONObject()
            .put("exit_code", 0)
            .put("output", largeOutput)
            .put("cwd", "/data/data/com.mobilefork.hermesagent/files")
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertTrue(parsed.getInt("_original_chars") > compacted.length)
        assertEquals(0, parsed.getInt("exit_code"))
        assertTrue(parsed.getString("output").contains("scroll-item"))
        assertTrue(parsed.getString("output").contains("hermes context compressed"))
        assertTrue(compacted.length < result.length)
    }

    @Test
    fun compactsDiagnosticArraysButKeepsTopRowsReadable() {
        val networks = JSONArray()
        repeat(60) { index ->
            networks.put(
                JSONObject()
                    .put("ssid", "Lab-$index")
                    .put("bssid_oui", "AC:BC:32")
                    .put("bssid_vendor", "Apple")
                    .put("rssi_dbm", -30 - index)
                    .put("signal_quality", "excellent")
                    .put("frequency_mhz", 2412 + index)
                    .put("channel", index + 1)
                    .put("channel_width", "20MHz")
                    .put("security_mode", "WPA2")
                    .put("capabilities", "[WPA2-PSK-CCMP][ESS]".repeat(8)),
            )
        }
        val channelRatings = JSONArray()
        repeat(20) { index ->
            channelRatings.put(
                JSONObject()
                    .put("band", "2.4GHz")
                    .put("channel", index + 1)
                    .put("score", 100 - index)
                    .put("rating_label", "good")
                    .put("network_count", index)
                    .put("overlap_count", index + 1)
                    .put("strongest_rssi_dbm", -40 - index)
                    .put("recommendation", "Use if this is the highest-scored row."),
            )
        }
        val channelUtilization = JSONArray()
        repeat(20) { index ->
            channelUtilization.put(
                JSONObject()
                    .put("band", "2.4GHz")
                    .put("channel", index + 1)
                    .put("channel_pressure_score", 90 - index)
                    .put("utilization_label", "crowded")
                    .put("network_count", index + 1)
                    .put("overlap_count", index + 2)
                    .put("strongest_rssi_dbm", -35 - index)
                    .put("average_rssi_dbm", -45 - index)
                    .put("max_channel_width_mhz", 40)
                    .put("wide_channel_count", 0)
                    .put("security_modes", JSONArray().put("WPA2"))
                    .put("sample_ssids", JSONArray().put("Lab-$index"))
                    .put("recommendation", "Crowded channel."),
            )
        }
        val channelGraph = JSONArray()
        repeat(20) { index ->
            channelGraph.put(
                JSONObject()
                    .put("rank", index + 1)
                    .put("display_ssid", "Graph-$index")
                    .put("ssid", "Graph-$index")
                    .put("bssid", "AC:BC:32:AA:00:$index")
                    .put("band", "5GHz")
                    .put("channel", 36 + index)
                    .put("graph_x_channel", 36 + index)
                    .put("rssi_dbm", -32 - index)
                    .put("graph_y_dbm", -32 - index)
                    .put("channel_width", "80MHz")
                    .put("channel_width_mhz", 80)
                    .put("graph_width_channels", 17)
                    .put("channel_span_start", 28 + index)
                    .put("channel_span_end", 44 + index)
                    .put("frequency_span_start_mhz", 5140 + (index * 5))
                    .put("frequency_span_end_mhz", 5220 + (index * 5))
                    .put("overlap_network_count", index + 1)
                    .put("same_channel_network_count", index)
                    .put("overlap_pressure_score", 80 - index)
                    .put("overlap_sample_ssids", JSONArray().put("Graph-peer-$index"))
                    .put("graph_shape", "channel_width_envelope")
                    .put("security_mode", "WPA2")
                    .put("bssid_vendor", "Apple"),
            )
        }
        val accessPointDetails = JSONArray()
        repeat(20) { index ->
            accessPointDetails.put(
                JSONObject()
                    .put("rank", index + 1)
                    .put("display_ssid", "Lab-$index")
                    .put("bssid", "AC:BC:32:00:00:$index")
                    .put("bssid_oui", "AC:BC:32")
                    .put("bssid_vendor", "Apple")
                    .put("rssi_dbm", -30 - index)
                    .put("signal_quality", "excellent")
                    .put("frequency_mhz", 5180)
                    .put("channel", 36)
                    .put("channel_width", "80MHz")
                    .put("channel_width_mhz", 80)
                    .put("wifi_standard", "802.11ac")
                    .put("security_mode", "WPA2")
                    .put("estimated_distance_m", 1.2),
            )
        }
        val accessPointSemantics = JSONArray()
        repeat(20) { index ->
            accessPointSemantics.put(
                JSONObject()
                    .put("rank", index + 1)
                    .put("display_ssid", "Lab-$index")
                    .put("semantic_label", if (index == 0) "guest/public hotspot" else "private router/AP")
                    .put("security_risk_label", if (index == 0) "open_network" else "standard_security")
                    .put("semantic_tags", JSONArray().put("private_router_ap").put("standard_security"))
                    .put("rssi_dbm", -30 - index)
                    .put("band", "5GHz")
                    .put("channel", 36)
                    .put("recommendation", "Semantic row $index"),
            )
        }
        val bandCoverage = JSONArray()
            .put(
                JSONObject()
                    .put("band", "5GHz")
                    .put("network_count", 60)
                    .put("visible_channels", JSONArray().put("36").put("40"))
                    .put("observed_widths", JSONArray().put("80MHz"))
                    .put("observed_standards", JSONArray().put("802.11ac"))
                    .put("recommended_channel", 36)
                    .put("recommended_score", 88)
                    .put("security_attention_count", 1),
            )
        val result = JSONObject()
            .put("success", true)
            .put("action", "wifi_scan")
            .put("wifi_vendor_count", 1)
            .put("wifi_filter_count", 4)
            .put("wifi_access_point_detail_count", 20)
            .put("wifi_access_point_semantic_count", 20)
            .put("wifi_band_coverage_count", 1)
            .put("wifi_channel_graph_count", 20)
            .put("wifi_channel_utilization_count", 20)
            .put("wifi_security_summary_count", 1)
            .put("wifi_width_summary_count", 1)
            .put("wifi_standard_summary_count", 1)
            .put("wifi_networks", networks)
            .put("wifi_access_point_details", accessPointDetails)
            .put("wifi_access_point_semantics", accessPointSemantics)
            .put("wifi_access_point_export", JSONObject().put("format", "json").put("row_count", 20).put("json_array_key", "wifi_access_point_details"))
            .put("wifi_channel_ratings", channelRatings)
            .put("wifi_channel_graph", channelGraph)
            .put("wifi_channel_utilization", channelUtilization)
            .put(
                "wifi_vendor_summary",
                JSONArray().put(
                    JSONObject()
                        .put("vendor", "Apple")
                        .put("network_count", 60)
                        .put("strongest_rssi_dbm", -30)
                        .put("bssid_ouis", JSONArray().put("AC:BC:32"))
                        .put("recommendation", "Strong nearby vendor group."),
                ),
            )
            .put(
                "wifi_analyzer_filters",
                JSONArray().put(
                    JSONObject()
                        .put("key", "security")
                        .put("label", "Security")
                        .put("options", JSONArray().put(JSONObject().put("value", "WPA2").put("count", 60))),
                ),
            )
            .put(
                "wifi_security_summary",
                JSONArray().put(JSONObject().put("security_mode", "WPA2").put("network_count", 60).put("strongest_rssi_dbm", -30)),
            )
            .put(
                "wifi_channel_width_summary",
                JSONArray().put(JSONObject().put("channel_width", "80MHz").put("channel_width_mhz", 80).put("network_count", 60)),
            )
            .put(
                "wifi_standard_summary",
                JSONArray().put(JSONObject().put("wifi_standard", "802.11ac").put("network_count", 60)),
            )
            .put(
                "recommended_wifi_channels",
                JSONArray().put(JSONObject().put("band", "2.4GHz").put("channel", 11).put("score", 96)),
            )
            .put("wifi_band_coverage", bandCoverage)
            .put("cards", JSONArray().put(JSONObject().put("title", "Wi-Fi Analyzer").put("body", "60 signals")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val wifiNetworks = parsed.getJSONObject("wifi_networks")
        val wifiDetails = parsed.getJSONObject("wifi_access_point_details")
        val wifiSemantics = parsed.getJSONObject("wifi_access_point_semantics")
        val wifiRatings = parsed.getJSONObject("wifi_channel_ratings")
        val wifiGraph = parsed.getJSONObject("wifi_channel_graph")
        val wifiUtilization = parsed.getJSONObject("wifi_channel_utilization")
        val compactedBandCoverage = parsed.getJSONArray("wifi_band_coverage")
        val vendorSummary = parsed.getJSONArray("wifi_vendor_summary")
        val analyzerFilters = parsed.getJSONArray("wifi_analyzer_filters")
        val securitySummary = parsed.getJSONArray("wifi_security_summary")
        val widthSummary = parsed.getJSONArray("wifi_channel_width_summary")
        val standardSummary = parsed.getJSONArray("wifi_standard_summary")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(1, parsed.getInt("wifi_vendor_count"))
        assertEquals(4, parsed.getInt("wifi_filter_count"))
        assertEquals(20, parsed.getInt("wifi_access_point_detail_count"))
        assertEquals(20, parsed.getInt("wifi_access_point_semantic_count"))
        assertEquals(1, parsed.getInt("wifi_band_coverage_count"))
        assertEquals(20, parsed.getInt("wifi_channel_graph_count"))
        assertEquals(20, parsed.getInt("wifi_channel_utilization_count"))
        assertEquals("array", wifiNetworks.getString("type"))
        assertEquals(60, wifiNetworks.getInt("original_count"))
        assertEquals(8, wifiNetworks.getJSONArray("items").length())
        assertEquals("Lab-0", wifiNetworks.getJSONArray("items").getJSONObject(0).getString("ssid"))
        assertEquals("Apple", wifiNetworks.getJSONArray("items").getJSONObject(0).getString("bssid_vendor"))
        assertEquals("WPA2", wifiNetworks.getJSONArray("items").getJSONObject(0).getString("security_mode"))
        assertEquals("array", wifiDetails.getString("type"))
        assertEquals(20, wifiDetails.getInt("original_count"))
        assertEquals("802.11ac", wifiDetails.getJSONArray("items").getJSONObject(0).getString("wifi_standard"))
        assertEquals(80, wifiDetails.getJSONArray("items").getJSONObject(0).getInt("channel_width_mhz"))
        assertEquals("array", wifiSemantics.getString("type"))
        assertEquals(20, wifiSemantics.getInt("original_count"))
        assertEquals("guest/public hotspot", wifiSemantics.getJSONArray("items").getJSONObject(0).getString("semantic_label"))
        assertEquals("open_network", wifiSemantics.getJSONArray("items").getJSONObject(0).getString("security_risk_label"))
        assertEquals("array", wifiRatings.getString("type"))
        assertEquals(20, wifiRatings.getInt("original_count"))
        assertEquals(1, wifiRatings.getJSONArray("items").getJSONObject(0).getInt("channel"))
        assertEquals("array", wifiGraph.getString("type"))
        assertEquals(20, wifiGraph.getInt("original_count"))
        assertEquals("channel_width_envelope", wifiGraph.getJSONArray("items").getJSONObject(0).getString("graph_shape"))
        assertEquals(17, wifiGraph.getJSONArray("items").getJSONObject(0).getInt("graph_width_channels"))
        assertEquals(80, wifiGraph.getJSONArray("items").getJSONObject(0).getInt("overlap_pressure_score"))
        assertEquals("array", wifiUtilization.getString("type"))
        assertEquals(20, wifiUtilization.getInt("original_count"))
        assertEquals(90, wifiUtilization.getJSONArray("items").getJSONObject(0).getInt("channel_pressure_score"))
        assertEquals("crowded", wifiUtilization.getJSONArray("items").getJSONObject(0).getString("utilization_label"))
        assertEquals("Apple", vendorSummary.getJSONObject(0).getString("vendor"))
        assertEquals("security", analyzerFilters.getJSONObject(0).getString("key"))
        assertEquals("WPA2", securitySummary.getJSONObject(0).getString("security_mode"))
        assertEquals("80MHz", widthSummary.getJSONObject(0).getString("channel_width"))
        assertEquals("802.11ac", standardSummary.getJSONObject(0).getString("wifi_standard"))
        assertEquals("5GHz", compactedBandCoverage.getJSONObject(0).getString("band"))
        assertEquals(60, compactedBandCoverage.getJSONObject(0).getInt("network_count"))
        assertEquals(11, parsed.getJSONArray("recommended_wifi_channels").getJSONObject(0).getInt("channel"))
        assertEquals("Wi-Fi Analyzer", parsed.getJSONArray("cards").getJSONObject(0).getString("title"))
    }

    @Test
    fun compactsWifiSignalHistoryWithoutDroppingTrendMetadata() {
        val history = JSONArray()
        repeat(16) { index ->
            history.put(
                JSONObject()
                    .put("ssid", "Lab-$index")
                    .put("bssid_vendor", "Apple")
                    .put("current_rssi_dbm", -35 - index)
                    .put("average_rssi_dbm", -40 - index)
                    .put("min_rssi_dbm", -50 - index)
                    .put("max_rssi_dbm", -35 - index)
                    .put("trend_db", index)
                    .put("trend_label", "improving")
                    .put("sample_count", 4)
                    .put("rssi_series", JSONArray().put(JSONObject().put("observed_at_ms", index.toLong()).put("rssi_dbm", -35 - index))),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "wifi_scan")
            .put("wifi_history_network_count", 16)
            .put("wifi_signal_history", history)
            .put("cards", JSONArray().put(JSONObject().put("title", "Wi-Fi History").put("body", "16 AP trends")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val signalHistory = parsed.getJSONObject("wifi_signal_history")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(16, parsed.getInt("wifi_history_network_count"))
        assertEquals("array", signalHistory.getString("type"))
        assertEquals(16, signalHistory.getInt("original_count"))
        assertEquals(4, signalHistory.getJSONArray("items").length())
        assertEquals(-35, signalHistory.getJSONArray("items").getJSONObject(0).getInt("current_rssi_dbm"))
        assertEquals("improving", signalHistory.getJSONArray("items").getJSONObject(0).getString("trend_label"))
    }

    @Test
    fun compactsAgentEnvironmentReportWithoutDroppingReadinessRows() {
        val capabilities = JSONArray()
        repeat(20) { index ->
            capabilities.put(
                JSONObject()
                    .put("category", "capability_$index")
                    .put("label", "Capability $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed capability row $index")
                    .put("recommendation", "Use the matching native tool.")
                    .put("fraction", 0.8),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "agent_environment_report")
            .put("agent_capability_count", 20)
            .put("ready_capability_count", 10)
            .put("kai_parity_count", 1)
            .put("kai_operations_count", 1)
            .put("ready_kai_operations_count", 1)
            .put("workflow_readiness_count", 1)
            .put("agent_tool_sandbox_count", 1)
            .put("ready_agent_tool_sandbox_count", 1)
            .put("mcp_tool_server_count", 1)
            .put("ready_mcp_tool_server_count", 0)
            .put("mcp_tool_server_route_count", 1)
            .put("ready_mcp_tool_server_route_count", 1)
            .put("agent_capability_matrix", capabilities)
            .put(
                "kai_parity_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "kai_parity")
                        .put("label", "Autonomous heartbeat")
                        .put("ready", true)
                        .put("value_label", "30s interval")
                        .put("parity_source", "Kai autonomous heartbeat"),
                    ),
            )
            .put(
                "kai_operations_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "kai_operations")
                        .put("label", "Tool and MCP bridge route")
                        .put("ready", true)
                        .put("value_label", "tool_catalog")
                        .put("tool_action", "android_device_diagnostics_tool:tool_catalog"),
                ),
            )
            .put(
                "agent_tool_sandbox_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "agent_tool_sandbox")
                        .put("label", "Terminal/Linux workspace surface")
                        .put("ready", true)
                        .put("value_label", "terminal_tool")
                        .put("tool_action", "terminal_tool")
                        .put("sandbox_scope", "app-private workspace")
                        .put("permission_gate", "app storage")
                        .put("host_access", "no host filesystem")
                        .put("remote_dispatch_capable", false)
                        .put("mcp_parity_status", "Kai Linux sandbox analogue"),
                ),
            )
            .put(
                "mcp_tool_server_registry",
                JSONArray().put(
                    JSONObject()
                        .put("category", "mcp_tool_server_registry")
                        .put("label", "Context7 documentation server")
                        .put("ready", false)
                        .put("value_label", "external docs MCP needed")
                        .put("tool_action", "external_mcp_context7")
                        .put("mcp_server_name", "Context7")
                        .put("route_status", "external_mcp_needed")
                        .put("remote_endpoint_required", true)
                        .put("streamable_http_supported", false)
                        .put("metadata_keys", JSONArray().put("library_id")),
                ),
            )
            .put(
                "mcp_tool_server_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "mcp_tool_server_route")
                        .put("label", "Prefer native Hermes tools first")
                        .put("ready", true)
                        .put("value_label", "native tools")
                        .put("tool_action", "android_device_diagnostics_tool:tool_catalog")
                        .put("route_policy", "native_tool_first")
                        .put("mcp_streamable_http_supported", false),
                ),
            )
            .put(
                "workflow_readiness_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "wireless_workflow")
                        .put("label", "Analyze nearby Wi-Fi")
                        .put("ready", true)
                        .put("value_label", "call wifi_ap_details"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Agent Environment").put("body", "20 rows")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val capabilityMatrix = parsed.getJSONObject("agent_capability_matrix")
        val kaiParity = parsed.getJSONArray("kai_parity_matrix")
        val kaiOperations = parsed.getJSONArray("kai_operations_matrix")
        val toolSandbox = parsed.getJSONArray("agent_tool_sandbox_matrix")
        val mcpRegistry = parsed.getJSONArray("mcp_tool_server_registry")
        val mcpRoutes = parsed.getJSONArray("mcp_tool_server_routes")
        val readiness = parsed.getJSONArray("workflow_readiness_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(20, parsed.getInt("agent_capability_count"))
        assertEquals(1, parsed.getInt("agent_tool_sandbox_count"))
        assertEquals(1, parsed.getInt("ready_agent_tool_sandbox_count"))
        assertEquals(1, parsed.getInt("mcp_tool_server_count"))
        assertEquals(1, parsed.getInt("mcp_tool_server_route_count"))
        assertEquals("array", capabilityMatrix.getString("type"))
        assertEquals(20, capabilityMatrix.getInt("original_count"))
        assertEquals("Capability 0", capabilityMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals(true, capabilityMatrix.getJSONArray("items").getJSONObject(0).getBoolean("ready"))
        assertEquals("Autonomous heartbeat", kaiParity.getJSONObject(0).getString("label"))
        assertEquals("Tool and MCP bridge route", kaiOperations.getJSONObject(0).getString("label"))
        assertEquals("android_device_diagnostics_tool:tool_catalog", kaiOperations.getJSONObject(0).getString("tool_action"))
        assertEquals("Terminal/Linux workspace surface", toolSandbox.getJSONObject(0).getString("label"))
        assertEquals("terminal_tool", toolSandbox.getJSONObject(0).getString("tool_action"))
        assertEquals("app-private workspace", toolSandbox.getJSONObject(0).getString("sandbox_scope"))
        assertEquals("Kai Linux sandbox analogue", toolSandbox.getJSONObject(0).getString("mcp_parity_status"))
        assertEquals("Context7 documentation server", mcpRegistry.getJSONObject(0).getString("label"))
        assertEquals("Context7", mcpRegistry.getJSONObject(0).getString("mcp_server_name"))
        assertEquals("external_mcp_needed", mcpRegistry.getJSONObject(0).getString("route_status"))
        assertFalse(mcpRegistry.getJSONObject(0).getBoolean("streamable_http_supported"))
        assertEquals("Prefer native Hermes tools first", mcpRoutes.getJSONObject(0).getString("label"))
        assertEquals("native_tool_first", mcpRoutes.getJSONObject(0).getString("route_policy"))
        assertEquals("Analyze nearby Wi-Fi", readiness.getJSONObject(0).getString("label"))
    }

    @Test
    fun compactsAgentCapabilityUpgradeReportWithoutDroppingObjectiveRows() {
        val objectives = JSONArray()
        repeat(12) { index ->
            objectives.put(
                JSONObject()
                    .put("category", "agent_upgrade_objective")
                    .put("label", "Upgrade objective $index")
                    .put("ready", index < 9)
                    .put("value_label", "objective value $index")
                    .put("detail", "Detailed upgrade objective row $index")
                    .put("recommendation", "Open the matching report.")
                    .put("evidence_status", "passive_analyzer_ready")
                    .put("bridge_required", index == 3)
                    .put("physical_device_validation_required", index == 5)
                    .put("source_actions", JSONArray().put("agent_capability_upgrade_report"))
                    .put("card_graph_types", JSONArray().put("agent_upgrade_objective_matrix"))
                    .put("fraction", 0.86),
            )
        }
        val routes = JSONArray().put(
            JSONObject()
                .put("category", "agent_upgrade_route")
                .put("label", "Start with full upgrade audit")
                .put("ready", true)
                .put("value_label", "agent_capability_upgrade_report")
                .put("tool_action", "agent_capability_upgrade_report")
                .put("graph_type", "agent_upgrade_objective_matrix")
                .put("bridge_required", false)
                .put("physical_device_validation_required", false),
        )
        val result = JSONObject()
            .put("success", true)
            .put("action", "agent_capability_upgrade_report")
            .put("agent_upgrade_objective_count", 12)
            .put("ready_agent_upgrade_objective_count", 9)
            .put("agent_upgrade_route_count", 1)
            .put("ready_agent_upgrade_route_count", 1)
            .put("agent_upgrade_objective_matrix", objectives)
            .put("agent_upgrade_route_matrix", routes)
            .put("gemma_upgrade_audit_directives", JSONArray().put("Read agent_upgrade_objective_matrix first."))
            .put("cards", JSONArray().put(JSONObject().put("title", "Upgrade Objective Matrix").put("body", "12 rows")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val objectiveMatrix = parsed.getJSONObject("agent_upgrade_objective_matrix")
        val routeMatrix = parsed.getJSONArray("agent_upgrade_route_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(12, parsed.getInt("agent_upgrade_objective_count"))
        assertEquals(9, parsed.getInt("ready_agent_upgrade_objective_count"))
        assertEquals("array", objectiveMatrix.getString("type"))
        assertEquals(12, objectiveMatrix.getInt("original_count"))
        assertEquals("Upgrade objective 0", objectiveMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("passive_analyzer_ready", objectiveMatrix.getJSONArray("items").getJSONObject(0).getString("evidence_status"))
        assertFalse(objectiveMatrix.getJSONArray("items").getJSONObject(0).getBoolean("bridge_required"))
        assertEquals("Start with full upgrade audit", routeMatrix.getJSONObject(0).getString("label"))
        assertEquals("agent_capability_upgrade_report", routeMatrix.getJSONObject(0).getString("tool_action"))
    }

    @Test
    fun compactsSignalWorkflowHandoffReportWithoutDroppingNextActionRows() {
        val handoffRows = JSONArray()
        repeat(12) { index ->
            handoffRows.put(
                JSONObject()
                    .put("category", "agent_signal_workflow_handoff")
                    .put("label", if (index == 0) "Open Wi-Fi Analyzer graph" else "Workflow handoff $index")
                    .put("ready", index < 9)
                    .put("value_label", "handoff value $index")
                    .put("detail", "Detailed workflow handoff row $index with refresh policy and permission gates")
                    .put("recommendation", "Open the matching next action.")
                    .put("handoff_status", "passive_first")
                    .put("tool_action", if (index == 0) "wifi_channel_graph" else "agent_signal_workflow_handoff_report")
                    .put("open_next_action", if (index == 0) "wifi_channel_graph" else "agent_signal_workflow_handoff_report")
                    .put("graph_type", if (index == 0) "wifi_channel_graph" else "agent_signal_workflow_handoff_matrix")
                    .put("refresh_policy", "passive_first")
                    .put("permission_gate", "source_report_permissions")
                    .put("bridge_required", index == 3)
                    .put("physical_device_validation_required", index == 5)
                    .put("fraction", 0.86),
            )
        }
        val routes = JSONArray()
        repeat(9) { index ->
            routes.put(
                JSONObject()
                    .put("category", "agent_signal_next_action_route")
                    .put("label", if (index == 0) "Open signal workflow handoff" else "Next action route $index")
                    .put("ready", true)
                    .put("value_label", "agent_signal_workflow_handoff_report")
                    .put("tool_action", "agent_signal_workflow_handoff_report")
                    .put("open_next_action", "agent_signal_workflow_handoff_report")
                    .put("graph_type", "agent_signal_workflow_handoff_matrix")
                    .put("refresh_policy", "passive_workflow_handoff")
                    .put("permission_gate", "source_report_permissions_and_hardware_boundaries")
                    .put("agent_question_patterns", JSONArray().put("what should i open next")),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "agent_signal_workflow_handoff_report")
            .put("agent_signal_workflow_handoff_count", 12)
            .put("ready_agent_signal_workflow_handoff_count", 9)
            .put("agent_signal_next_action_route_count", 9)
            .put("ready_agent_signal_next_action_route_count", 9)
            .put("agent_signal_workflow_handoff_matrix", handoffRows)
            .put("agent_signal_next_action_routes", routes)
            .put("gemma_signal_workflow_handoff_directives", JSONArray().put("Read open_next_action first."))
            .put("cards", JSONArray().put(JSONObject().put("title", "Signal Workflow Handoff").put("body", "12 rows")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val compactedHandoff = parsed.getJSONObject("agent_signal_workflow_handoff_matrix")
        val compactedRoutes = parsed.getJSONObject("agent_signal_next_action_routes")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(12, parsed.getInt("agent_signal_workflow_handoff_count"))
        assertEquals(9, parsed.getInt("ready_agent_signal_workflow_handoff_count"))
        assertEquals(9, parsed.getInt("agent_signal_next_action_route_count"))
        assertEquals("array", compactedHandoff.getString("type"))
        assertEquals(12, compactedHandoff.getInt("original_count"))
        assertEquals("Open Wi-Fi Analyzer graph", compactedHandoff.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("wifi_channel_graph", compactedHandoff.getJSONArray("items").getJSONObject(0).getString("open_next_action"))
        assertEquals("array", compactedRoutes.getString("type"))
        assertEquals(9, compactedRoutes.getInt("original_count"))
        assertEquals("Open signal workflow handoff", compactedRoutes.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("agent_signal_workflow_handoff_report", compactedRoutes.getJSONArray("items").getJSONObject(0).getString("tool_action"))
    }

    @Test
    fun compactsSignalPermissionRunbookReportWithoutDroppingRefreshRoutes() {
        val runbookRows = JSONArray()
        repeat(10) { index ->
            runbookRows.put(
                JSONObject()
                    .put("category", "agent_signal_permission_runbook")
                    .put("label", if (index == 0) "Prepare active Wi-Fi scan" else "Permission runbook $index")
                    .put("ready", index < 5)
                    .put("value_label", if (index == 0) "wifi_scan" else "action $index")
                    .put("detail", "Detailed permission runbook row $index with Android settings, user consent, active refresh arguments, passive fallback, bridge boundaries, and physical-device proof requirements.")
                    .put("recommendation", "Use active_refresh_arguments only after user consent and Android permission gates are satisfied.")
                    .put("tool_action", if (index == 0) "wifi_scan" else "agent_signal_permission_runbook_report")
                    .put("open_next_action", if (index == 0) "wifi_scan" else "agent_signal_permission_runbook_report")
                    .put("graph_type", "agent_signal_permission_runbook_matrix")
                    .put("permission_gate", "user_consent_and_android_permissions")
                    .put("settings_actions", JSONArray().put("open_app_settings").put("open_location_settings"))
                    .put("required_permissions", JSONArray().put("android.permission.ACCESS_FINE_LOCATION"))
                    .put("required_settings", JSONArray().put("location_services_enabled"))
                    .put("active_refresh_arguments", JSONObject().put("action", "wifi_scan").put("refresh", true))
                    .put("passive_fallback_action", "wifi_analyzer_report")
                    .put("user_consent_required", true)
                    .put("bridge_required", index == 3)
                    .put("physical_device_validation_required", true)
                    .put("active_refresh_requires_physical_device", true)
                    .put("runbook_status", "permission_gate_pending")
                    .put("fraction", 0.86),
            )
        }
        val routes = JSONArray()
        repeat(9) { index ->
            routes.put(
                JSONObject()
                    .put("category", "agent_signal_active_refresh_route")
                    .put("label", if (index == 0) "Run active Wi-Fi scan" else "Active refresh route $index")
                    .put("ready", true)
                    .put("value_label", "wifi_scan")
                    .put("detail", "Route row $index for safe active refresh after consent.")
                    .put("tool_action", "wifi_scan")
                    .put("open_next_action", "wifi_scan")
                    .put("graph_type", "agent_signal_active_refresh_routes")
                    .put("permission_gate", "nearby_wifi_or_location_permission")
                    .put("settings_actions", JSONArray().put("open_app_settings").put("open_location_settings"))
                    .put("active_refresh_arguments", JSONObject().put("action", "wifi_scan").put("refresh", true))
                    .put("passive_fallback_action", "agent_signal_permission_runbook_report")
                    .put("user_consent_required", true),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "agent_signal_permission_runbook_report")
            .put("agent_signal_permission_runbook_count", 10)
            .put("ready_agent_signal_permission_runbook_count", 5)
            .put("agent_signal_active_refresh_route_count", 9)
            .put("ready_agent_signal_active_refresh_route_count", 9)
            .put("agent_signal_permission_runbook_matrix", runbookRows)
            .put("agent_signal_active_refresh_routes", routes)
            .put("gemma_signal_permission_runbook_directives", JSONArray().put("Use active_refresh_arguments exactly."))
            .put("cards", JSONArray().put(JSONObject().put("title", "Signal Permission Runbook").put("body", "10 rows")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val compactedRunbook = parsed.getJSONObject("agent_signal_permission_runbook_matrix")
        val compactedRoutes = parsed.getJSONObject("agent_signal_active_refresh_routes")
        val firstRunbookRow = compactedRunbook.getJSONArray("items").getJSONObject(0)
        val firstRouteRow = compactedRoutes.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(10, parsed.getInt("agent_signal_permission_runbook_count"))
        assertEquals(5, parsed.getInt("ready_agent_signal_permission_runbook_count"))
        assertEquals(9, parsed.getInt("agent_signal_active_refresh_route_count"))
        assertEquals(9, parsed.getInt("ready_agent_signal_active_refresh_route_count"))
        assertEquals("array", compactedRunbook.getString("type"))
        assertEquals(10, compactedRunbook.getInt("original_count"))
        assertEquals("Prepare active Wi-Fi scan", firstRunbookRow.getString("label"))
        assertEquals("open_app_settings", firstRunbookRow.getJSONArray("settings_actions").getString(0))
        assertEquals("android.permission.ACCESS_FINE_LOCATION", firstRunbookRow.getJSONArray("required_permissions").getString(0))
        assertEquals("wifi_scan", firstRunbookRow.getJSONObject("active_refresh_arguments").getString("action"))
        assertEquals("wifi_analyzer_report", firstRunbookRow.getString("passive_fallback_action"))
        assertTrue(firstRunbookRow.getBoolean("user_consent_required"))
        assertTrue(firstRunbookRow.getBoolean("active_refresh_requires_physical_device"))
        assertEquals("array", compactedRoutes.getString("type"))
        assertEquals("Run active Wi-Fi scan", firstRouteRow.getString("label"))
        assertEquals("wifi_scan", firstRouteRow.getJSONObject("active_refresh_arguments").getString("action"))
        assertEquals("Use active_refresh_arguments exactly.", parsed.getJSONArray("gemma_signal_permission_runbook_directives").getString(0))
    }

    @Test
    fun compactsAgentObservationReportWithoutDroppingDashboardRows() {
        val observations = JSONArray()
        repeat(20) { index ->
            observations.put(
                JSONObject()
                    .put("category", "agent_observation")
                    .put("label", "Observation $index")
                    .put("ready", true)
                    .put("value_label", "route $index")
                    .put("detail", "Detailed observation dashboard row $index")
                    .put("recommendation", "Open the matching analyzer card.")
                    .put("tool_action", "agent_observation_report"),
            )
        }
        val cardManifest = JSONArray()
        repeat(16) { index ->
            cardManifest.put(
                JSONObject()
                    .put("category", "agent_card_manifest")
                    .put("label", if (index == 0) "Wi-Fi Channel Graph" else "Card manifest row $index")
                    .put("ready", true)
                    .put("value_label", if (index == 0) "wifi_channel_graph via wifi_analyzer_report" else "graph_$index")
                    .put("detail", "Detailed card manifest row $index")
                    .put("recommendation", "Open this expandable card for evidence.")
                    .put("tool_action", "wifi_analyzer_report")
                    .put("source_action", "wifi_analyzer_report")
                    .put("graph_type", if (index == 0) "wifi_channel_graph" else "graph_$index")
                    .put("card_title", if (index == 0) "Wi-Fi Channel Graph" else "Card $index")
                    .put("refresh_policy", "passive_by_default_refresh_when_needed")
                    .put("permission_gate", "nearby_wifi_or_location_permission"),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "agent_observation_report")
            .put("agent_observation_count", 20)
            .put("ready_agent_observation_count", 20)
            .put("agent_observation_route_count", 1)
            .put("agent_card_manifest_count", 16)
            .put("ready_agent_card_manifest_count", 16)
            .put("agent_observation_matrix", observations)
            .put("agent_card_manifest", cardManifest)
            .put("agent_card_graph_types", JSONArray().put("wifi_channel_graph").put("bluetooth_signal_history"))
            .put(
                "agent_observation_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "agent_observation_route")
                        .put("label", "Open Wi-Fi analyzer cards")
                        .put("ready", true)
                        .put("value_label", "wifi_analyzer_report")
                        .put("tool_action", "wifi_analyzer_report"),
                ),
            )
            .put("gemma_observation_directives", JSONArray().put("Read agent_observation_matrix first"))
            .put("cards", JSONArray().put(JSONObject().put("title", "Agent Observation").put("body", "20 rows")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val compactedObservations = parsed.getJSONObject("agent_observation_matrix")
        val compactedCardManifest = parsed.getJSONObject("agent_card_manifest")
        val routes = parsed.getJSONArray("agent_observation_routes")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(20, parsed.getInt("agent_observation_count"))
        assertEquals(16, parsed.getInt("agent_card_manifest_count"))
        assertEquals("array", compactedObservations.getString("type"))
        assertEquals(20, compactedObservations.getInt("original_count"))
        assertEquals("Observation 0", compactedObservations.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("agent_observation_report", compactedObservations.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("array", compactedCardManifest.getString("type"))
        assertEquals(16, compactedCardManifest.getInt("original_count"))
        assertEquals("Wi-Fi Channel Graph", compactedCardManifest.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("wifi_channel_graph", compactedCardManifest.getJSONArray("items").getJSONObject(0).getString("graph_type"))
        assertEquals("passive_by_default_refresh_when_needed", compactedCardManifest.getJSONArray("items").getJSONObject(0).getString("refresh_policy"))
        assertEquals("nearby_wifi_or_location_permission", compactedCardManifest.getJSONArray("items").getJSONObject(0).getString("permission_gate"))
        assertEquals("wifi_channel_graph", parsed.getJSONArray("agent_card_graph_types").getString(0))
        assertEquals("Open Wi-Fi analyzer cards", routes.getJSONObject(0).getString("label"))
        assertEquals("Read agent_observation_matrix first", parsed.getJSONArray("gemma_observation_directives").getString(0))
    }

    @Test
    fun compactsSignalAwarenessReportWithoutDroppingRoutesOrConstraints() {
        val awareness = JSONArray()
        repeat(18) { index ->
            awareness.put(
                JSONObject()
                    .put("category", "signal_awareness")
                    .put("label", "Signal row $index")
                    .put("ready", index % 3 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed signal awareness row $index")
                    .put("recommendation", "Use the right scanner.")
                    .put("tool_action", "wifi_ap_details")
                    .put("fraction", 0.75),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "signal_awareness_report")
            .put("signal_awareness_count", 18)
            .put("ready_signal_awareness_count", 6)
            .put("signal_workflow_route_count", 1)
            .put("signal_constraint_count", 1)
            .put("cached_wifi_history_network_count", 3)
            .put("signal_awareness_matrix", awareness)
            .put(
                "signal_workflow_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "signal_route")
                        .put("label", "Route Wi-Fi analyzer work")
                        .put("ready", true)
                        .put("value_label", "wifi_ap_details")
                        .put("tool_action", "wifi_ap_details"),
                ),
            )
            .put(
                "signal_constraint_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "signal_constraint")
                        .put("label", "AM/FM tuner public API")
                        .put("ready", false)
                        .put("value_label", "not public")
                        .put("constraint_type", "hardware_api"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Signal Awareness").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val awarenessMatrix = parsed.getJSONObject("signal_awareness_matrix")
        val routes = parsed.getJSONArray("signal_workflow_routes")
        val constraints = parsed.getJSONArray("signal_constraint_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("signal_awareness_count"))
        assertEquals(3, parsed.getInt("cached_wifi_history_network_count"))
        assertEquals("array", awarenessMatrix.getString("type"))
        assertEquals("Signal row 0", awarenessMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("wifi_ap_details", awarenessMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route Wi-Fi analyzer work", routes.getJSONObject(0).getString("label"))
        assertEquals("AM/FM tuner public API", constraints.getJSONObject(0).getString("label"))
        assertEquals("hardware_api", constraints.getJSONObject(0).getString("constraint_type"))
    }

    @Test
    fun compactsSocCompatibilityReportWithoutDroppingBackendPolicyRows() {
        val backend = JSONArray()
        repeat(18) { index ->
            backend.put(
                JSONObject()
                    .put("category", "soc_backend_parity")
                    .put("label", "SOC backend row $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "backend value $index")
                    .put("detail", "Detailed SOC backend compatibility row $index with MediaTek Mali PowerVR coverage")
                    .put("recommendation", "Use soc_compatibility_report before local inference decisions.")
                    .put("tool_action", "soc_compatibility_report")
                    .put("fraction", 0.82),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "soc_compatibility_report")
            .put("soc_backend_feature_count", 18)
            .put("ready_soc_backend_feature_count", 9)
            .put("soc_backend_route_count", 1)
            .put("soc_backend_constraint_count", 1)
            .put("likely_mediatek", true)
            .put("likely_snapdragon", false)
            .put("soc_backend_matrix", backend)
            .put(
                "soc_backend_policy_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "soc_backend_route")
                        .put("label", "Route SOC compatibility report")
                        .put("ready", true)
                        .put("value_label", "soc_compatibility_report")
                        .put("tool_action", "soc_compatibility_report"),
                ),
            )
            .put(
                "soc_backend_constraint_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "soc_backend_constraint")
                        .put("label", "Avoid Adreno-only assumptions")
                        .put("ready", true)
                        .put("value_label", "SOC-neutral")
                        .put("constraint_type", "soc_policy"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "SOC Compatibility").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val backendMatrix = parsed.getJSONObject("soc_backend_matrix")
        val routes = parsed.getJSONArray("soc_backend_policy_routes")
        val constraints = parsed.getJSONArray("soc_backend_constraint_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("soc_backend_feature_count"))
        assertEquals(9, parsed.getInt("ready_soc_backend_feature_count"))
        assertEquals(true, parsed.getBoolean("likely_mediatek"))
        assertEquals("array", backendMatrix.getString("type"))
        assertEquals("SOC backend row 0", backendMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("soc_compatibility_report", backendMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route SOC compatibility report", routes.getJSONObject(0).getString("label"))
        assertEquals("Avoid Adreno-only assumptions", constraints.getJSONObject(0).getString("label"))
    }

    @Test
    fun compactsGpuBackendRiskReportWithoutDroppingRiskRows() {
        val risks = JSONArray()
        repeat(18) { index ->
            risks.put(
                JSONObject()
                    .put("category", "gpu_backend_risk")
                    .put("label", if (index == 0) "Live accelerator acceptance" else "GPU risk row $index")
                    .put("ready", index < 8)
                    .put("value_label", if (index == 0) "gpu" else "risk value $index")
                    .put("detail", "Detailed GPU backend risk row $index with MediaTek Mali fallback context.")
                    .put("recommendation", "Use gpu_backend_risk_report before changing local inference policy.")
                    .put("risk_level", if (index == 0) "low" else "moderate")
                    .put("risk_score", if (index == 0) 10 else 45)
                    .put("runtime_signal", "accelerator_acceptance")
                    .put("mitigation", "CPU fallback is available.")
                    .put("tool_action", "local_backend_runtime_report"),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "gpu_backend_risk_report")
            .put("gpu_backend_risk_count", 18)
            .put("high_gpu_backend_risk_count", 0)
            .put("ready_gpu_backend_risk_count", 8)
            .put("gpu_backend_risk_route_count", 1)
            .put("gpu_backend_risk_level", "moderate")
            .put("gpu_backend_risk_score", 45)
            .put("gpu_backend_risk_matrix", risks)
            .put(
                "gpu_backend_risk_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "gpu_backend_risk_route")
                        .put("label", "Route GPU backend risk triage")
                        .put("ready", true)
                        .put("value_label", "gpu_backend_risk_report")
                        .put("tool_action", "gpu_backend_risk_report"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "GPU Backend Risk").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val riskMatrix = parsed.getJSONObject("gpu_backend_risk_matrix")
        val routes = parsed.getJSONArray("gpu_backend_risk_routes")
        val first = riskMatrix.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("gpu_backend_risk_count"))
        assertEquals("moderate", parsed.getString("gpu_backend_risk_level"))
        assertEquals(45, parsed.getInt("gpu_backend_risk_score"))
        assertEquals("array", riskMatrix.getString("type"))
        assertEquals("Live accelerator acceptance", first.getString("label"))
        assertEquals("low", first.getString("risk_level"))
        assertEquals(10, first.getInt("risk_score"))
        assertEquals("accelerator_acceptance", first.getString("runtime_signal"))
        assertEquals("local_backend_runtime_report", first.getString("tool_action"))
        assertEquals("Route GPU backend risk triage", routes.getJSONObject(0).getString("label"))
    }

    @Test
    fun compactsLocalInferenceCompatibilityReportWithoutDroppingScorecardRows() {
        val rows = JSONArray()
        repeat(18) { index ->
            rows.put(
                JSONObject()
                    .put("category", "local_inference_compatibility")
                    .put("label", if (index == 0) "MediaTek and non-Adreno fallback policy" else "Compatibility row $index")
                    .put("ready", index < 10)
                    .put("value_label", if (index == 0) "non-Adreno path visible" else "scorecard value $index")
                    .put("detail", "Detailed local inference compatibility row $index with MediaTek Mali PowerVR fallback context.")
                    .put("recommendation", "Use local_inference_compatibility_report before local acceleration claims.")
                    .put("tool_action", if (index == 0) "gpu_backend_risk_report" else "soc_compatibility_report")
                    .put("graph_type", if (index == 0) "gpu_backend_risk_matrix" else "soc_backend_matrix"),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "local_inference_compatibility_report")
            .put("local_inference_compatibility_score", 74)
            .put("local_inference_compatibility_level", "watch")
            .put("local_inference_compatibility_count", 18)
            .put("ready_local_inference_compatibility_count", 10)
            .put("local_inference_compatibility_matrix", rows)
            .put("cards", JSONArray().put(JSONObject().put("title", "Local Inference Compatibility").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val matrix = parsed.getJSONObject("local_inference_compatibility_matrix")
        val first = matrix.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(74, parsed.getInt("local_inference_compatibility_score"))
        assertEquals("watch", parsed.getString("local_inference_compatibility_level"))
        assertEquals(18, parsed.getInt("local_inference_compatibility_count"))
        assertEquals(10, parsed.getInt("ready_local_inference_compatibility_count"))
        assertEquals("array", matrix.getString("type"))
        assertEquals("MediaTek and non-Adreno fallback policy", first.getString("label"))
        assertEquals("gpu_backend_risk_report", first.getString("tool_action"))
        assertEquals("gpu_backend_risk_matrix", first.getString("graph_type"))
    }

    @Test
    fun compactsSignalEvidenceReportWithoutDroppingEvidenceRows() {
        val rows = JSONArray()
        repeat(18) { index ->
            rows.put(
                JSONObject()
                    .put("category", "signal_evidence")
                    .put("label", if (index == 0) "Wi-Fi AP and channel evidence" else "Signal evidence row $index")
                    .put("ready", index < 9)
                    .put("value_label", if (index == 0) "3 AP(s), 3 graph row(s)" else "evidence value $index")
                    .put("detail", "Detailed signal evidence row $index with Wi-Fi Bluetooth motion radio and local inference context.")
                    .put("recommendation", "Use agent_signal_evidence_report before explaining what Gemma can currently view.")
                    .put("tool_action", if (index == 0) "wifi_channel_graph" else "agent_signal_evidence_report")
                    .put("graph_type", if (index == 0) "wifi_channel_graph" else "signal_evidence_matrix")
                    .put("evidence_key", if (index == 0) "wifi_ap_channel" else "evidence_$index"),
            )
        }
        val routes = JSONArray().put(
            JSONObject()
                .put("category", "signal_evidence_route")
                .put("label", "Open signal evidence bundle")
                .put("ready", true)
                .put("value_label", "agent_signal_evidence_report")
                .put("tool_action", "agent_signal_evidence_report"),
        )
        val result = JSONObject()
            .put("success", true)
            .put("action", "agent_signal_evidence_report")
            .put("signal_evidence_count", 18)
            .put("ready_signal_evidence_count", 9)
            .put("signal_evidence_route_count", 1)
            .put("signal_evidence_graph_type_count", 3)
            .put("signal_evidence_matrix", rows)
            .put("signal_evidence_routes", routes)
            .put("signal_evidence_graph_types", JSONArray().put("signal_evidence_matrix").put("wifi_channel_graph").put("local_inference_compatibility_matrix"))
            .put("cards", JSONArray().put(JSONObject().put("title", "Signal Evidence Bundle").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val matrix = parsed.getJSONObject("signal_evidence_matrix")
        val first = matrix.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("signal_evidence_count"))
        assertEquals(9, parsed.getInt("ready_signal_evidence_count"))
        assertEquals(1, parsed.getInt("signal_evidence_route_count"))
        assertEquals(3, parsed.getInt("signal_evidence_graph_type_count"))
        assertEquals("array", matrix.getString("type"))
        assertEquals("Wi-Fi AP and channel evidence", first.getString("label"))
        assertEquals("wifi_channel_graph", first.getString("tool_action"))
        assertEquals("wifi_channel_graph", first.getString("graph_type"))
        assertEquals("wifi_ap_channel", first.getString("evidence_key"))
    }

    @Test
    fun compactsRuntimeBackendReportWithoutDroppingHealthRows() {
        val rows = JSONArray()
        repeat(18) { index ->
            rows.put(
                JSONObject()
                    .put("category", "runtime_backend")
                    .put("label", if (index == 0) "LiteRT-LM /health accelerator" else "Runtime row $index")
                    .put("ready", index == 0)
                    .put("value_label", if (index == 0) "gpu" else "runtime")
                    .put("detail", "Detailed runtime row $index with MediaTek Mali fallback and /health accelerator context.")
                    .put("recommendation", "Use local_backend_runtime_report before local inference decisions.")
                    .put("source_surface", "/health")
                    .put("health_url", "http://127.0.0.1:15436/health"),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "local_backend_runtime_report")
            .put("runtime_backend_feature_count", 18)
            .put("ready_runtime_backend_feature_count", 1)
            .put(
                "current_local_backend",
                JSONObject()
                    .put("backend_kind", "litert-lm")
                    .put("started", true)
                    .put("base_url", "http://127.0.0.1:15436/v1")
                    .put("health_url", "http://127.0.0.1:15436/health"),
            )
            .put(
                "litert_runtime_health",
                JSONObject()
                    .put("status", "ok")
                    .put("accelerator", "gpu")
                    .put("gpu_policy", "enabled: OpenCL library was loadable for ARM MediaTek/Mali")
                    .put("gpu_fallback_to_cpu", false),
            )
            .put("runtime_backend_matrix", rows)
            .put("cards", JSONArray().put(JSONObject().put("title", "Runtime Backend Health").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val matrix = parsed.getJSONObject("runtime_backend_matrix")
        val first = matrix.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("runtime_backend_feature_count"))
        assertEquals(1, parsed.getInt("ready_runtime_backend_feature_count"))
        assertEquals("litert-lm", parsed.getJSONObject("current_local_backend").getString("backend_kind"))
        assertEquals("gpu", parsed.getJSONObject("litert_runtime_health").getString("accelerator"))
        assertEquals("array", matrix.getString("type"))
        assertEquals(18, matrix.getInt("original_count"))
        assertEquals("LiteRT-LM /health accelerator", first.getString("label"))
        assertEquals("/health", first.getString("source_surface"))
        assertEquals("http://127.0.0.1:15436/health", first.getString("health_url"))
    }

    @Test
    fun compactsAcceleratorPreflightReportWithoutDroppingDelegateRows() {
        val rows = JSONArray()
        repeat(18) { index ->
            rows.put(
                JSONObject()
                    .put("category", "accelerator_preflight")
                    .put("label", if (index == 0) "OpenCL library visibility" else "Accelerator row $index")
                    .put("ready", index < 4)
                    .put("value_label", if (index == 0) "visible" else "delegate preflight")
                    .put("detail", "Detailed accelerator preflight row $index with MediaTek Mali OpenCL and CPU fallback context.")
                    .put("recommendation", "Use accelerator_preflight_report before starting the local model.")
                    .put("opencl_library_visible", index == 0)
                    .put("tool_action", if (index == 0) "local_backend_runtime_report" else "soc_compatibility_report"),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "accelerator_preflight_report")
            .put("accelerator_preflight_count", 18)
            .put("ready_accelerator_preflight_count", 4)
            .put("accelerator_preflight_matrix", rows)
            .put("cards", JSONArray().put(JSONObject().put("title", "Accelerator Preflight").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val matrix = parsed.getJSONObject("accelerator_preflight_matrix")
        val first = matrix.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("accelerator_preflight_count"))
        assertEquals(4, parsed.getInt("ready_accelerator_preflight_count"))
        assertEquals("array", matrix.getString("type"))
        assertEquals(18, matrix.getInt("original_count"))
        assertEquals("OpenCL library visibility", first.getString("label"))
        assertEquals(true, first.getBoolean("opencl_library_visible"))
        assertEquals("local_backend_runtime_report", first.getString("tool_action"))
    }

    @Test
    fun compactsNonAdrenoBackendAdvisorWithoutDroppingLaunchRows() {
        val rows = JSONArray()
        repeat(18) { index ->
            rows.put(
                JSONObject()
                    .put("category", "non_adreno_backend_advisor")
                    .put("label", if (index == 0) "Classify device family before launch" else "Backend advisor row $index")
                    .put("ready", index < 5)
                    .put("value_label", if (index == 0) "MediaTek / Mali" else "launch advisor")
                    .put("detail", "Detailed non-Adreno backend advisor row $index with MediaTek Mali PowerVR launch context.")
                    .put("recommendation", "Use non_adreno_backend_advisor_report before starting local inference.")
                    .put("tool_action", if (index == 0) "soc_compatibility_report" else "accelerator_preflight_report")
                    .put("graph_type", if (index == 0) "soc_backend_matrix" else "accelerator_preflight_matrix")
                    .put("non_adreno_policy_active", index == 0),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "non_adreno_backend_advisor_report")
            .put("non_adreno_backend_advisor_score", 76)
            .put("non_adreno_backend_advisor_level", "good")
            .put("non_adreno_backend_advisor_count", 18)
            .put("ready_non_adreno_backend_advisor_count", 5)
            .put("non_adreno_backend_advisor_matrix", rows)
            .put(
                "non_adreno_backend_launch_sequence",
                JSONArray()
                    .put("soc_compatibility_report")
                    .put("accelerator_preflight_report")
                    .put("local_backend_runtime_report"),
            )
            .put(
                "gemma_non_adreno_backend_directives",
                JSONArray().put("Open non_adreno_backend_advisor_matrix before starting local inference."),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Non-Adreno Backend Advisor").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val matrix = parsed.getJSONObject("non_adreno_backend_advisor_matrix")
        val first = matrix.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(76, parsed.getInt("non_adreno_backend_advisor_score"))
        assertEquals("good", parsed.getString("non_adreno_backend_advisor_level"))
        assertEquals(18, parsed.getInt("non_adreno_backend_advisor_count"))
        assertEquals(5, parsed.getInt("ready_non_adreno_backend_advisor_count"))
        assertEquals("array", matrix.getString("type"))
        assertEquals(18, matrix.getInt("original_count"))
        assertEquals("Classify device family before launch", first.getString("label"))
        assertEquals("soc_compatibility_report", first.getString("tool_action"))
        assertEquals("soc_backend_matrix", first.getString("graph_type"))
        assertTrue(parsed.get("non_adreno_backend_launch_sequence").toString().contains("local_backend_runtime_report"))
        assertTrue(parsed.get("gemma_non_adreno_backend_directives").toString().contains("before starting local inference"))
    }

    @Test
    fun compactsMediatekBackendLaunchChecklistWithoutDroppingGates() {
        val rows = JSONArray()
        repeat(16) { index ->
            rows.put(
                JSONObject()
                    .put("category", "mediatek_backend_launch_checklist")
                    .put("label", if (index == 0) "Verify GPU proof or name CPU fallback" else "Launch gate $index")
                    .put("ready", index < 6)
                    .put("value_label", if (index == 0) "cpu fallback" else "gate")
                    .put("detail", "Detailed MediaTek launch gate $index")
                    .put("recommendation", "Use the source report before starting local inference.")
                    .put("launch_step", index + 1)
                    .put("launch_gate_status", if (index < 6) "ready" else "needs_runtime")
                    .put("status_label", if (index < 6) "ready" else "needs_runtime")
                    .put("tool_action", if (index == 0) "local_backend_runtime_report" else "accelerator_preflight_report")
                    .put("graph_type", if (index == 0) "runtime_backend_matrix" else "accelerator_preflight_matrix")
                    .put("live_runtime_proof", index == 0)
                    .put("cpu_fallback_explicit", index == 0),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "mediatek_backend_launch_checklist_report")
            .put("mediatek_backend_launch_checklist_count", 16)
            .put("ready_mediatek_backend_launch_checklist_count", 6)
            .put("blocked_mediatek_backend_launch_checklist_count", 2)
            .put("mediatek_backend_launch_checklist_matrix", rows)
            .put(
                "gemma_mediatek_launch_directives",
                JSONArray().put("Open mediatek_backend_launch_checklist_matrix before starting local inference."),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "MediaTek Launch Checklist").put("body", "16 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val matrix = parsed.getJSONObject("mediatek_backend_launch_checklist_matrix")
        val first = matrix.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(16, parsed.getInt("mediatek_backend_launch_checklist_count"))
        assertEquals(6, parsed.getInt("ready_mediatek_backend_launch_checklist_count"))
        assertEquals(2, parsed.getInt("blocked_mediatek_backend_launch_checklist_count"))
        assertEquals("array", matrix.getString("type"))
        assertEquals(16, matrix.getInt("original_count"))
        assertEquals("Verify GPU proof or name CPU fallback", first.getString("label"))
        assertEquals(1, first.getInt("launch_step"))
        assertEquals("ready", first.getString("launch_gate_status"))
        assertEquals("local_backend_runtime_report", first.getString("tool_action"))
        assertTrue(first.getBoolean("live_runtime_proof"))
        assertTrue(first.getBoolean("cpu_fallback_explicit"))
        assertTrue(parsed.get("gemma_mediatek_launch_directives").toString().contains("before starting local inference"))
    }

    @Test
    fun compactsWifiAnalyzerReportWithoutDroppingPolicyRoutes() {
        val features = JSONArray()
        repeat(18) { index ->
            features.put(
                JSONObject()
                    .put("category", "wifi_analyzer_parity")
                    .put("label", "Wi-Fi analyzer feature $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed Wi-Fi analyzer feature row $index")
                    .put("recommendation", "Route through the matching Wi-Fi diagnostic action.")
                    .put("feature_source", "WiFiAnalyzer parity")
                    .put("tool_action", "wifi_channel_rating")
                    .put("fraction", 0.82),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "wifi_analyzer_report")
            .put("wifi_analyzer_feature_count", 18)
            .put("ready_wifi_analyzer_feature_count", 9)
            .put("wifi_analyzer_workflow_route_count", 1)
            .put("wifi_scan_policy_count", 1)
            .put("wifi_analyzer_feature_matrix", features)
            .put(
                "wifi_analyzer_workflow_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "wifi_analyzer_route")
                        .put("label", "Route best-channel analysis")
                        .put("ready", true)
                        .put("value_label", "wifi_channel_rating")
                        .put("tool_action", "wifi_channel_rating"),
                ),
            )
            .put(
                "wifi_scan_policy_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "wifi_scan_policy")
                        .put("label", "Android scan throttling")
                        .put("ready", true)
                        .put("value_label", "passive by default")
                        .put("constraint_type", "platform_policy")
                        .put("permission_gate", "android_wifi_scan"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Wi-Fi Analyzer Readiness").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val featureMatrix = parsed.getJSONObject("wifi_analyzer_feature_matrix")
        val routes = parsed.getJSONArray("wifi_analyzer_workflow_routes")
        val policies = parsed.getJSONArray("wifi_scan_policy_matrix")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("wifi_analyzer_feature_count"))
        assertEquals(9, parsed.getInt("ready_wifi_analyzer_feature_count"))
        assertEquals("array", featureMatrix.getString("type"))
        assertEquals("Wi-Fi analyzer feature 0", featureMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("WiFiAnalyzer parity", featureMatrix.getJSONArray("items").getJSONObject(0).getString("feature_source"))
        assertEquals("wifi_channel_rating", featureMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route best-channel analysis", routes.getJSONObject(0).getString("label"))
        assertEquals("Android scan throttling", policies.getJSONObject(0).getString("label"))
        assertEquals("platform_policy", policies.getJSONObject(0).getString("constraint_type"))
        assertEquals("android_wifi_scan", policies.getJSONObject(0).getString("permission_gate"))
    }

    @Test
    fun compactsBluetoothAnalyzerReportWithoutDroppingPolicyRoutes() {
        val features = JSONArray()
        repeat(18) { index ->
            features.put(
                JSONObject()
                    .put("category", "bluetooth_analyzer_parity")
                    .put("label", "Bluetooth analyzer feature $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed Bluetooth analyzer feature row $index with scan, service UUID, manufacturer, and proximity context.")
                    .put("recommendation", "Route through the matching Bluetooth diagnostic action.")
                    .put("feature_source", "Android BluetoothLeScanner")
                    .put("tool_action", "bluetooth_scan")
                    .put("fraction", 0.82),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "bluetooth_analyzer_report")
            .put("bluetooth_analyzer_feature_count", 18)
            .put("ready_bluetooth_analyzer_feature_count", 9)
            .put("bluetooth_analyzer_workflow_route_count", 1)
            .put("bluetooth_scan_policy_count", 1)
            .put("bluetooth_signal_history_count", 1)
            .put("bluetooth_analyzer_feature_matrix", features)
            .put(
                "bluetooth_analyzer_workflow_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "bluetooth_analyzer_route")
                        .put("label", "Route nearby Bluetooth scan")
                        .put("ready", true)
                        .put("value_label", "bluetooth_scan")
                        .put("tool_action", "bluetooth_scan"),
                ),
            )
            .put(
                "bluetooth_scan_policy_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "bluetooth_scan_policy")
                        .put("label", "Connect and scan permissions")
                        .put("ready", true)
                        .put("value_label", "connect and scan granted")
                        .put("constraint_type", "permission")
                        .put("permission_gate", "Bluetooth connect and scan"),
                ),
            )
            .put(
                "bluetooth_signal_history",
                JSONArray().put(
                    JSONObject()
                        .put("device_name", "Heart Strap")
                        .put("current_rssi_dbm", -58)
                        .put("average_rssi_dbm", -65)
                        .put("trend_label", "approaching")
                        .put("trend_db", 14)
                        .put("service_labels", JSONArray().put("Heart Rate"))
                        .put("manufacturer_names", JSONArray().put("Apple"))
                        .put("semantic_context", "services=Heart Rate | manufacturers=Apple"),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Bluetooth Analyzer Readiness").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val featureMatrix = parsed.getJSONObject("bluetooth_analyzer_feature_matrix")
        val routes = parsed.getJSONArray("bluetooth_analyzer_workflow_routes")
        val policies = parsed.getJSONArray("bluetooth_scan_policy_matrix")
        val history = parsed.getJSONArray("bluetooth_signal_history")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("bluetooth_analyzer_feature_count"))
        assertEquals(9, parsed.getInt("ready_bluetooth_analyzer_feature_count"))
        assertEquals("array", featureMatrix.getString("type"))
        assertEquals("Bluetooth analyzer feature 0", featureMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("Android BluetoothLeScanner", featureMatrix.getJSONArray("items").getJSONObject(0).getString("feature_source"))
        assertEquals("bluetooth_scan", featureMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route nearby Bluetooth scan", routes.getJSONObject(0).getString("label"))
        assertEquals("Connect and scan permissions", policies.getJSONObject(0).getString("label"))
        assertEquals("permission", policies.getJSONObject(0).getString("constraint_type"))
        assertEquals("Bluetooth connect and scan", policies.getJSONObject(0).getString("permission_gate"))
        assertEquals("Heart Strap", history.getJSONObject(0).getString("device_name"))
        assertEquals("Heart Rate", history.getJSONObject(0).getJSONArray("service_labels").getString(0))
        assertEquals("Apple", history.getJSONObject(0).getJSONArray("manufacturer_names").getString(0))
        assertTrue(history.getJSONObject(0).getString("semantic_context").contains("manufacturers=Apple"))
        assertEquals(1, parsed.getInt("bluetooth_signal_history_count"))
    }

    @Test
    fun compactsBluetoothAdvisorReportWithoutDroppingDecisionRowsOrCandidates() {
        val advisorRows = JSONArray()
        repeat(14) { index ->
            advisorRows.put(
                JSONObject()
                    .put("category", "bluetooth_signal_advisor")
                    .put("label", "Bluetooth advisor decision $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "decision $index")
                    .put("detail", "Bluetooth advisor row $index with candidate, trend, metadata, permission, detail, and RF coexistence evidence.")
                    .put("recommendation", "Route through the Bluetooth advisor matrix.")
                    .put("tool_action", "bluetooth_device_details")
                    .put("graph_type", "bluetooth_signal_advisor_matrix")
                    .put("fraction", 0.84),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "bluetooth_signal_advisor_report")
            .put("bluetooth_signal_advisor_count", 14)
            .put("ready_bluetooth_signal_advisor_count", 7)
            .put("bluetooth_device_candidate_count", 1)
            .put("bluetooth_signal_advisor_matrix", advisorRows)
            .put(
                "bluetooth_device_candidates",
                JSONArray().put(
                    JSONObject()
                        .put("category", "bluetooth_device_candidate")
                        .put("label", "Hermes Heart")
                        .put("ready", true)
                        .put("value_label", "-47 dBm near")
                        .put("candidate_score", 94)
                        .put("rssi_dbm", -47)
                        .put("address_suffix", "...11:22")
                        .put("paired", true)
                        .put("service_labels", JSONArray().put("Heart Rate"))
                        .put("manufacturer_names", JSONArray().put("Apple"))
                        .put("tool_action", "bluetooth_device_details"),
                ),
            )
            .put(
                "gemma_bluetooth_advisor_directives",
                JSONArray().put("Use bluetooth_signal_advisor_matrix first."),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Bluetooth Advisor").put("body", "14 rows")))

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result.toString()))
        val matrix = parsed.getJSONObject("bluetooth_signal_advisor_matrix")
        val candidates = parsed.getJSONArray("bluetooth_device_candidates")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(14, parsed.getInt("bluetooth_signal_advisor_count"))
        assertEquals(7, parsed.getInt("ready_bluetooth_signal_advisor_count"))
        assertEquals("array", matrix.getString("type"))
        assertEquals("Bluetooth advisor decision 0", matrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("bluetooth_device_details", matrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Hermes Heart", candidates.getJSONObject(0).getString("label"))
        assertEquals(94, candidates.getJSONObject(0).getInt("candidate_score"))
        assertEquals("Heart Rate", candidates.getJSONObject(0).getJSONArray("service_labels").getString(0))
        assertEquals("Use bluetooth_signal_advisor_matrix first.", parsed.getJSONArray("gemma_bluetooth_advisor_directives").getString(0))
    }

    @Test
    fun compactsBluetoothNearbyDecisionPacketWithoutDroppingRoutesOrBoundaries() {
        val packetRows = JSONArray()
        repeat(12) { index ->
            packetRows.put(
                JSONObject()
                    .put("category", "bluetooth_nearby_decision_packet")
                    .put("label", "Bluetooth nearby packet $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "packet $index")
                    .put("detail", "Bluetooth nearby decision packet $index with metadata, RF, and MediaTek context.")
                    .put("recommendation", "Keep Bluetooth claims bounded.")
                    .put("tool_action", "bluetooth_nearby_decision_packet_report")
                    .put("graph_type", "bluetooth_nearby_decision_packet")
                    .put("source_graph_type", "bluetooth_device_candidates")
                    .put("decision_status", "candidate_available")
                    .put("claim_scope", "Android-visible Bluetooth metadata only")
                    .put("active_refresh_action", "bluetooth_scan")
                    .put("passive_fallback_action", "bluetooth_signal_advisor_report")
                    .put("mediatek_sensitive", true)
                    .put("rf_coexistence_sensitive", true)
                    .put("fraction", 0.84),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "bluetooth_nearby_decision_packet_report")
            .put("bluetooth_nearby_decision_packet_count", 12)
            .put("ready_bluetooth_nearby_decision_packet_count", 6)
            .put("bluetooth_nearby_decision_packet", packetRows)
            .put(
                "bluetooth_nearby_decision_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "bluetooth_nearby_decision_route")
                        .put("label", "Nearby BLE refresh route")
                        .put("ready", true)
                        .put("tool_action", "bluetooth_scan")
                        .put("target_graph_type", "bluetooth_rssi"),
                ),
            )
            .put(
                "bluetooth_nearby_claim_boundaries",
                JSONArray().put(
                    JSONObject()
                        .put("category", "bluetooth_nearby_claim_boundary")
                        .put("label", "MediaTek/backend boundary")
                        .put("ready", true)
                        .put("claim_scope", "SOC/backend compatibility context only"),
                ),
            )
            .put(
                "gemma_bluetooth_nearby_directives",
                JSONArray().put("Use bluetooth_nearby_decision_packet first."),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Bluetooth Nearby Decision").put("body", "12 rows")))

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result.toString()))
        val packet = parsed.getJSONObject("bluetooth_nearby_decision_packet")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(12, parsed.getInt("bluetooth_nearby_decision_packet_count"))
        assertEquals("array", packet.getString("type"))
        assertEquals("Bluetooth nearby packet 0", packet.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("candidate_available", packet.getJSONArray("items").getJSONObject(0).getString("decision_status"))
        assertTrue(packet.getJSONArray("items").getJSONObject(0).getBoolean("mediatek_sensitive"))
        assertEquals("Nearby BLE refresh route", parsed.getJSONArray("bluetooth_nearby_decision_routes").getJSONObject(0).getString("label"))
        assertEquals("MediaTek/backend boundary", parsed.getJSONArray("bluetooth_nearby_claim_boundaries").getJSONObject(0).getString("label"))
        assertEquals("Use bluetooth_nearby_decision_packet first.", parsed.getJSONArray("gemma_bluetooth_nearby_directives").getString(0))
    }

    @Test
    fun compactsSensorAnalyzerReportWithoutDroppingPolicyRoutes() {
        val features = JSONArray()
        repeat(18) { index ->
            features.put(
                JSONObject()
                    .put("category", "sensor_analyzer_parity")
                    .put("label", "Sensor analyzer feature $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "value $index")
                    .put("detail", "Detailed Sensor Analyzer row $index with accelerometer, gyroscope, metadata, and sampling-policy context.")
                    .put("recommendation", "Route through the matching sensor diagnostic action.")
                    .put("feature_source", "Android SensorManager")
                    .put("tool_action", "sensor_snapshot")
                    .put("fraction", 0.82),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "sensor_analyzer_report")
            .put("sensor_analyzer_feature_count", 18)
            .put("ready_sensor_analyzer_feature_count", 9)
            .put("sensor_analyzer_workflow_route_count", 1)
            .put("sensor_sampling_policy_count", 1)
            .put("motion_sensor_quality_count", 1)
            .put("ready_motion_sensor_quality_count", 1)
            .put("motion_sensor_quality_score", 91)
            .put("motion_sensor_quality_level", "ready")
            .put(
                "sensor_sampling_status",
                JSONObject()
                    .put("active_sample_requested", false)
                    .put("passive_report_default", true)
                    .put("requested_available_sensor_count", 2),
            )
            .put("sensor_analyzer_feature_matrix", features)
            .put(
                "sensor_analyzer_workflow_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "sensor_analyzer_route")
                        .put("label", "Route one-shot motion sample")
                        .put("ready", true)
                        .put("value_label", "sensor_snapshot")
                        .put("tool_action", "sensor_snapshot"),
                ),
            )
            .put(
                "sensor_sampling_policy_matrix",
                JSONArray().put(
                    JSONObject()
                        .put("category", "sensor_sampling_policy")
                        .put("label", "Passive report default")
                        .put("ready", true)
                        .put("value_label", "no live sample")
                        .put("constraint_type", "sampling_cadence"),
                ),
            )
            .put(
                "motion_sensor_quality",
                JSONArray().put(
                    JSONObject()
                        .put("category", "motion_sensor_quality")
                        .put("label", "IMU fusion source coverage")
                        .put("ready", true)
                        .put("value_label", "4/6 source(s)")
                        .put("quality_signal", "fusion_sources")
                        .put("tool_action", "motion_sensor_quality")
                        .put("source_sensors", JSONArray().put("accelerometer").put("gyroscope").put("rotation_vector")),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Sensor Analyzer Readiness").put("body", "18 rows")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val featureMatrix = parsed.getJSONObject("sensor_analyzer_feature_matrix")
        val routes = parsed.getJSONArray("sensor_analyzer_workflow_routes")
        val policies = parsed.getJSONArray("sensor_sampling_policy_matrix")
        val quality = parsed.getJSONArray("motion_sensor_quality")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(18, parsed.getInt("sensor_analyzer_feature_count"))
        assertEquals(9, parsed.getInt("ready_sensor_analyzer_feature_count"))
        assertEquals(1, parsed.getInt("motion_sensor_quality_count"))
        assertEquals("ready", parsed.getString("motion_sensor_quality_level"))
        assertFalse(parsed.getJSONObject("sensor_sampling_status").getBoolean("active_sample_requested"))
        assertEquals("array", featureMatrix.getString("type"))
        assertEquals("Sensor analyzer feature 0", featureMatrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("Android SensorManager", featureMatrix.getJSONArray("items").getJSONObject(0).getString("feature_source"))
        assertEquals("sensor_snapshot", featureMatrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Route one-shot motion sample", routes.getJSONObject(0).getString("label"))
        assertEquals("Passive report default", policies.getJSONObject(0).getString("label"))
        assertEquals("sampling_cadence", policies.getJSONObject(0).getString("constraint_type"))
        assertEquals("IMU fusion source coverage", quality.getJSONObject(0).getString("label"))
        assertEquals("fusion_sources", quality.getJSONObject(0).getString("quality_signal"))
        assertEquals("motion_sensor_quality", quality.getJSONObject(0).getString("tool_action"))
    }

    @Test
    fun compactsSensorWorkflowAdvisorReportWithoutDroppingCandidates() {
        val advisorRows = JSONArray()
        repeat(14) { index ->
            advisorRows.put(
                JSONObject()
                    .put("category", "sensor_workflow_advisor")
                    .put("label", "Sensor advisor decision $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "decision $index")
                    .put("detail", "Sensor advisor row $index with accelerometer, gyroscope, sampling, and runtime evidence.")
                    .put("recommendation", "Use the matching sensor workflow route.")
                    .put("tool_action", "sensor_snapshot")
                    .put("graph_type", "sensor_workflow_advisor_matrix")
                    .put("fraction", 0.86),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "sensor_workflow_advisor_report")
            .put("sensor_workflow_advisor_count", 14)
            .put("ready_sensor_workflow_advisor_count", 7)
            .put("sensor_workflow_candidate_count", 1)
            .put("sensor_workflow_advisor_matrix", advisorRows)
            .put(
                "sensor_workflow_candidates",
                JSONArray().put(
                    JSONObject()
                        .put("category", "sensor_workflow_candidate")
                        .put("label", "Accelerometer")
                        .put("ready", true)
                        .put("sensor_type", "accelerometer")
                        .put("value_label", "available")
                        .put("candidate_score", 92)
                        .put("power_ma", 0.5)
                        .put("tool_action", "motion_sensor_quality"),
                ),
            )
            .put(
                "gemma_sensor_workflow_directives",
                JSONArray().put("Use sensor_workflow_advisor_matrix first."),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Sensor Workflow Advisor").put("body", "14 rows")))

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result.toString()))
        val matrix = parsed.getJSONObject("sensor_workflow_advisor_matrix")
        val candidates = parsed.getJSONArray("sensor_workflow_candidates")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(14, parsed.getInt("sensor_workflow_advisor_count"))
        assertEquals(7, parsed.getInt("ready_sensor_workflow_advisor_count"))
        assertEquals("array", matrix.getString("type"))
        assertEquals("Sensor advisor decision 0", matrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("sensor_snapshot", matrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("Accelerometer", candidates.getJSONObject(0).getString("label"))
        assertEquals("accelerometer", candidates.getJSONObject(0).getString("sensor_type"))
        assertEquals(92, candidates.getJSONObject(0).getInt("candidate_score"))
        assertEquals("Use sensor_workflow_advisor_matrix first.", parsed.getJSONArray("gemma_sensor_workflow_directives").getString(0))
    }

    @Test
    fun compactsMotionSensorDecisionPacketWithoutDroppingRoutesOrBoundaries() {
        val packetRows = JSONArray()
        repeat(12) { index ->
            packetRows.put(
                JSONObject()
                    .put("category", "motion_sensor_decision_packet")
                    .put("label", "Motion sensor packet $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "packet $index")
                    .put("detail", "Motion sensor decision packet $index with accelerometer, gyroscope, pose, sampling, and MediaTek context.")
                    .put("recommendation", "Keep motion claims bounded.")
                    .put("tool_action", "motion_sensor_decision_packet_report")
                    .put("graph_type", "motion_sensor_decision_packet")
                    .put("source_graph_type", "motion_sensor_quality")
                    .put("decision_status", "motion_context_available")
                    .put("claim_scope", "Android SensorManager motion metadata only")
                    .put("active_refresh_action", "motion_sensor_quality")
                    .put("passive_fallback_action", "sensor_analyzer_report")
                    .put("mediatek_sensitive", true)
                    .put("sensor_privacy_sensitive", true)
                    .put("fraction", 0.88),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "motion_sensor_decision_packet_report")
            .put("motion_sensor_decision_packet_count", 12)
            .put("ready_motion_sensor_decision_packet_count", 6)
            .put("motion_sensor_decision_packet", packetRows)
            .put(
                "motion_sensor_decision_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "motion_sensor_decision_route")
                        .put("label", "Open motion quality gates")
                        .put("ready", true)
                        .put("tool_action", "motion_sensor_quality")
                        .put("target_graph_type", "motion_sensor_quality"),
                ),
            )
            .put(
                "motion_sensor_claim_boundaries",
                JSONArray().put(
                    JSONObject()
                        .put("category", "motion_sensor_claim_boundary")
                        .put("label", "Android SensorManager boundary")
                        .put("ready", true)
                        .put("claim_scope", "Android-reported motion sensor metadata and bounded samples only")
                        .put("sensor_privacy_sensitive", true),
                ),
            )
            .put(
                "gemma_motion_sensor_decision_directives",
                JSONArray().put("Use motion_sensor_decision_packet first."),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Motion Sensor Decision").put("body", "12 rows")))

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result.toString()))
        val packet = parsed.getJSONObject("motion_sensor_decision_packet")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(12, parsed.getInt("motion_sensor_decision_packet_count"))
        assertEquals("array", packet.getString("type"))
        assertEquals("Motion sensor packet 0", packet.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("motion_context_available", packet.getJSONArray("items").getJSONObject(0).getString("decision_status"))
        assertTrue(packet.getJSONArray("items").getJSONObject(0).getBoolean("sensor_privacy_sensitive"))
        assertEquals("Open motion quality gates", parsed.getJSONArray("motion_sensor_decision_routes").getJSONObject(0).getString("label"))
        assertEquals("Android SensorManager boundary", parsed.getJSONArray("motion_sensor_claim_boundaries").getJSONObject(0).getString("label"))
        assertEquals("Use motion_sensor_decision_packet first.", parsed.getJSONArray("gemma_motion_sensor_decision_directives").getString(0))
    }

    @Test
    fun compactsRadioSignalAdvisorReportWithoutDroppingReceiverDecisions() {
        val advisorRows = JSONArray()
        repeat(14) { index ->
            advisorRows.put(
                JSONObject()
                    .put("category", "radio_signal_advisor")
                    .put("label", "Radio advisor decision $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "decision $index")
                    .put("detail", "Radio advisor row $index with receiver, AM/FM, SDR bridge, metadata, and safety-boundary evidence.")
                    .put("recommendation", "Use the matching receiver route.")
                    .put("tool_action", "radio_signal_graph")
                    .put("graph_type", "radio_signal_advisor_matrix")
                    .put("fraction", 0.88),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "radio_signal_advisor_report")
            .put("radio_signal_advisor_count", 14)
            .put("ready_radio_signal_advisor_count", 7)
            .put("radio_receiver_candidate_count", 1)
            .put("radio_signal_advisor_matrix", advisorRows)
            .put(
                "radio_receiver_candidates",
                JSONArray().put(
                    JSONObject()
                        .put("category", "radio_receiver_candidate")
                        .put("label", "FM station receiver profile")
                        .put("ready", true)
                        .put("value_label", "1 sample(s)")
                        .put("candidate_score", 92)
                        .put("receiver_id", "fm_vendor_or_sdr")
                        .put("sample_count", 1)
                        .put("ready_metadata_count", 1)
                        .put("tool_action", "radio_signal_graph"),
                ),
            )
            .put(
                "gemma_radio_advisor_directives",
                JSONArray().put("Use radio_signal_advisor_matrix first."),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Radio Signal Advisor").put("body", "14 rows")))

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result.toString()))
        val matrix = parsed.getJSONObject("radio_signal_advisor_matrix")
        val candidates = parsed.getJSONArray("radio_receiver_candidates")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(14, parsed.getInt("radio_signal_advisor_count"))
        assertEquals(7, parsed.getInt("ready_radio_signal_advisor_count"))
        assertEquals("array", matrix.getString("type"))
        assertEquals("Radio advisor decision 0", matrix.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("radio_signal_graph", matrix.getJSONArray("items").getJSONObject(0).getString("tool_action"))
        assertEquals("FM station receiver profile", candidates.getJSONObject(0).getString("label"))
        assertEquals("fm_vendor_or_sdr", candidates.getJSONObject(0).getString("receiver_id"))
        assertEquals(92, candidates.getJSONObject(0).getInt("candidate_score"))
        assertEquals("Use radio_signal_advisor_matrix first.", parsed.getJSONArray("gemma_radio_advisor_directives").getString(0))
    }

    @Test
    fun compactsRadioSignalDecisionPacketWithoutDroppingRoutesOrBoundaries() {
        val packetRows = JSONArray()
        repeat(12) { index ->
            packetRows.put(
                JSONObject()
                    .put("category", "radio_signal_decision_packet")
                    .put("label", "Radio decision packet $index")
                    .put("ready", index % 2 == 0)
                    .put("value_label", "packet $index")
                    .put("detail", "Radio signal decision packet $index with AM/FM, SDR, RF, and MediaTek context.")
                    .put("recommendation", "Keep radio claims bounded.")
                    .put("tool_action", "radio_signal_decision_packet_report")
                    .put("graph_type", "radio_signal_decision_packet")
                    .put("source_graph_type", "radio_signal_graph")
                    .put("decision_status", "am_fm_samples_available")
                    .put("claim_scope", "receiver-provided samples only")
                    .put("active_refresh_action", "radio_signal_graph")
                    .put("passive_fallback_action", "radio_signal_status")
                    .put("mediatek_sensitive", true)
                    .put("rf_coexistence_sensitive", true)
                    .put("fraction", 0.88),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "radio_signal_decision_packet_report")
            .put("radio_signal_decision_packet_count", 12)
            .put("ready_radio_signal_decision_packet_count", 6)
            .put("radio_signal_decision_packet", packetRows)
            .put(
                "radio_signal_decision_routes",
                JSONArray().put(
                    JSONObject()
                        .put("category", "radio_signal_decision_route")
                        .put("label", "AM/FM or SDR graph route")
                        .put("ready", true)
                        .put("tool_action", "radio_signal_graph")
                        .put("target_graph_type", "radio_signal_graph"),
                ),
            )
            .put(
                "radio_signal_claim_boundaries",
                JSONArray().put(
                    JSONObject()
                        .put("category", "radio_signal_claim_boundary")
                        .put("label", "Public Android radio boundary")
                        .put("ready", true)
                        .put("claim_scope", "public Android radio capability limits"),
                ),
            )
            .put(
                "gemma_radio_signal_decision_directives",
                JSONArray().put("Use radio_signal_decision_packet first."),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Radio Signal Decision").put("body", "12 rows")))

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result.toString()))
        val packet = parsed.getJSONObject("radio_signal_decision_packet")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(12, parsed.getInt("radio_signal_decision_packet_count"))
        assertEquals("array", packet.getString("type"))
        assertEquals("Radio decision packet 0", packet.getJSONArray("items").getJSONObject(0).getString("label"))
        assertEquals("am_fm_samples_available", packet.getJSONArray("items").getJSONObject(0).getString("decision_status"))
        assertTrue(packet.getJSONArray("items").getJSONObject(0).getBoolean("rf_coexistence_sensitive"))
        assertEquals("AM/FM or SDR graph route", parsed.getJSONArray("radio_signal_decision_routes").getJSONObject(0).getString("label"))
        assertEquals("Public Android radio boundary", parsed.getJSONArray("radio_signal_claim_boundaries").getJSONObject(0).getString("label"))
        assertEquals("Use radio_signal_decision_packet first.", parsed.getJSONArray("gemma_radio_signal_decision_directives").getString(0))
    }

    @Test
    fun compactsBluetoothAndRadioDiagnosticRowsWithoutDroppingSignalMetadata() {
        val devices = JSONArray()
        repeat(30) { index ->
            devices.put(
                JSONObject()
                    .put("device_name", "Beacon-$index")
                    .put("address", "AA:BB:CC:00:00:$index")
                    .put("rssi_dbm", -40 - index)
                    .put("device_type", "le")
                    .put("device_category", "wearable_health")
                    .put("service_uuids", JSONArray().put("0000180d-0000-1000-8000-00805f9b34fb"))
                    .put("manufacturer_ids", JSONArray().put("0x004C"))
                    .put("proximity_label", "near")
                    .put("scan_record", "ff".repeat(200)),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "bluetooth_scan")
            .put("bluetooth_metadata_count", 3)
            .put("bluetooth_service_uuid_count", 1)
            .put("bluetooth_manufacturer_id_count", 1)
            .put("bluetooth_devices", devices)
            .put(
                "bluetooth_metadata_summary",
                JSONArray().put(
                    JSONObject()
                        .put("summary_type", "manufacturer_id")
                        .put("label", "0x004C")
                        .put("count", 30)
                        .put("strongest_rssi_dbm", -40)
                        .put("recommendation", "Manufacturer data advertised nearby."),
                ),
            )
            .put("cards", JSONArray().put(JSONObject().put("title", "Bluetooth Nearby").put("body", "30 devices")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val compactedDevices = parsed.getJSONObject("bluetooth_devices")
        val metadataSummary = parsed.getJSONArray("bluetooth_metadata_summary")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(3, parsed.getInt("bluetooth_metadata_count"))
        assertEquals(1, parsed.getInt("bluetooth_service_uuid_count"))
        assertEquals(1, parsed.getInt("bluetooth_manufacturer_id_count"))
        assertEquals(30, compactedDevices.getInt("original_count"))
        assertEquals("Beacon-0", compactedDevices.getJSONArray("items").getJSONObject(0).getString("device_name"))
        assertEquals(-40, compactedDevices.getJSONArray("items").getJSONObject(0).getInt("rssi_dbm"))
        assertEquals("wearable_health", compactedDevices.getJSONArray("items").getJSONObject(0).getString("device_category"))
        assertEquals("near", compactedDevices.getJSONArray("items").getJSONObject(0).getString("proximity_label"))
        assertEquals("0x004C", metadataSummary.getJSONObject(0).getString("label"))
    }

    @Test
    fun compactsBluetoothDeviceDetailsWithoutDroppingExportEvidence() {
        val details = JSONArray()
        repeat(24) { index ->
            details.put(
                JSONObject()
                    .put("display_label", "Hermes Heart $index")
                    .put("device_name", "Heart Strap $index")
                    .put("advertised_name", "Hermes Heart $index")
                    .put("address", "AA:BB:CC:00:11:$index")
                    .put("device_type", "le")
                    .put("device_category", "wearable_health")
                    .put("bond_state", "bonded")
                    .put("semantic_label", "health or fitness device")
                    .put("rssi_dbm", -48 - index)
                    .put("proximity_label", "near")
                    .put("service_labels", JSONArray().put("Heart Rate"))
                    .put("service_uuids", JSONArray().put("0000180d-0000-1000-8000-00805f9b34fb"))
                    .put("manufacturer_names", JSONArray().put("Apple"))
                    .put("manufacturer_ids", JSONArray().put("0x004C"))
                    .put("metadata_completeness_score", 92)
                    .put("evidence_summary", "health or fitness device | services=Heart Rate | manufacturers=Apple")
                    .put("scan_record_bytes", 48),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "bluetooth_export")
            .put("bluetooth_device_detail_count", 24)
            .put("bluetooth_filtered_device_count", 24)
            .put("bluetooth_device_details", details)
            .put(
                "bluetooth_device_export",
                JSONObject()
                    .put("format", "both")
                    .put("row_count", 24)
                    .put("json_array_key", "bluetooth_device_details")
                    .put("csv_key", "bluetooth_device_export_csv")
                    .put("included_fields", JSONArray().put("display_label").put("metadata_completeness_score")),
            )
            .put("bluetooth_device_export_csv", "display_label,metadata_completeness_score\n" + (0 until 24).joinToString("\n") { "Hermes Heart $it,92" })
            .put("cards", JSONArray().put(JSONObject().put("title", "Bluetooth Device Details").put("body", "24 details")))
            .toString()

        val parsed = JSONObject(NativeToolContextCompressor.compactToolResult(result))
        val compactedDetails = parsed.getJSONObject("bluetooth_device_details")

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(24, parsed.getInt("bluetooth_device_detail_count"))
        assertEquals(24, parsed.getInt("bluetooth_filtered_device_count"))
        assertEquals("both", parsed.getJSONObject("bluetooth_device_export").getString("format"))
        assertEquals("Hermes Heart 0", compactedDetails.getJSONArray("items").getJSONObject(0).getString("display_label"))
        assertEquals(92, compactedDetails.getJSONArray("items").getJSONObject(0).getInt("metadata_completeness_score"))
        assertTrue(compactedDetails.getJSONArray("items").getJSONObject(0).getString("evidence_summary").contains("Heart Rate"))
        assertTrue(parsed.getString("bluetooth_device_export_csv").contains("Hermes Heart 0"))
    }

    @Test
    fun compactsSensorCapabilitiesWithoutDroppingHardwareMetadata() {
        val capabilities = JSONArray()
        repeat(20) { index ->
            capabilities.put(
                JSONObject()
                    .put("sensor_type", if (index % 2 == 0) "accelerometer" else "gyroscope")
                    .put("sensor_label", if (index % 2 == 0) "Accelerometer" else "Gyroscope")
                    .put("sensor_name", "Motion Sensor $index")
                    .put("vendor", "Vendor $index")
                    .put("available", true)
                    .put("unit", if (index % 2 == 0) "m/s^2" else "rad/s")
                    .put("maximum_range", 19.6 + index)
                    .put("resolution", 0.001)
                    .put("power_ma", 0.8)
                    .put("min_delay_us", 5000)
                    .put("max_delay_us", 200000)
                    .put("reporting_mode", "continuous")
                    .put("wake_up", index == 0)
                    .put("fifo_max_event_count", 512),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "sensor_snapshot")
            .put("sensor_capability_count", 20)
            .put("motion_sensor_count", 20)
            .put("wake_up_sensor_count", 1)
            .put("sensor_capabilities", capabilities)
            .put("cards", JSONArray().put(JSONObject().put("title", "Sensor Hardware").put("body", "20 sensors")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val compactedCapabilities = parsed.getJSONObject("sensor_capabilities")
        val first = compactedCapabilities.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(20, parsed.getInt("sensor_capability_count"))
        assertEquals(20, parsed.getInt("motion_sensor_count"))
        assertEquals(1, parsed.getInt("wake_up_sensor_count"))
        assertEquals(20, compactedCapabilities.getInt("original_count"))
        assertEquals("accelerometer", first.getString("sensor_type"))
        assertEquals("Motion Sensor 0", first.getString("sensor_name"))
        assertEquals(19.6, first.getDouble("maximum_range"), 0.01)
        assertEquals("continuous", first.getString("reporting_mode"))
        assertTrue(first.getBoolean("wake_up"))
    }

    @Test
    fun compactsMotionPoseEstimatesWithoutDroppingFusionMetadata() {
        val poses = JSONArray()
        repeat(12) { index ->
            poses.put(
                JSONObject()
                    .put("pose_type", if (index == 0) "device_pose" else "angular_motion")
                    .put("label", if (index == 0) "Device pose estimate" else "Angular motion state $index")
                    .put("value_label", if (index == 0) "face up | heading E" else "0.${index} rad/s steady")
                    .put("pose_source", if (index == 0) "accelerometer+magnetic_field" else "gyroscope")
                    .put("source_sensors", JSONArray().put("accelerometer").put("magnetic_field"))
                    .put("roll_degrees", 0.0)
                    .put("pitch_degrees", 0.0)
                    .put("tilt_degrees", 0.0)
                    .put("azimuth_degrees", 90.0)
                    .put("heading_label", "E")
                    .put("confidence_label", "high")
                    .put("workflow_hint", "Use for heading-aware workflows.")
                    .put("fraction", 0.9),
            )
        }
        val result = JSONObject()
            .put("success", true)
            .put("action", "sensor_snapshot")
            .put("motion_pose_estimate_count", 12)
            .put("motion_pose_estimates", poses)
            .put("cards", JSONArray().put(JSONObject().put("title", "Motion Pose Estimate").put("body", "12 poses")))
            .toString()

        val compacted = NativeToolContextCompressor.compactToolResult(result)
        val parsed = JSONObject(compacted)
        val compactedPoses = parsed.getJSONObject("motion_pose_estimates")
        val first = compactedPoses.getJSONArray("items").getJSONObject(0)

        assertTrue(parsed.getBoolean("_hermes_context_compressed"))
        assertEquals(12, parsed.getInt("motion_pose_estimate_count"))
        assertEquals(12, compactedPoses.getInt("original_count"))
        assertEquals("device_pose", first.getString("pose_type"))
        assertEquals("accelerometer+magnetic_field", first.getString("pose_source"))
        assertEquals("E", first.getString("heading_label"))
        assertEquals("high", first.getString("confidence_label"))
        assertTrue(first.getJSONArray("source_sensors").toString().contains("magnetic_field"))
    }

    @Test
    fun compactsCompletedNativeToolRoundsButKeepsLatestAssistantBlock() {
        val messages = JSONArray()
            .put(JSONObject().put("role", "system").put("content", "tools enabled"))
            .put(JSONObject().put("role", "user").put("content", "Scroll TikTok, reply to DMs, then draft email"))
        repeat(6) { index ->
            messages
                .put(
                    assistantToolCall(
                        id = "call_$index",
                        name = "android_ui_tool",
                    ),
                )
                .put(
                    JSONObject()
                        .put("role", "tool")
                        .put("tool_call_id", "call_$index")
                        .put("name", "android_ui_tool")
                        .put(
                            "content",
                            NativeToolContextCompressor.compactToolResult(
                                "Scrolled feed $index\n" + "visible row $index ".repeat(1_800),
                            ),
                        ),
                )
        }

        val compacted = NativeToolContextCompressor.compactMessages(messages)

        assertTrue(compacted.length() < messages.length())
        assertEquals("system", compacted.getJSONObject(0).getString("role"))
        assertEquals("user", compacted.getJSONObject(1).getString("role"))
        assertTrue(compacted.getJSONObject(2).getString("content").contains("compacted prior native tool context"))
        val latestAssistant = compacted.getJSONObject(compacted.length() - 2)
        assertEquals("assistant", latestAssistant.getString("role"))
        assertEquals("call_5", latestAssistant.getJSONArray("tool_calls").getJSONObject(0).getString("id"))
        assertEquals("tool", compacted.getJSONObject(compacted.length() - 1).getString("role"))
    }

    private fun assistantToolCall(id: String, name: String): JSONObject {
        return JSONObject()
            .put("role", "assistant")
            .put("content", JSONObject.NULL)
            .put(
                "tool_calls",
                JSONArray()
                    .put(
                        JSONObject()
                            .put("id", id)
                            .put("type", "function")
                            .put(
                                "function",
                                JSONObject()
                                    .put("name", name)
                                    .put("arguments", JSONObject().put("action", "scroll_forward").toString()),
                            ),
                    ),
            )
    }
}
