package com.mobilefork.hermesagent.settings

import android.app.Application
import android.content.Context
import com.mobilefork.hermesagent.data.McpConfigurationMode
import com.mobilefork.hermesagent.data.McpPromptCacheResendPolicy
import com.mobilefork.hermesagent.data.McpSettingsMessages
import com.mobilefork.hermesagent.data.McpSettingsStore
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(application = android.app.Application::class)
class McpSettingsStoreTest {
    @Test
    fun simpleAutoSetupPersistsConfigAndReloadStatus() {
        val store = freshStore()

        val result = store.autoSetupSimpleConfiguration(nowEpochMs = 1234L)

        assertTrue(result.success)
        assertEquals(1, result.serverCount)
        assertEquals(1234L, result.lastReloadEpochMs)
        assertTrue(result.statusMessage.contains("Reloaded 1 MCP server"))
        assertTrue(File(store.configFilePath()).isFile)

        val reloaded = store.load()
        assertEquals(McpConfigurationMode.SIMPLE, reloaded.mode)
        assertEquals(1234L, reloaded.lastReloadEpochMs)
        assertTrue(JSONObject(reloaded.configText).getJSONObject("mcpServers").has("hermes-native-tools"))
    }

    @Test
    fun advancedModeSavesValidJsonAndIgnoresDisabledServersInReloadCount() {
        val store = freshStore()
        val advancedConfig = """
            {
              "mcpServers": {
                "context7": { "command": "context7", "enabled": true },
                "deepwiki": { "command": "deepwiki", "enabled": false },
                "globalping": { "command": "globalping" }
              }
            }
        """.trimIndent()

        val result = store.saveAdvancedConfigTextAndReload(advancedConfig, nowEpochMs = 2000L)

        assertTrue(result.success)
        assertEquals(2, result.serverCount)
        assertTrue(result.statusMessage.contains("Reloaded 2 MCP server definitions"))

        val reloaded = store.load()
        assertEquals(McpConfigurationMode.ADVANCED, reloaded.mode)
        assertEquals(2000L, reloaded.lastReloadEpochMs)
        assertTrue(reloaded.configText.contains("context7"))
        assertTrue(reloaded.configText.contains("deepwiki"))
    }

    @Test
    fun advancedModeRejectsInvalidJsonWithoutOverwritingLastGoodConfig() {
        val store = freshStore()
        val good = """{"mcpServers":{"context7":{"command":"context7"}}}"""
        val saved = store.saveAdvancedConfigTextAndReload(good, nowEpochMs = 3000L)
        assertTrue(saved.success)

        val invalid = store.saveAdvancedConfigTextAndReload("""{"mcpServers":""", nowEpochMs = 4000L)

        assertFalse(invalid.success)
        assertTrue(invalid.statusMessage.contains("invalid"))
        val reloaded = store.load()
        assertEquals(3000L, reloaded.lastReloadEpochMs)
        assertTrue(reloaded.configText.contains("context7"))
        assertFalse(reloaded.configText.trim() == """{"mcpServers":""")
    }

    @Test
    fun providerCacheResendTogglePersistsAndPolicyBlocksDisallowedProviders() {
        val store = freshStore()

        assertFalse(McpPromptCacheResendPolicy.shouldResendCachedContext("openrouter", store.load()))

        val enabled = store.saveProviderPromptCacheResendEnabled(true)
        assertTrue(enabled.providerPromptCacheResendEnabled)
        assertTrue(McpPromptCacheResendPolicy.shouldResendCachedContext("openrouter", enabled))
        assertFalse(McpPromptCacheResendPolicy.shouldResendCachedContext("litert_lm", enabled))
        assertTrue(McpPromptCacheResendPolicy.statusFor("litert_lm", enabled).contains("disallows"))

        val disabled = store.saveProviderPromptCacheResendEnabled(false)
        assertFalse(disabled.providerPromptCacheResendEnabled)
        assertFalse(McpPromptCacheResendPolicy.shouldResendCachedContext("openrouter", disabled))
        assertEquals(McpSettingsMessages.CACHE_RESEND_DISABLED, disabled.lastStatusMessage)
    }

    @Test
    fun detectExistingConfigurationReportsMissingFileWithoutCreatingIt() {
        val store = freshStore()

        val result = store.detectExistingConfiguration()

        assertFalse(result.success)
        assertTrue(result.statusMessage.contains("No MCP config file found"))
        assertFalse(File(store.configFilePath()).exists())
    }

    private fun freshStore(): McpSettingsStore {
        val application = RuntimeEnvironment.getApplication() as Application
        application
            .getSharedPreferences("hermes_android_mcp_settings", Context.MODE_PRIVATE)
            .edit()
            .clear()
            .commit()
        val store = McpSettingsStore(application)
        File(store.configFilePath()).delete()
        return store
    }
}
