package com.mobilefork.hermesagent.ui.settings

import android.app.Application
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilefork.hermesagent.data.McpConfigurationMode
import com.mobilefork.hermesagent.data.McpPromptCacheResendPolicy
import com.mobilefork.hermesagent.data.McpSettings
import com.mobilefork.hermesagent.data.McpSettingsMessages
import com.mobilefork.hermesagent.data.McpSettingsStore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class McpSettingsUiState(
    val mode: McpConfigurationMode = McpConfigurationMode.SIMPLE,
    val configText: String = "",
    val providerPromptCacheResendEnabled: Boolean = false,
    val statusMessage: String = McpSettingsMessages.SIMPLE_READY,
    val configFilePath: String = "",
    val lastReloadEpochMs: Long = 0L,
)

class McpSettingsViewModel(application: Application) : AndroidViewModel(application) {
    private val store = McpSettingsStore(application)
    private val _uiState = MutableStateFlow(store.load().toUiState(store.configFilePath()))
    val uiState: StateFlow<McpSettingsUiState> = _uiState.asStateFlow()

    fun reloadFromDisk() {
        _uiState.value = store.load().toUiState(store.configFilePath())
    }

    fun selectMode(mode: McpConfigurationMode) {
        _uiState.value = store.saveMode(mode).toUiState(store.configFilePath())
    }

    fun detectExistingConfiguration() {
        val result = store.detectExistingConfiguration()
        _uiState.update {
            it.copy(
                configText = result.configText,
                statusMessage = result.statusMessage,
                lastReloadEpochMs = result.lastReloadEpochMs.takeIf { value -> value > 0L } ?: it.lastReloadEpochMs,
            )
        }
    }

    fun autoFillSimpleConfiguration() {
        val result = store.autoFillSimpleConfiguration()
        _uiState.update {
            it.copy(
                mode = McpConfigurationMode.SIMPLE,
                configText = result.configText,
                statusMessage = result.statusMessage,
            )
        }
    }

    fun autoSetupSimpleConfiguration() {
        val result = store.autoSetupSimpleConfiguration()
        _uiState.update {
            it.copy(
                mode = McpConfigurationMode.SIMPLE,
                configText = result.configText,
                statusMessage = result.statusMessage,
                lastReloadEpochMs = result.lastReloadEpochMs,
            )
        }
    }

    fun updateAdvancedConfigText(value: String) {
        _uiState.update { it.copy(configText = value) }
    }

    fun saveAdvancedConfigAndReload() {
        val result = store.saveAdvancedConfigTextAndReload(_uiState.value.configText)
        _uiState.update {
            it.copy(
                mode = McpConfigurationMode.ADVANCED,
                configText = result.configText,
                statusMessage = result.statusMessage,
                lastReloadEpochMs = result.lastReloadEpochMs.takeIf { value -> value > 0L } ?: it.lastReloadEpochMs,
            )
        }
    }

    fun reloadServers() {
        val result = store.reloadServers()
        _uiState.update {
            it.copy(
                configText = result.configText,
                statusMessage = result.statusMessage,
                lastReloadEpochMs = result.lastReloadEpochMs.takeIf { value -> value > 0L } ?: it.lastReloadEpochMs,
            )
        }
    }

    fun updateProviderPromptCacheResend(enabled: Boolean, providerId: String = "") {
        val updated = store.saveProviderPromptCacheResendEnabled(enabled)
        _uiState.value = updated.toUiState(
            configFilePath = store.configFilePath(),
            statusOverride = McpPromptCacheResendPolicy.statusFor(providerId, updated),
        )
    }
}

@Composable
fun McpSettingsSection(
    modifier: Modifier = Modifier,
    selectedProviderId: String = "",
    viewModel: McpSettingsViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsState()
    McpSettingsCard(
        modifier = modifier,
        uiState = uiState,
        selectedProviderId = selectedProviderId,
        onModeChange = viewModel::selectMode,
        onDetect = viewModel::detectExistingConfiguration,
        onAutoFill = viewModel::autoFillSimpleConfiguration,
        onAutoSetup = viewModel::autoSetupSimpleConfiguration,
        onConfigTextChange = viewModel::updateAdvancedConfigText,
        onSaveAdvanced = viewModel::saveAdvancedConfigAndReload,
        onReloadServers = viewModel::reloadServers,
        onProviderPromptCacheResendChange = viewModel::updateProviderPromptCacheResend,
    )
}

@Composable
fun McpSettingsCard(
    uiState: McpSettingsUiState,
    selectedProviderId: String,
    onModeChange: (McpConfigurationMode) -> Unit,
    onDetect: () -> Unit,
    onAutoFill: () -> Unit,
    onAutoSetup: () -> Unit,
    onConfigTextChange: (String) -> Unit,
    onSaveAdvanced: () -> Unit,
    onReloadServers: () -> Unit,
    onProviderPromptCacheResendChange: (Boolean, String) -> Unit,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        tonalElevation = 2.dp,
        shape = MaterialTheme.shapes.medium,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("MCP configuration", style = MaterialTheme.typography.titleMedium)
            Text(
                "Simple mode auto-detects and writes a safe native-tools config. Advanced mode edits the raw MCP JSON file.",
                style = MaterialTheme.typography.bodySmall,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    modifier = Modifier.testTag("McpSimpleModeButton"),
                    onClick = { onModeChange(McpConfigurationMode.SIMPLE) },
                    enabled = uiState.mode != McpConfigurationMode.SIMPLE,
                ) {
                    Text("Simple")
                }
                Button(
                    modifier = Modifier.testTag("McpAdvancedModeButton"),
                    onClick = { onModeChange(McpConfigurationMode.ADVANCED) },
                    enabled = uiState.mode != McpConfigurationMode.ADVANCED,
                ) {
                    Text("Advanced")
                }
            }
            Text(
                "Config file: ${uiState.configFilePath}",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.testTag("McpConfigFilePath"),
            )
            if (uiState.mode == McpConfigurationMode.SIMPLE) {
                SimpleMcpOnboardingControls(
                    configText = uiState.configText,
                    onDetect = onDetect,
                    onAutoFill = onAutoFill,
                    onAutoSetup = onAutoSetup,
                )
            } else {
                AdvancedMcpConfigEditor(
                    configText = uiState.configText,
                    onConfigTextChange = onConfigTextChange,
                    onSaveAdvanced = onSaveAdvanced,
                    onReloadServers = onReloadServers,
                )
            }
            ProviderPromptCacheControls(
                enabled = uiState.providerPromptCacheResendEnabled,
                selectedProviderId = selectedProviderId,
                onChange = onProviderPromptCacheResendChange,
            )
            Text(
                uiState.statusMessage,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.testTag("McpStatusMessage"),
            )
        }
    }
}

@Composable
private fun SimpleMcpOnboardingControls(
    configText: String,
    onDetect: () -> Unit,
    onAutoFill: () -> Unit,
    onAutoSetup: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(
                modifier = Modifier.testTag("McpAutoDetectButton"),
                onClick = onDetect,
            ) {
                Text("Auto detect")
            }
            Button(
                modifier = Modifier.testTag("McpAutoFillButton"),
                onClick = onAutoFill,
            ) {
                Text("Auto fill")
            }
            Button(
                modifier = Modifier.testTag("McpAutoSetupButton"),
                onClick = onAutoSetup,
            ) {
                Text("Auto setup")
            }
        }
        OutlinedTextField(
            value = configText,
            onValueChange = {},
            modifier = Modifier
                .fillMaxWidth()
                .testTag("McpSimpleConfigPreview"),
            label = { Text("Preview") },
            minLines = 4,
            maxLines = 10,
            readOnly = true,
        )
    }
}

@Composable
private fun AdvancedMcpConfigEditor(
    configText: String,
    onConfigTextChange: (String) -> Unit,
    onSaveAdvanced: () -> Unit,
    onReloadServers: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(
                modifier = Modifier.testTag("McpSaveAdvancedButton"),
                onClick = onSaveAdvanced,
            ) {
                Text("Save and reload")
            }
            Button(
                modifier = Modifier.testTag("McpReloadServersButton"),
                onClick = onReloadServers,
            ) {
                Text("Reload servers")
            }
        }
        OutlinedTextField(
            value = configText,
            onValueChange = onConfigTextChange,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("McpAdvancedConfigText"),
            label = { Text("MCP config JSON") },
            minLines = 8,
            maxLines = 18,
        )
    }
}

@Composable
private fun ProviderPromptCacheControls(
    enabled: Boolean,
    selectedProviderId: String,
    onChange: (Boolean, String) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text("Provider cache resend", style = MaterialTheme.typography.titleSmall)
            Text(
                "When enabled, Hermes may resend stable prior/tool-output context for provider input-token caching. When disabled, cached context resend is blocked.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
        Switch(
            modifier = Modifier.testTag("McpProviderCacheResendSwitch"),
            checked = enabled,
            onCheckedChange = { onChange(it, selectedProviderId) },
        )
    }
}

private fun McpSettings.toUiState(
    configFilePath: String,
    statusOverride: String? = null,
): McpSettingsUiState {
    return McpSettingsUiState(
        mode = mode,
        configText = configText,
        providerPromptCacheResendEnabled = providerPromptCacheResendEnabled,
        statusMessage = statusOverride ?: lastStatusMessage,
        configFilePath = configFilePath,
        lastReloadEpochMs = lastReloadEpochMs,
    )
}
