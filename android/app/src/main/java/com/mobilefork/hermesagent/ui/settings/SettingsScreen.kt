@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.mobilefork.hermesagent.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.text.KeyboardOptions
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilefork.hermesagent.api.HermesEndpointUrl
import com.mobilefork.hermesagent.backend.BackendKind
import com.mobilefork.hermesagent.data.AppSettings
import com.mobilefork.hermesagent.data.ProviderPresets
import com.mobilefork.hermesagent.ui.i18n.AppLanguage
import com.mobilefork.hermesagent.ui.i18n.LocalHermesStrings
import com.mobilefork.hermesagent.ui.shell.ShellActionItem

@Composable
fun SettingsScreen(
    modifier: Modifier = Modifier,
    viewModel: SettingsViewModel = viewModel(),
    extraBottomSpacing: Dp = 0.dp,
    onContextActionsChanged: (List<ShellActionItem>) -> Unit = {},
) {
    val uiState by viewModel.uiState.collectAsState()
    val strings = LocalHermesStrings.current
    val scrollState = rememberScrollState()
    val selectedPreset = ProviderPresets.find(uiState.provider)
    val selectedProviderLabel = strings.providerDisplayLabel(
        uiState.provider,
        selectedPreset?.label ?: uiState.provider,
    )

    SideEffect {
        onContextActionsChanged(emptyList())
    }

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .widthIn(max = 920.dp)
                        .verticalScroll(scrollState)
                        .imePadding()
                        .padding(horizontal = 16.dp, vertical = 12.dp)
                        .padding(bottom = extraBottomSpacing),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    SettingsHelpCard(
                        providerLabel = selectedProviderLabel,
                        strings = strings,
                    )
                    LanguagePickerCard(
                        currentLanguageTag = uiState.languageTag,
                        onSelectLanguage = viewModel::selectLanguage,
                        strings = strings,
                    )
                    AgentPersonaCard(
                        customSystemPrompt = uiState.customSystemPrompt,
                        onPromptChange = viewModel::updateCustomSystemPrompt,
                        onSave = viewModel::saveAgentPersona,
                        onClear = viewModel::clearAgentPersona,
                        strings = strings,
                    )
                    AppearanceCard(
                        chatDisplayMode = uiState.chatDisplayMode,
                        keywordHighlightingEnabled = uiState.keywordHighlightingEnabled,
                        themePrimaryHex = uiState.themePrimaryHex,
                        themeSecondaryHex = uiState.themeSecondaryHex,
                        themeBackgroundHex = uiState.themeBackgroundHex,
                        themeSurfaceHex = uiState.themeSurfaceHex,
                        themeSurfaceVariantHex = uiState.themeSurfaceVariantHex,
                        themeCardShape = uiState.themeCardShape,
                        onChatDisplayModeChange = viewModel::updateChatDisplayMode,
                        onKeywordHighlightingChange = viewModel::updateKeywordHighlighting,
                        onPrimaryHexChange = viewModel::updateThemePrimaryHex,
                        onSecondaryHexChange = viewModel::updateThemeSecondaryHex,
                        onBackgroundHexChange = viewModel::updateThemeBackgroundHex,
                        onSurfaceHexChange = viewModel::updateThemeSurfaceHex,
                        onSurfaceVariantHexChange = viewModel::updateThemeSurfaceVariantHex,
                        onCardShapeChange = viewModel::updateThemeCardShape,
                        onApplyPreset = viewModel::applyThemePreset,
                        onSaveAppearance = viewModel::saveAppearance,
                        strings = strings,
                    )
                    OnDeviceInferenceCard(
                        onDeviceBackend = uiState.onDeviceBackend,
                        speculativeDecodingMode = uiState.liteRtLmSpeculativeDecodingMode,
                        onSelectBackend = viewModel::updateOnDeviceBackend,
                        onSelectSpeculativeDecodingMode = viewModel::updateLiteRtLmSpeculativeDecodingMode,
                        onStartRuntime = viewModel::startLocalRuntimeForFlavor,
                        summary = uiState.onDeviceSummary,
                        strings = strings,
                    )
                    OfflineAirplaneCard(
                        enabled = uiState.offlineAirplaneMode,
                        onChange = viewModel::updateOfflineAirplaneMode,
                        strings = strings,
                    )
                    LocalModelDownloadsSection(
                        dataSaverMode = uiState.dataSaverMode,
                        offlineAirplaneMode = uiState.offlineAirplaneMode,
                        onDataSaverModeChange = viewModel::updateDataSaverMode,
                        selectedBackend = uiState.onDeviceBackend,
                        onRuntimeFlavorSelected = viewModel::syncOnDeviceBackendWithRuntimeFlavor,
                        onCompletedDownloadReady = viewModel::startLocalRuntimeForFlavor,
                    )

                    RemoteFallbackCard(
                        providerId = uiState.provider,
                        providerLabel = selectedProviderLabel,
                        baseUrl = uiState.baseUrl,
                        model = uiState.model,
                        apiKey = uiState.apiKey,
                        status = uiState.status,
                        onSelectProvider = viewModel::updateProvider,
                        onBaseUrlChange = viewModel::updateBaseUrl,
                        onModelChange = viewModel::updateModel,
                        onApiKeyChange = viewModel::updateApiKey,
                        onOpenProviderKeyPage = viewModel::openProviderKeyPage,
                        onCopyProviderKeyPage = viewModel::copyProviderKeyPage,
                        onCheckProviderKeyPage = viewModel::checkProviderKeyPage,
                        onImportProviderCredential = viewModel::importSavedProviderCredential,
                        onSave = viewModel::save,
                        strings = strings,
                    )

                    ToolProfileCard()

                    if (uiState.status.isNotBlank()) {
                        Text(uiState.status)
                    }
                }
            }
        }
    }
}

@Composable
private fun AgentPersonaCard(
    customSystemPrompt: String,
    onPromptChange: (String) -> Unit,
    onSave: () -> Unit,
    onClear: () -> Unit,
    strings: com.mobilefork.hermesagent.ui.i18n.HermesStrings,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
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
            Text(strings.agentPersonaTitle(), style = MaterialTheme.typography.titleMedium)
            OutlinedTextField(
                value = customSystemPrompt,
                onValueChange = onPromptChange,
                label = { Text(strings.customSystemPromptLabel()) },
                placeholder = { Text(strings.customSystemPromptPlaceholder()) },
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("AgentPersonaPrompt"),
                minLines = 3,
                maxLines = 8,
            )
            Text(
                strings.characterCount(customSystemPrompt.length, AppSettings.MAX_CUSTOM_SYSTEM_PROMPT_CHARS),
                style = MaterialTheme.typography.bodySmall,
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(
                    modifier = Modifier.testTag("SaveAgentPersonaButton"),
                    onClick = onSave,
                ) {
                    Text(strings.savePersonaLabel())
                }
                Button(
                    modifier = Modifier.testTag("ClearAgentPersonaButton"),
                    onClick = onClear,
                    enabled = customSystemPrompt.isNotBlank(),
                ) {
                    Text(strings.clearLabel())
                }
            }
        }
    }
}

@Composable
private fun AppearanceCard(
    chatDisplayMode: String,
    keywordHighlightingEnabled: Boolean,
    themePrimaryHex: String,
    themeSecondaryHex: String,
    themeBackgroundHex: String,
    themeSurfaceHex: String,
    themeSurfaceVariantHex: String,
    themeCardShape: String,
    onChatDisplayModeChange: (String) -> Unit,
    onKeywordHighlightingChange: (Boolean) -> Unit,
    onPrimaryHexChange: (String) -> Unit,
    onSecondaryHexChange: (String) -> Unit,
    onBackgroundHexChange: (String) -> Unit,
    onSurfaceHexChange: (String) -> Unit,
    onSurfaceVariantHexChange: (String) -> Unit,
    onCardShapeChange: (String) -> Unit,
    onApplyPreset: (AppearanceThemePreset) -> Unit,
    onSaveAppearance: () -> Unit,
    strings: com.mobilefork.hermesagent.ui.i18n.HermesStrings,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
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
            Text(strings.appearanceTitle(), style = MaterialTheme.typography.titleMedium)
            Text(
                strings.appearanceDescription(),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(strings.chatDisplayLabel(), style = MaterialTheme.typography.titleSmall)
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(
                    modifier = Modifier.testTag("ChatDisplayCompact"),
                    onClick = { onChatDisplayModeChange("compact") },
                    enabled = chatDisplayMode != "compact",
                ) {
                    Text(strings.compactModeLabel())
                }
                Button(
                    modifier = Modifier.testTag("ChatDisplayExpanded"),
                    onClick = { onChatDisplayModeChange("expanded") },
                    enabled = chatDisplayMode != "expanded",
                ) {
                    Text(strings.expandedModeLabel())
                }
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(strings.keywordHighlightingTitle(), style = MaterialTheme.typography.titleSmall)
                    Text(strings.keywordHighlightingDescription(), style = MaterialTheme.typography.bodySmall)
                }
                Switch(checked = keywordHighlightingEnabled, onCheckedChange = onKeywordHighlightingChange)
            }
            Text(strings.colourPresetsTitle(), style = MaterialTheme.typography.titleSmall)
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                appearanceThemePresets.forEach { preset ->
                    Button(onClick = { onApplyPreset(preset) }) {
                        Text(strings.appearancePresetLabel(preset.id, preset.label))
                    }
                }
            }
            OutlinedTextField(
                value = themePrimaryHex,
                onValueChange = onPrimaryHexChange,
                label = { Text(strings.accentHexLabel()) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            OutlinedTextField(
                value = themeSecondaryHex,
                onValueChange = onSecondaryHexChange,
                label = { Text(strings.secondaryAccentHexLabel()) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            OutlinedTextField(
                value = themeBackgroundHex,
                onValueChange = onBackgroundHexChange,
                label = { Text(strings.backgroundHexLabel()) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            OutlinedTextField(
                value = themeSurfaceHex,
                onValueChange = onSurfaceHexChange,
                label = { Text(strings.composerSurfaceHexLabel()) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            OutlinedTextField(
                value = themeSurfaceVariantHex,
                onValueChange = onSurfaceVariantHexChange,
                label = { Text(strings.assistantPanelHexLabel()) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            Text(strings.cardsAndBoxesTitle(), style = MaterialTheme.typography.titleSmall)
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                listOf("rounded", "soft", "square").forEach { shape ->
                    Button(
                        modifier = Modifier.testTag("CardShape-$shape"),
                        onClick = { onCardShapeChange(shape) },
                        enabled = themeCardShape != shape,
                    ) {
                        Text(strings.cardShapeLabel(shape))
                    }
                }
            }
            Button(
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("SaveAppearanceButton"),
                onClick = onSaveAppearance,
            ) {
                Text(strings.saveAppearanceLabel())
            }
        }
    }
}

@Composable
private fun OfflineAirplaneCard(
    enabled: Boolean,
    onChange: (Boolean) -> Unit,
    strings: com.mobilefork.hermesagent.ui.i18n.HermesStrings,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
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
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(strings.offlineAirplaneModeTitle(), style = MaterialTheme.typography.titleMedium)
                    Text(
                        strings.offlineAirplaneModeDescription(),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Switch(checked = enabled, onCheckedChange = onChange)
            }
            Button(onClick = { onChange(!enabled) }) {
                Text(strings.offlineAirplaneToggleLabel(enabled))
            }
        }
    }
}

@Composable
private fun SettingsHelpCard(
    providerLabel: String,
    strings: com.mobilefork.hermesagent.ui.i18n.HermesStrings,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        tonalElevation = 2.dp,
        shape = MaterialTheme.shapes.medium,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            // Text("New here?")
            Text(strings.settingsNewHereTitle.ifBlank { "New here?" }, style = MaterialTheme.typography.titleMedium)
            Text(
                strings.forkDisclosure(),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(strings.settingsHelpStart)
            // Accounts keeps app sign-in separate from provider key setup.
            Text(strings.settingsHelpAccounts)
            Text(strings.currentProviderProfile(providerLabel))
        }
    }
}

@Composable
private fun RemoteFallbackCard(
    providerId: String,
    providerLabel: String,
    baseUrl: String,
    model: String,
    apiKey: String,
    status: String,
    onSelectProvider: (String) -> Unit,
    onBaseUrlChange: (String) -> Unit,
    onModelChange: (String) -> Unit,
    onApiKeyChange: (String) -> Unit,
    onOpenProviderKeyPage: (String, String) -> Unit,
    onCopyProviderKeyPage: (String, String) -> Unit,
    onCheckProviderKeyPage: (String, String) -> Unit,
    onImportProviderCredential: () -> Unit,
    onSave: () -> Unit,
    strings: com.mobilefork.hermesagent.ui.i18n.HermesStrings,
) {
    val providerPreset = ProviderPresets.find(providerId)
    val customEndpointPreview = if (providerId == "custom") {
        runCatching { HermesEndpointUrl.chatCompletionsUrl(baseUrl) }.getOrNull()
    } else {
        null
    }
    Surface(
        modifier = Modifier.fillMaxWidth(),
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
            Text(strings.remoteFallbackTitle(), style = MaterialTheme.typography.titleMedium)
            Text(strings.remoteFallbackDescription(), style = MaterialTheme.typography.bodySmall)
            Text(strings.providerLabel(), style = MaterialTheme.typography.titleSmall)
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                ProviderPresets.androidSettingsDefaults.forEach { preset ->
                    Button(
                        onClick = { onSelectProvider(preset.id) },
                        enabled = preset.id != providerId,
                    ) {
                        Text(strings.providerDisplayLabel(preset.id, preset.label))
                    }
                }
            }
            Text(strings.currentProviderProfile(providerLabel), style = MaterialTheme.typography.bodySmall)
            providerPreset?.apiKeyUrl?.takeIf { it.isNotBlank() }?.let { apiKeyUrl ->
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Button(onClick = { onOpenProviderKeyPage(providerId, apiKeyUrl) }) {
                        Text(strings.openProviderKeyPage(providerLabel))
                    }
                    Button(onClick = { onCopyProviderKeyPage(providerId, apiKeyUrl) }) {
                        Text(strings.copyProviderSetupUrl())
                    }
                    Button(onClick = { onCheckProviderKeyPage(providerId, apiKeyUrl) }) {
                        Text(strings.checkProviderSetupUrl())
                    }
                }
            }
            Button(onClick = onImportProviderCredential) {
                Text(strings.importSavedProviderCredential())
            }
            if (status.isNotBlank()) {
                Text(status, style = MaterialTheme.typography.bodySmall)
            }
            OutlinedTextField(
                value = baseUrl,
                onValueChange = onBaseUrlChange,
                label = { Text(strings.baseUrlLabel()) },
                modifier = Modifier.fillMaxWidth(),
            )
            if (providerId == "custom") {
                Text(
                    strings.customEndpointConnectionHint(),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                customEndpointPreview?.let { preview ->
                    Text(
                        text = strings.customEndpointPreview(preview),
                        modifier = Modifier.testTag("HermesEndpointDebugPreview"),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }
            providerPreset?.modelHint?.takeIf { it.isNotBlank() }?.let { modelHint ->
                Button(onClick = { onModelChange(modelHint) }) {
                    Text(strings.suggestedModelSummary(modelHint))
                }
            }
            OutlinedTextField(
                value = model,
                onValueChange = onModelChange,
                label = { Text(strings.modelLabel()) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            OutlinedTextField(
                value = apiKey,
                onValueChange = onApiKeyChange,
                label = { Text(strings.apiKeyLabel()) },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
            )
            Text(strings.apiKeyHelp(), style = MaterialTheme.typography.bodySmall)
            Text(
                strings.providerCredentialInputHelp(ProviderPresets.apiKeyEnvVars(providerId)),
                style = MaterialTheme.typography.bodySmall,
            )
            Button(onClick = onSave) {
                Text(strings.saveLabel())
            }
        }
    }
}

@Composable
private fun OnDeviceInferenceCard(
    onDeviceBackend: String,
    speculativeDecodingMode: String,
    onSelectBackend: (String) -> Unit,
    onSelectSpeculativeDecodingMode: (String) -> Unit,
    onStartRuntime: (String) -> Unit,
    summary: String,
    strings: com.mobilefork.hermesagent.ui.i18n.HermesStrings,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
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
            Text(strings.onDeviceInferenceTitle.ifBlank { "On-device inference" }, style = MaterialTheme.typography.titleMedium)
            Text(strings.onDeviceInferenceDescription, style = MaterialTheme.typography.bodySmall)
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(
                    onClick = {
                        onSelectBackend(BackendKind.LLAMA_CPP.persistedValue)
                        onStartRuntime("GGUF")
                    },
                    enabled = onDeviceBackend != BackendKind.LLAMA_CPP.persistedValue,
                ) {
                    Text(strings.llamaCppLabel.ifBlank { "llama.cpp (GGUF)" })
                }
                Button(
                    onClick = {
                        onSelectBackend(BackendKind.LITERT_LM.persistedValue)
                        onStartRuntime("LiteRT-LM")
                    },
                    enabled = onDeviceBackend != BackendKind.LITERT_LM.persistedValue,
                ) {
                    Text(strings.liteRtLmLabel.ifBlank { "LiteRT-LM" })
                }
                Button(
                    onClick = { onSelectBackend(BackendKind.NONE.persistedValue) },
                    enabled = onDeviceBackend != BackendKind.NONE.persistedValue,
                ) {
                    Text(strings.remoteOnly())
                }
            }
            Text(strings.llamaCppDescription, style = MaterialTheme.typography.bodySmall)
            Text(strings.liteRtLmDescription, style = MaterialTheme.typography.bodySmall)
            Text(strings.gemma4MtpTitle(), style = MaterialTheme.typography.titleSmall)
            Text(strings.gemma4MtpDescription(), style = MaterialTheme.typography.bodySmall)
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                speculativeDecodingChoices().forEach { choice ->
                    Button(
                        modifier = Modifier.testTag("LiteRtLmMtpMode-${choice.value}"),
                        onClick = { onSelectSpeculativeDecodingMode(choice.value) },
                        enabled = speculativeDecodingMode != choice.value,
                    ) {
                        Text(choice.label(strings))
                    }
                }
            }
            Text(localizedOnDeviceSummary(summary, strings), style = MaterialTheme.typography.bodySmall)
        }
    }
}

private data class SpeculativeDecodingChoice(
    val value: String,
    val label: (com.mobilefork.hermesagent.ui.i18n.HermesStrings) -> String,
)

private fun speculativeDecodingChoices(): List<SpeculativeDecodingChoice> = listOf(
    SpeculativeDecodingChoice("auto") { it.gemma4MtpAutoLabel() },
    SpeculativeDecodingChoice("enabled") { it.gemma4MtpEnabledLabel() },
    SpeculativeDecodingChoice("disabled") { it.gemma4MtpDisabledLabel() },
)

private fun localizedOnDeviceSummary(
    summary: String,
    strings: com.mobilefork.hermesagent.ui.i18n.HermesStrings,
): String {
    val trimmed = summary.trim()
    return when {
        trimmed.isBlank() -> strings.noCompatibleLocalModel
        trimmed == "Remote provider mode" -> strings.remoteProviderMode()
        trimmed == "Checking preferred local model…" -> strings.checkingPreferredLocalModel()
        trimmed.startsWith("No preferred local model") -> strings.noCompatibleLocalModel
        trimmed.startsWith("Preferred local model:") ->
            "${strings.preferredLocalModel}: ${trimmed.substringAfter(':').trim()}"
        else -> trimmed
    }
}

@Composable
private fun LanguagePickerCard(
    currentLanguageTag: String,
    onSelectLanguage: (AppLanguage) -> Unit,
    strings: com.mobilefork.hermesagent.ui.i18n.HermesStrings,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
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
            Text(strings.appLanguageTitle.ifBlank { "App language" }, style = MaterialTheme.typography.titleMedium)
            Text(strings.appLanguageDescription, style = MaterialTheme.typography.bodySmall)
            // Supported flags: 🇬🇧 🇨🇳 🇪🇸 🇩🇪 🇵🇹 🇫🇷
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                AppLanguage.entries.forEach { language ->
                    Button(
                        onClick = { onSelectLanguage(language) },
                        enabled = currentLanguageTag != language.tag,
                    ) {
                        Text("${language.flag} ${language.nativeLabel}")
                    }
                }
            }
        }
    }
}
