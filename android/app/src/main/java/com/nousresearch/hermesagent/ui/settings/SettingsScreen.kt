@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.nousresearch.hermesagent.ui.settings

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
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nousresearch.hermesagent.data.ProviderPresets
import com.nousresearch.hermesagent.backend.BackendKind
import com.nousresearch.hermesagent.ui.i18n.AppLanguage
import com.nousresearch.hermesagent.ui.i18n.LocalHermesStrings
import com.nousresearch.hermesagent.ui.shell.ShellActionItem

@OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    modifier: Modifier = Modifier,
    viewModel: SettingsViewModel = viewModel(),
    extraBottomSpacing: Dp = 0.dp,
    onContextActionsChanged: (List<ShellActionItem>) -> Unit = {},
) {
    val uiState by viewModel.uiState.collectAsState()
    val strings = LocalHermesStrings.current
    var expanded by remember { mutableStateOf(false) }
    var modelExpanded by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()
    val selectedPreset = ProviderPresets.find(uiState.provider)

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
                        providerLabel = selectedPreset?.label ?: uiState.provider,
                        strings = strings,
                    )
                    LanguagePickerCard(
                        currentLanguageTag = uiState.languageTag,
                        onSelectLanguage = viewModel::selectLanguage,
                        strings = strings,
                    )
                    OnDeviceInferenceCard(
                        onDeviceBackend = uiState.onDeviceBackend,
                        onSelectBackend = viewModel::updateOnDeviceBackend,
                        summary = uiState.onDeviceSummary,
                        strings = strings,
                    )

                    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = !expanded }) {
                        OutlinedTextField(
                            value = uiState.provider,
                            onValueChange = {},
                            readOnly = true,
                            label = { Text(strings.providerLabel()) },
                            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                            modifier = Modifier
                                .fillMaxWidth()
                                .menuAnchor()
                                .testTag("HermesProviderDropdown"),
                        )
                        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                            ProviderPresets.defaults.forEach { preset ->
                                DropdownMenuItem(
                                    text = { Text(preset.label) },
                                    onClick = {
                                        viewModel.updateProvider(preset.id)
                                        if (uiState.baseUrl.isBlank()) {
                                            viewModel.updateBaseUrl(preset.baseUrl)
                                        }
                                        if (uiState.model.isBlank()) {
                                            viewModel.updateModel(preset.modelHint)
                                        }
                                        expanded = false
                                    },
                                )
                            }
                        }
                    }
                    Text(
                        strings.providerDirectCallHelp(),
                        style = MaterialTheme.typography.bodySmall,
                    )

                    OutlinedTextField(
                        value = uiState.baseUrl,
                        onValueChange = viewModel::updateBaseUrl,
                        label = { Text(strings.baseUrlLabel()) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Text(
                        strings.defaultBaseUrlSummary(
                            selectedPreset?.label ?: uiState.provider,
                            selectedPreset?.baseUrl?.ifBlank { "provider default / optional" } ?: "provider default / optional",
                        ),
                        style = MaterialTheme.typography.bodySmall,
                    )

                    ModelSelectionCard(
                        expanded = modelExpanded,
                        onExpandedChange = { modelExpanded = !modelExpanded },
                        selectedModel = uiState.model,
                        providerId = uiState.provider,
                        suggestedModel = selectedPreset?.modelHint?.ifBlank { "choose a provider-supported model" }
                            ?: "choose a provider-supported model",
                        onSelectModel = { modelId ->
                            viewModel.updateModel(modelId)
                            modelExpanded = false
                        },
                        onModelTextChange = viewModel::updateModel,
                        strings = strings,
                    )

                    OutlinedTextField(
                        value = uiState.apiKey,
                        onValueChange = viewModel::updateApiKey,
                        label = { Text(strings.apiKeyLabel()) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Text(
                        strings.apiKeyHelp(),
                        style = MaterialTheme.typography.bodySmall,
                    )

                    ToolProfileCard()
                    LocalModelDownloadsSection(
                        dataSaverMode = uiState.dataSaverMode,
                        onDataSaverModeChange = viewModel::updateDataSaverMode,
                        selectedBackend = uiState.onDeviceBackend,
                        onRuntimeFlavorSelected = viewModel::syncOnDeviceBackendWithRuntimeFlavor,
                    )

                    Button(onClick = viewModel::save) {
                        Text(strings.saveLabel())
                    }

                    if (uiState.status.isNotBlank()) {
                        Text(uiState.status)
                    }
                }
            }
        }
    }
}

@OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
private fun ModelSelectionCard(
    expanded: Boolean,
    onExpandedChange: () -> Unit,
    selectedModel: String,
    providerId: String,
    suggestedModel: String,
    onSelectModel: (String) -> Unit,
    onModelTextChange: (String) -> Unit,
    strings: com.nousresearch.hermesagent.ui.i18n.HermesStrings,
) {
    val options = ProviderPresets.modelSelections(providerId)
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
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(strings.modelSelectionTitle(), style = MaterialTheme.typography.titleMedium)
            Text(strings.modelSelectionDescription(), style = MaterialTheme.typography.bodySmall)
            ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { onExpandedChange() }) {
                OutlinedTextField(
                    value = selectedModel,
                    onValueChange = onModelTextChange,
                    label = { Text(strings.modelLabel()) },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                        .testTag("HermesModelDropdown"),
                    singleLine = true,
                )
                DropdownMenu(expanded = expanded, onDismissRequest = onExpandedChange) {
                    options.forEach { model ->
                        DropdownMenuItem(
                            text = {
                                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                                    Text(model.label)
                                    Text(model.description, style = MaterialTheme.typography.bodySmall)
                                }
                            },
                            onClick = { onSelectModel(model.id) },
                        )
                    }
                }
            }
            Text(strings.suggestedModelSummary(suggestedModel), style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun SettingsHelpCard(
    providerLabel: String,
    strings: com.nousresearch.hermesagent.ui.i18n.HermesStrings,
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
            Text(strings.settingsHelpStart)
            // Use Accounts if you want Corr3xt-based sign-in flows
            Text(strings.settingsHelpAccounts)
            Text(strings.currentProviderProfile(providerLabel))
        }
    }
}

@Composable
private fun OnDeviceInferenceCard(
    onDeviceBackend: String,
    onSelectBackend: (String) -> Unit,
    summary: String,
    strings: com.nousresearch.hermesagent.ui.i18n.HermesStrings,
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
            BackendSwitchRow(
                title = strings.llamaCppLabel.ifBlank { "llama.cpp (GGUF)" },
                description = strings.llamaCppDescription,
                checked = onDeviceBackend == BackendKind.LLAMA_CPP.persistedValue,
                onCheckedChange = { enabled ->
                    onSelectBackend(if (enabled) BackendKind.LLAMA_CPP.persistedValue else BackendKind.NONE.persistedValue)
                },
            )
            BackendSwitchRow(
                title = strings.liteRtLmLabel.ifBlank { "LiteRT-LM" },
                description = strings.liteRtLmDescription,
                checked = onDeviceBackend == BackendKind.LITERT_LM.persistedValue,
                onCheckedChange = { enabled ->
                    onSelectBackend(if (enabled) BackendKind.LITERT_LM.persistedValue else BackendKind.NONE.persistedValue)
                },
            )
            Text(localizedOnDeviceSummary(summary, strings), style = MaterialTheme.typography.bodySmall)
        }
    }
}

private fun localizedOnDeviceSummary(
    summary: String,
    strings: com.nousresearch.hermesagent.ui.i18n.HermesStrings,
): String {
    val trimmed = summary.trim()
    return when {
        trimmed.isBlank() -> strings.noCompatibleLocalModel
        trimmed.startsWith("No preferred local model") -> strings.noCompatibleLocalModel
        trimmed.startsWith("Preferred local model:") ->
            "${strings.preferredLocalModel}: ${trimmed.substringAfter(':').trim()}"
        else -> trimmed
    }
}

@Composable
private fun BackendSwitchRow(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
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
            Text(title, style = MaterialTheme.typography.titleSmall)
            Text(description, style = MaterialTheme.typography.bodySmall)
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun LanguagePickerCard(
    currentLanguageTag: String,
    onSelectLanguage: (AppLanguage) -> Unit,
    strings: com.nousresearch.hermesagent.ui.i18n.HermesStrings,
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
