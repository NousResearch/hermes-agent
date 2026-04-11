package com.nousresearch.hermesagent.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nousresearch.hermesagent.data.ProviderPresets

@OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    modifier: Modifier = Modifier,
    viewModel: SettingsViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsState()
    var expanded by remember { mutableStateOf(false) }
    val selectedPreset = ProviderPresets.find(uiState.provider)

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("Settings", style = MaterialTheme.typography.headlineSmall)
                SettingsHelpCard(providerLabel = selectedPreset?.label ?: uiState.provider)

                ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = !expanded }) {
                    OutlinedTextField(
                        value = uiState.provider,
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("Provider") },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                        modifier = Modifier
                            .menuAnchor()
                            .fillMaxWidth(),
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
                    "Choose the provider you want Hermes to call directly. Use Accounts for browser-based sign-ins; use Settings for API-key based setup.",
                    style = MaterialTheme.typography.bodySmall,
                )

                OutlinedTextField(
                    value = uiState.baseUrl,
                    onValueChange = viewModel::updateBaseUrl,
                    label = { Text("Base URL") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    "Default for ${selectedPreset?.label ?: uiState.provider}: ${selectedPreset?.baseUrl?.ifBlank { "provider default / optional" } ?: "provider default / optional"}",
                    style = MaterialTheme.typography.bodySmall,
                )

                OutlinedTextField(
                    value = uiState.model,
                    onValueChange = viewModel::updateModel,
                    label = { Text("Model") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    "Suggested model: ${selectedPreset?.modelHint?.ifBlank { "choose a provider-supported model" } ?: "choose a provider-supported model"}",
                    style = MaterialTheme.typography.bodySmall,
                )

                OutlinedTextField(
                    value = uiState.apiKey,
                    onValueChange = viewModel::updateApiKey,
                    label = { Text("API Key") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    "Paste the key for the selected provider, then tap Save to restart the local Hermes backend with the new config.",
                    style = MaterialTheme.typography.bodySmall,
                )

                ToolProfileCard()

                Button(onClick = viewModel::save) {
                    Text("Save")
                }

                if (uiState.status.isNotBlank()) {
                    Text(uiState.status)
                }
            }
        }
    }
}

@Composable
private fun SettingsHelpCard(providerLabel: String) {
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
            Text("New here?", style = MaterialTheme.typography.titleMedium)
            Text("Start with OpenRouter or another API provider if you already have a key.")
            Text("Use Accounts if you want Corr3xt-based sign-in flows for ChatGPT, Claude, Gemini, email, phone, or Google.")
            Text("Current provider profile: $providerLabel")
        }
    }
}
