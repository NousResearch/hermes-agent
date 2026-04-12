@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.nousresearch.hermesagent.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nousresearch.hermesagent.ui.i18n.LocalHermesStrings

@Composable
fun LocalModelDownloadsSection(
    dataSaverMode: Boolean,
    onDataSaverModeChange: (Boolean) -> Unit,
    selectedBackend: String,
    onRuntimeFlavorSelected: (String) -> Unit,
    viewModel: LocalModelDownloadsViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsState()
    val strings = LocalHermesStrings.current
    val effectiveRuntimeFlavor = when (selectedBackend) {
        "llama.cpp" -> "GGUF"
        "litert-lm" -> "LiteRT-LM"
        else -> uiState.runtimeFlavor
    }

    LaunchedEffect(selectedBackend) {
        viewModel.syncSelectedBackend(selectedBackend)
    }

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.large,
        tonalElevation = 2.dp,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(strings.localDownloadsTitle.ifBlank { "Hugging Face local model downloads" }, style = MaterialTheme.typography.titleMedium)
            Text(
                strings.localDownloadsDescription.ifBlank {
                    "Download full model files directly to the phone, keep progress in Android's system download manager, and resume safely after network loss or a phone restart. PocketPal AI is a good reference for the kind of mobile-local model hub Hermes is moving toward."
                },
                style = MaterialTheme.typography.bodySmall,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(strings.dataSaverModeTitle.ifBlank { "Data saver mode" }, style = MaterialTheme.typography.titleSmall)
                    Text(
                        strings.dataSaverModeDescription.ifBlank {
                            "When enabled, large model downloads wait for Wi‑Fi / unmetered connectivity so Hermes uses only minimal mobile data."
                        },
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Switch(
                    checked = dataSaverMode,
                    onCheckedChange = onDataSaverModeChange,
                )
            }
            OutlinedTextField(
                value = uiState.huggingFaceToken,
                onValueChange = viewModel::updateHuggingFaceToken,
                label = { Text(strings.huggingFaceTokenOptional.ifBlank { "Hugging Face token (optional)" }) },
                modifier = Modifier.fillMaxWidth(),
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = viewModel::saveHuggingFaceToken) {
                    Text(strings.saveToken.ifBlank { "Save token" })
                }
                Button(onClick = viewModel::refreshDownloads) {
                    Text(strings.refreshDownloads.ifBlank { "Refresh downloads" })
                }
            }
            HorizontalDivider()
            OutlinedTextField(
                value = uiState.repoOrUrl,
                onValueChange = viewModel::updateRepoOrUrl,
                label = { Text(strings.repoIdOrDirectUrl.ifBlank { "Repo ID or direct URL" }) },
                modifier = Modifier.fillMaxWidth(),
            )
            OutlinedTextField(
                value = uiState.filePath,
                onValueChange = viewModel::updateFilePath,
                label = { Text(strings.filePathInsideRepo.ifBlank { "File path inside repo" }) },
                modifier = Modifier.fillMaxWidth(),
            )
            OutlinedTextField(
                value = uiState.revision,
                onValueChange = viewModel::updateRevision,
                label = { Text(strings.revision.ifBlank { "Revision" }) },
                modifier = Modifier.fillMaxWidth(),
            )
            Text(strings.runtimeTarget.ifBlank { "Runtime target" }, style = MaterialTheme.typography.titleSmall)
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = {
                    viewModel.updateRuntimeFlavor("GGUF")
                    onRuntimeFlavorSelected("GGUF")
                }, enabled = effectiveRuntimeFlavor != "GGUF") {
                    Text("GGUF")
                }
                Button(onClick = {
                    viewModel.updateRuntimeFlavor("LiteRT-LM")
                    onRuntimeFlavorSelected("LiteRT-LM")
                }, enabled = effectiveRuntimeFlavor != "LiteRT-LM") {
                    Text("LiteRT-LM")
                }
            }
            Text(
                strings.localDownloadsExampleGuidance(),
                style = MaterialTheme.typography.bodySmall,
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = { viewModel.inspectCandidate(runtimeFlavorOverride = effectiveRuntimeFlavor) }) {
                    Text(strings.inspect.ifBlank { "Inspect" })
                }
                Button(onClick = { viewModel.startDownload(dataSaverMode, runtimeFlavorOverride = effectiveRuntimeFlavor) }) {
                    Text(strings.download.ifBlank { "Download" })
                }
            }
            if (uiState.inspectionStatus.isNotBlank()) {
                Text(uiState.inspectionStatus, style = MaterialTheme.typography.bodySmall)
            }
            if (uiState.candidateSummary.isNotBlank()) {
                Text(uiState.candidateSummary, style = MaterialTheme.typography.bodySmall)
            }
            if (uiState.candidateRamWarning.isNotBlank()) {
                Text(uiState.candidateRamWarning, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
            HorizontalDivider()
            Text(strings.downloadManagerTitle.ifBlank { "Download manager" }, style = MaterialTheme.typography.titleSmall)
            Text(
                strings.downloadManagerReliabilityDescription(),
                style = MaterialTheme.typography.bodySmall,
            )
            if (uiState.downloads.isEmpty()) {
                Text(strings.noLocalModelDownloadsYet.ifBlank { "No local model downloads yet." }, style = MaterialTheme.typography.bodySmall)
            } else {
                uiState.downloads.forEach { item ->
                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        color = MaterialTheme.colorScheme.surface,
                        shape = MaterialTheme.shapes.medium,
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(14.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                    Text(item.title, style = MaterialTheme.typography.titleSmall)
                                    Text(strings.localDownloadStatusLine(item.runtimeFlavor, item.statusLabel), style = MaterialTheme.typography.labelMedium)
                                }
                                if (item.isPreferred) {
                                    Text(strings.preferredLocalModel.ifBlank { "Preferred local model" }, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.secondary)
                                }
                            }
                            LinearProgressIndicator(
                                progress = { item.progressFraction },
                                modifier = Modifier.fillMaxWidth(),
                            )
                            Text(item.progressLabel, style = MaterialTheme.typography.bodySmall)
                            Text(item.statusMessage, style = MaterialTheme.typography.bodySmall)
                            if (item.ramWarning.isNotBlank()) {
                                Text(item.ramWarning, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                            }
                            Text(item.localPath, style = MaterialTheme.typography.bodySmall)
                            FlowRow(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                if (!item.isPreferred && item.statusLabel == "completed") {
                                    Button(onClick = { viewModel.setPreferredDownload(item.id) }) {
                                        Text(strings.setPreferred.ifBlank { "Set preferred" })
                                    }
                                }
                                if (item.canRestartOnMobileData) {
                                    Button(onClick = { viewModel.restartDownloadOnMobileData(item.id) }) {
                                        Text(strings.restartOnMobileData())
                                    }
                                }
                                if (item.canOpenSystemDownloads) {
                                    Button(onClick = viewModel::openSystemDownloads) {
                                        Text(strings.openSystemDownloads())
                                    }
                                }
                                Button(onClick = { viewModel.removeDownload(item.id) }) {
                                    Text(strings.remove.ifBlank { "Remove" })
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
