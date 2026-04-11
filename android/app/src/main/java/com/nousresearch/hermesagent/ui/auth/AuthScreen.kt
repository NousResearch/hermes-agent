package com.nousresearch.hermesagent.ui.auth

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun AuthScreen(
    modifier: Modifier = Modifier,
    viewModel: AuthViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsState()
    val scrollState = rememberScrollState()

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scrollState)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("Accounts", style = MaterialTheme.typography.headlineSmall)
                Text(uiState.globalStatus, style = MaterialTheme.typography.bodyMedium)

                OutlinedTextField(
                    value = uiState.corr3xtBaseUrl,
                    onValueChange = viewModel::updateCorr3xtBaseUrl,
                    label = { Text("Corr3xt auth base URL") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = viewModel::saveCorr3xtBaseUrl) {
                        Text("Save auth URL")
                    }
                    Button(onClick = viewModel::refresh) {
                        Text("Refresh")
                    }
                }

                uiState.options.forEach { option ->
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
                            Text(option.label, style = MaterialTheme.typography.titleMedium)
                            Text(option.description, style = MaterialTheme.typography.bodySmall)
                            Text(option.status, style = MaterialTheme.typography.bodyMedium)
                            if (option.accountHint.isNotBlank()) {
                                Text(option.accountHint, style = MaterialTheme.typography.bodySmall)
                            }
                            if (option.runtimeProvider.isNotBlank()) {
                                Text(
                                    "Hermes provider: ${option.runtimeProvider}",
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            }
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(12.dp),
                            ) {
                                Button(onClick = { viewModel.startAuth(option.id) }) {
                                    Text(if (option.signedIn) "Reconnect" else "Sign in")
                                }
                                if (option.signedIn) {
                                    Button(onClick = { viewModel.signOut(option.id) }) {
                                        Text("Sign out")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
