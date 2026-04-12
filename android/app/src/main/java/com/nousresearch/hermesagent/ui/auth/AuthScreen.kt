package com.nousresearch.hermesagent.ui.auth

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.ui.shell.ShellActionItem

@Composable
fun AuthScreen(
    modifier: Modifier = Modifier,
    viewModel: AuthViewModel = viewModel(),
    extraBottomSpacing: Dp = 0.dp,
    onContextActionsChanged: (List<ShellActionItem>) -> Unit = {},
) {
    val uiState by viewModel.uiState.collectAsState()
    val scrollState = rememberScrollState()

    SideEffect {
        val actions = buildList {
            add(
                ShellActionItem(
                    label = "Refresh auth state",
                    description = "Reload local Corr3xt and provider auth status.",
                    iconRes = R.drawable.ic_action_refresh,
                    onClick = viewModel::refresh,
                )
            )
            if (uiState.hasPendingRequest) {
                add(
                    ShellActionItem(
                        label = "Cancel pending sign-in",
                        description = "Stop waiting for the current Corr3xt callback.",
                        iconRes = R.drawable.ic_nav_settings,
                        onClick = viewModel::cancelPendingRequest,
                    )
                )
            }
        }
        onContextActionsChanged(actions)
    }

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .widthIn(max = 920.dp)
                        .verticalScroll(scrollState)
                        .padding(horizontal = 16.dp, vertical = 12.dp)
                        .padding(bottom = extraBottomSpacing),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                Text(uiState.globalStatus, style = MaterialTheme.typography.bodyMedium)
                Text(
                    "Corr3xt opens in your browser and returns to Hermes through a secure callback.",
                    style = MaterialTheme.typography.bodySmall,
                )

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

                if (uiState.hasPendingRequest) {
                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        color = MaterialTheme.colorScheme.secondaryContainer,
                        tonalElevation = 1.dp,
                        shape = MaterialTheme.shapes.medium,
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Text("Pending Corr3xt sign-in", style = MaterialTheme.typography.titleMedium)
                            Text(
                                "Waiting for Corr3xt callback for ${uiState.pendingMethodLabel}.",
                                style = MaterialTheme.typography.bodySmall,
                            )
                            Button(onClick = viewModel::cancelPendingRequest) {
                                Text("Cancel pending sign-in")
                            }
                        }
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
}

