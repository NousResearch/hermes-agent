package com.nousresearch.hermesagent.ui.shell

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.weight
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.ui.auth.AuthScreen
import com.nousresearch.hermesagent.ui.boot.BootUiState
import com.nousresearch.hermesagent.ui.chat.ChatScreen
import com.nousresearch.hermesagent.ui.portal.NousPortalScreen
import com.nousresearch.hermesagent.ui.settings.SettingsScreen
import com.nousresearch.hermesagent.ui.theme.HermesTheme

enum class AppSection(val label: String) {
    Hermes("Hermes"),
    Accounts("Accounts"),
    NousPortal("Nous Portal"),
    Settings("Settings"),
}

@Composable
fun AppShellScreen(
    bootUiState: BootUiState,
    onRetryHermes: () -> Unit,
) {
    var currentSection by rememberSaveable { mutableStateOf(AppSection.Hermes) }

    HermesTheme {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(modifier = Modifier.fillMaxSize()) {
                HermesBrandBar()
                TabRow(selectedTabIndex = currentSection.ordinal) {
                    AppSection.values().forEach { section ->
                        Tab(
                            selected = currentSection == section,
                            onClick = { currentSection = section },
                            text = { Text(section.label) },
                        )
                    }
                }

                when (currentSection) {
                    AppSection.Hermes -> HermesSection(
                        uiState = bootUiState,
                        onRetry = onRetryHermes,
                        modifier = Modifier.fillMaxSize(),
                    )

                    AppSection.Accounts -> AuthScreen(modifier = Modifier.fillMaxSize())
                    AppSection.NousPortal -> NousPortalScreen(modifier = Modifier.fillMaxSize())
                    AppSection.Settings -> SettingsScreen(modifier = Modifier.fillMaxSize())
                }
            }
        }
    }
}

@Composable
private fun HermesBrandBar() {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.primaryContainer,
        tonalElevation = 2.dp,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Image(
                painter = painterResource(id = R.drawable.ic_hermes_logo),
                contentDescription = "Hermes logo",
                modifier = Modifier.size(36.dp),
            )
            Column(modifier = Modifier.weight(1f)) {
                Text("Hermes", style = MaterialTheme.typography.titleLarge)
                Text(
                    "Android alpha · local runtime + portal access",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            Surface(
                color = MaterialTheme.colorScheme.secondary,
                shape = MaterialTheme.shapes.small,
            ) {
                Text(
                    text = "ALPHA",
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                    color = MaterialTheme.colorScheme.onSecondary,
                    style = MaterialTheme.typography.labelMedium,
                )
            }
        }
    }
}

@Composable
private fun HermesSection(
    uiState: BootUiState,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier,
) {
    if (uiState.ready) {
        ChatScreen(modifier = modifier)
        return
    }

    Column(
        modifier = modifier.padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Image(
            painter = painterResource(id = R.drawable.ic_hermes_logo),
            contentDescription = "Hermes logo",
            modifier = Modifier.size(72.dp),
        )
        Text(
            text = uiState.status,
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(top = 16.dp),
        )
        if (uiState.baseUrl.isNotBlank()) {
            Text(text = uiState.baseUrl, modifier = Modifier.padding(top = 12.dp))
        }
        if (uiState.probeResult.isNotBlank()) {
            Text(text = uiState.probeResult, modifier = Modifier.padding(top = 12.dp))
        }
        if (uiState.error.isNotBlank()) {
            Text(
                text = uiState.error,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(top = 12.dp),
            )
        }
        Button(onClick = onRetry, modifier = Modifier.padding(top = 20.dp)) {
            Text("Retry Hermes")
        }
        Text(
            text = "The Nous Portal section is available independently while the local Hermes runtime is booting.",
            modifier = Modifier.padding(top = 16.dp),
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}
