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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
import com.nousresearch.hermesagent.ui.device.DeviceScreen
import com.nousresearch.hermesagent.ui.portal.NousPortalScreen
import com.nousresearch.hermesagent.ui.settings.SettingsScreen
import com.nousresearch.hermesagent.ui.theme.HermesTheme

enum class AppSection(val label: String) {
    Hermes("Hermes"),
    Accounts("Accounts"),
    NousPortal("Nous Portal"),
    Device("Device"),
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
                        onOpenAccounts = { currentSection = AppSection.Accounts },
                        onOpenPortal = { currentSection = AppSection.NousPortal },
                        onOpenDevice = { currentSection = AppSection.Device },
                        onOpenSettings = { currentSection = AppSection.Settings },
                        modifier = Modifier.fillMaxSize(),
                    )

                    AppSection.Accounts -> AuthScreen(modifier = Modifier.fillMaxSize())
                    AppSection.NousPortal -> NousPortalScreen(modifier = Modifier.fillMaxSize())
                    AppSection.Device -> DeviceScreen(modifier = Modifier.fillMaxSize())
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
    onOpenAccounts: () -> Unit,
    onOpenPortal: () -> Unit,
    onOpenDevice: () -> Unit,
    onOpenSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    if (uiState.ready) {
        ChatScreen(modifier = modifier)
        return
    }

    val scrollState = rememberScrollState()
    Column(
        modifier = modifier
            .verticalScroll(scrollState)
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
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
        )
        if (uiState.baseUrl.isNotBlank()) {
            Text(text = uiState.baseUrl)
        }
        if (uiState.probeResult.isNotBlank()) {
            Text(text = uiState.probeResult)
        }
        if (uiState.error.isNotBlank()) {
            Text(
                text = uiState.error,
                color = MaterialTheme.colorScheme.error,
            )
        }
        Button(onClick = onRetry) {
            Text("Retry Hermes")
        }
        GettingStartedCard(
            onOpenAccounts = onOpenAccounts,
            onOpenPortal = onOpenPortal,
            onOpenDevice = onOpenDevice,
            onOpenSettings = onOpenSettings,
        )
        Text(
            text = "The Nous Portal and Device sections are available independently while the local Hermes runtime is booting.",
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}

@Composable
private fun GettingStartedCard(
    onOpenAccounts: () -> Unit,
    onOpenPortal: () -> Unit,
    onOpenDevice: () -> Unit,
    onOpenSettings: () -> Unit,
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
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text("Getting started", style = MaterialTheme.typography.titleMedium)
            Text("1. Accounts: connect ChatGPT, Claude, Gemini, email, phone, or Google.")
            Text("2. Settings: choose a provider, confirm the base URL/model, and save your API key.")
            Text("3. Nous Portal: open the full portal experience in your browser if the embedded preview stays limited.")
            Text("4. Device: import files into the Hermes workspace and enable accessibility controls if you want high-level phone actions.")
            Text("5. Hermes: return here and tap Retry Hermes after setup changes.")
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onOpenAccounts, modifier = Modifier.weight(1f)) {
                    Text("Accounts")
                }
                Button(onClick = onOpenSettings, modifier = Modifier.weight(1f)) {
                    Text("Settings")
                }
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Button(onClick = onOpenPortal, modifier = Modifier.weight(1f)) {
                    Text("Open Nous Portal")
                }
                Button(onClick = onOpenDevice, modifier = Modifier.weight(1f)) {
                    Text("Device")
                }
            }
        }
    }
}
