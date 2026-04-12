package com.nousresearch.hermesagent.ui.shell

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.ui.auth.AuthScreen
import com.nousresearch.hermesagent.ui.auth.AuthViewModel
import com.nousresearch.hermesagent.ui.boot.BootUiState
import com.nousresearch.hermesagent.ui.chat.ChatScreen
import com.nousresearch.hermesagent.ui.chat.ChatViewModel
import com.nousresearch.hermesagent.ui.device.DeviceScreen
import com.nousresearch.hermesagent.ui.device.DeviceViewModel
import com.nousresearch.hermesagent.ui.portal.NousPortalScreen
import com.nousresearch.hermesagent.ui.portal.NousPortalViewModel
import com.nousresearch.hermesagent.ui.i18n.AppLanguage
import com.nousresearch.hermesagent.ui.i18n.LocalHermesStrings
import com.nousresearch.hermesagent.ui.i18n.hermesStringsFor
import com.nousresearch.hermesagent.ui.settings.SettingsScreen
import com.nousresearch.hermesagent.ui.settings.SettingsViewModel
import com.nousresearch.hermesagent.ui.theme.HermesTheme

@Composable
fun AppShellScreen(
    bootUiState: BootUiState,
    onRetryHermes: () -> Unit,
) {
    var currentSection by rememberSaveable { mutableStateOf(AppSection.Hermes) }
    var currentActions by remember { mutableStateOf<List<ShellActionItem>>(emptyList()) }
    var showActionSheet by rememberSaveable { mutableStateOf(false) }

    val authViewModel: AuthViewModel = viewModel()
    val settingsViewModel: SettingsViewModel = viewModel()
    val deviceViewModel: DeviceViewModel = viewModel()
    val portalViewModel: NousPortalViewModel = viewModel()
    val chatViewModel: ChatViewModel = viewModel()
    val settingsState by settingsViewModel.uiState.collectAsState()
    val strings = hermesStringsFor(AppLanguage.fromTag(settingsState.languageTag))

    fun setActions(actions: List<ShellActionItem>) {
        currentActions = actions
        if (actions.isEmpty()) {
            showActionSheet = false
        }
    }

    val pageBottomClearance = if (currentActions.isNotEmpty() && currentSection != AppSection.Hermes) 104.dp else 24.dp

    LaunchedEffect(currentSection) {
        setActions(emptyList())
    }

    HermesTheme {
        CompositionLocalProvider(LocalHermesStrings provides strings) {
            Scaffold(
                modifier = Modifier.fillMaxSize(),
                containerColor = MaterialTheme.colorScheme.background,
                topBar = {
                    HermesTopBar(
                        section = currentSection,
                        bootUiState = bootUiState,
                    )
                },
                bottomBar = {
                    HermesBottomNavigation(
                        currentSection = currentSection,
                        onSelect = { currentSection = it },
                    )
                },
                floatingActionButton = {
                    if (currentActions.isNotEmpty() && currentSection != AppSection.Hermes) {
                        FloatingActionButton(onClick = { showActionSheet = true }) {
                            Icon(
                                painter = painterResource(id = R.drawable.ic_action_cog),
                                contentDescription = strings.openPageActions,
                            )
                        }
                    }
                },
            ) { innerPadding ->
                Surface(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(innerPadding),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    when (currentSection) {
                    AppSection.Hermes -> {
                        if (bootUiState.ready) {
                            ChatScreen(
                                modifier = Modifier.fillMaxSize(),
                                viewModel = chatViewModel,
                                settingsViewModel = settingsViewModel,
                                authViewModel = authViewModel,
                                onNavigateToSection = { currentSection = it },
                                onContextActionsChanged = ::setActions,
                                onOpenContextActions = { if (currentActions.isNotEmpty()) showActionSheet = true },
                            )
                        } else {
                            HermesSetupScreen(
                                uiState = bootUiState,
                                onRetry = onRetryHermes,
                                onOpenAccounts = { currentSection = AppSection.Accounts },
                                onOpenPortal = { currentSection = AppSection.NousPortal },
                                onOpenDevice = { currentSection = AppSection.Device },
                                onOpenSettings = { currentSection = AppSection.Settings },
                                onContextActionsChanged = ::setActions,
                                modifier = Modifier.fillMaxSize(),
                            )
                        }
                    }

                    AppSection.Accounts -> AuthScreen(
                        modifier = Modifier.fillMaxSize(),
                        viewModel = authViewModel,
                        extraBottomSpacing = pageBottomClearance,
                        onContextActionsChanged = ::setActions,
                    )

                    AppSection.NousPortal -> NousPortalScreen(
                        modifier = Modifier.fillMaxSize(),
                        viewModel = portalViewModel,
                        extraBottomSpacing = pageBottomClearance,
                        onContextActionsChanged = ::setActions,
                    )

                    AppSection.Device -> DeviceScreen(
                        modifier = Modifier.fillMaxSize(),
                        viewModel = deviceViewModel,
                        extraBottomSpacing = pageBottomClearance,
                        onContextActionsChanged = ::setActions,
                    )

                    AppSection.Settings -> SettingsScreen(
                        modifier = Modifier.fillMaxSize(),
                        viewModel = settingsViewModel,
                        extraBottomSpacing = pageBottomClearance,
                        onContextActionsChanged = ::setActions,
                    )
                }
            }

            if (showActionSheet && currentActions.isNotEmpty()) {
                ContextActionSheet(
                    section = currentSection,
                    actions = currentActions,
                    onDismiss = { showActionSheet = false },
                )
            }
        }
    }
}
}

@Composable
private fun HermesTopBar(
    section: AppSection,
    bootUiState: BootUiState,
) {
    val strings = LocalHermesStrings.current
    val subtitle = if (section == AppSection.Hermes && !bootUiState.ready) {
        strings.runtimeSetupAndOnboarding.ifBlank { "Runtime setup and onboarding" }
    } else {
        section.subtitle(strings)
    }
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.primaryContainer,
        tonalElevation = 2.dp,
    ) {
        Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .widthIn(max = 960.dp)
                    .statusBarsPadding()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Image(
                    painter = painterResource(id = R.drawable.ic_hermes_logo),
                    contentDescription = strings.hermesLogoDescription,
                    modifier = Modifier.size(34.dp),
                )
                Column(modifier = Modifier.weight(1f)) {
                    Text(section.title(strings), style = MaterialTheme.typography.titleLarge)
                    Text(subtitle, style = MaterialTheme.typography.bodySmall)
                }
                Surface(
                    color = MaterialTheme.colorScheme.secondary,
                    shape = MaterialTheme.shapes.small,
                ) {
                    Text(
                        text = strings.alphaBadge,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        color = MaterialTheme.colorScheme.onSecondary,
                        style = MaterialTheme.typography.labelMedium,
                    )
                }
            }
        }
    }
}

@Composable
private fun HermesBottomNavigation(
    currentSection: AppSection,
    onSelect: (AppSection) -> Unit,
) {
    val strings = LocalHermesStrings.current
    NavigationBar(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding(),
        tonalElevation = 8.dp,
    ) {
        AppSection.values().forEach { section ->
            NavigationBarItem(
                selected = currentSection == section,
                onClick = { onSelect(section) },
                icon = {
                    Icon(
                        painter = painterResource(id = section.iconRes),
                        contentDescription = section.label(strings),
                    )
                },
                label = { Text(section.label(strings)) },
            )
        }
    }
}

@Composable
private fun HermesSetupScreen(
    uiState: BootUiState,
    onRetry: () -> Unit,
    onOpenAccounts: () -> Unit,
    onOpenPortal: () -> Unit,
    onOpenDevice: () -> Unit,
    onOpenSettings: () -> Unit,
    onContextActionsChanged: (List<ShellActionItem>) -> Unit,
    modifier: Modifier = Modifier,
) {
    val strings = LocalHermesStrings.current
    LaunchedEffect(uiState.ready, uiState.error, uiState.status) {
        onContextActionsChanged(
            listOf(
                ShellActionItem(
                    label = strings.accounts.ifBlank { "Accounts" },
                    description = "Connect Corr3xt and provider sign-ins.",
                    iconRes = R.drawable.ic_nav_accounts,
                    onClick = onOpenAccounts,
                ),
                ShellActionItem(
                    label = strings.settings.ifBlank { "Settings" },
                    description = "Configure provider, model, and API key.",
                    iconRes = R.drawable.ic_nav_settings,
                    onClick = onOpenSettings,
                ),
                // label = "Nous Portal"
                ShellActionItem(
                    label = strings.portalTitle.ifBlank { "Nous Portal" },
                    description = "Open the portal page while Hermes boots.",
                    iconRes = R.drawable.ic_nav_portal,
                    onClick = onOpenPortal,
                ),
                // label = "Device"
                ShellActionItem(
                    label = strings.sectionDevice.ifBlank { "Device" },
                    description = "Grant files, Linux tools, and phone controls.",
                    iconRes = R.drawable.ic_nav_device,
                    onClick = onOpenDevice,
                ),
            )
        )
    }

    Column(
        modifier = modifier
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Image(
            painter = painterResource(id = R.drawable.ic_hermes_logo),
            contentDescription = strings.hermesLogoDescription,
            modifier = Modifier.size(72.dp),
        )
        Text(uiState.status, style = MaterialTheme.typography.headlineSmall)
        if (uiState.baseUrl.isNotBlank()) {
            Text(uiState.baseUrl, style = MaterialTheme.typography.bodySmall)
        }
        if (uiState.probeResult.isNotBlank()) {
            Text(uiState.probeResult, style = MaterialTheme.typography.bodySmall)
        }
        if (uiState.error.isNotBlank()) {
            Surface(
                modifier = Modifier.fillMaxWidth(),
                color = MaterialTheme.colorScheme.error.copy(alpha = 0.12f),
                shape = MaterialTheme.shapes.large,
            ) {
                Text(
                    text = uiState.error,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(14.dp),
                )
            }
        }
        Button(onClick = onRetry) {
            Text("Retry Hermes")
        }
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = MaterialTheme.colorScheme.surfaceVariant,
            shape = MaterialTheme.shapes.large,
            tonalElevation = 1.dp,
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
                Text("3. Device: grant shared-folder access if you want Hermes to edit real mobile files directly.")
                Text("4. Hermes chat: use voice input, chat commands, or the cog button for page-specific actions once the runtime is ready.")
            }
        }
    }
}
