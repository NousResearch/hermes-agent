package com.mobilefork.hermesagent.ui.shell

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.Image
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.zIndex
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilefork.hermesagent.R
import com.mobilefork.hermesagent.data.AppSettingsStore
import com.mobilefork.hermesagent.data.ProviderPresets
import com.mobilefork.hermesagent.ui.auth.AuthScreen
import com.mobilefork.hermesagent.ui.auth.AuthViewModel
import com.mobilefork.hermesagent.ui.boot.BootUiState
import com.mobilefork.hermesagent.ui.chat.ChatScreen
import com.mobilefork.hermesagent.ui.chat.ChatViewModel
import com.mobilefork.hermesagent.ui.device.DeviceScreen
import com.mobilefork.hermesagent.ui.device.DeviceViewModel
import com.mobilefork.hermesagent.ui.portal.NousPortalScreen
import com.mobilefork.hermesagent.ui.portal.NousPortalViewModel
import com.mobilefork.hermesagent.ui.i18n.AppLanguage
import com.mobilefork.hermesagent.ui.i18n.LocalHermesStrings
import com.mobilefork.hermesagent.ui.i18n.hermesStringsFor
import com.mobilefork.hermesagent.ui.settings.SettingsScreen
import com.mobilefork.hermesagent.ui.settings.SettingsViewModel
import com.mobilefork.hermesagent.ui.theme.HermesThemeConfig
import com.mobilefork.hermesagent.ui.theme.HermesTheme
import com.mobilefork.hermesagent.ui.theme.normalizeThemeHex

internal fun shellDrawerNavigationSections(): List<AppSection> = AppSection.values().toList()

private data class ShellSettingsState(
    val languageTag: String = AppLanguage.ENGLISH.tag,
    val chatDisplayMode: String = "compact",
    val keywordHighlightingEnabled: Boolean = true,
    val themePrimaryHex: String = "#8C7BFF",
    val themeSecondaryHex: String = "#C6A15B",
    val themeBackgroundHex: String = "#090B10",
    val themeSurfaceHex: String = "#11141C",
    val themeSurfaceVariantHex: String = "#1B202B",
    val themeCardShape: String = "rounded",
)

private fun loadShellSettingsState(settingsStore: AppSettingsStore): ShellSettingsState {
    val stored = settingsStore.load()
    return ShellSettingsState(
        languageTag = AppLanguage.fromTag(stored.languageTag).tag,
        chatDisplayMode = normalizeShellChatDisplayMode(stored.chatDisplayMode),
        keywordHighlightingEnabled = stored.keywordHighlightingEnabled,
        themePrimaryHex = normalizeThemeHex(stored.themePrimaryHex, "#8C7BFF"),
        themeSecondaryHex = normalizeThemeHex(stored.themeSecondaryHex, "#C6A15B"),
        themeBackgroundHex = normalizeThemeHex(stored.themeBackgroundHex, "#090B10"),
        themeSurfaceHex = normalizeThemeHex(stored.themeSurfaceHex, "#11141C"),
        themeSurfaceVariantHex = normalizeThemeHex(stored.themeSurfaceVariantHex, "#1B202B"),
        themeCardShape = normalizeShellThemeCardShape(stored.themeCardShape),
    )
}

private fun normalizeShellChatDisplayMode(value: String): String = if (value == "expanded") "expanded" else "compact"

private fun normalizeShellThemeCardShape(value: String): String = when (value) {
    "cut", "rounded", "soft" -> value
    else -> "rounded"
}

@Composable
fun AppShellScreen(
    bootUiState: BootUiState,
    onRetryHermes: () -> Unit,
) {
    var currentSection by rememberSaveable { mutableStateOf(AppSection.Hermes) }
    var currentActions by remember { mutableStateOf<List<ShellActionItem>>(emptyList()) }
    var showActionSheet by rememberSaveable { mutableStateOf(false) }
    var showNavigationDrawer by rememberSaveable { mutableStateOf(false) }
    var sectionContentReady by rememberSaveable { mutableStateOf(true) }

    val context = LocalContext.current.applicationContext
    val appSettingsStore = remember(context) { AppSettingsStore(context) }
    var shellSettings by remember { mutableStateOf(loadShellSettingsState(appSettingsStore)) }
    val strings = hermesStringsFor(AppLanguage.fromTag(shellSettings.languageTag))

    fun refreshShellSettings() {
        shellSettings = loadShellSettingsState(appSettingsStore)
    }

    fun updateChatDisplayMode(value: String) {
        val normalized = normalizeShellChatDisplayMode(value)
        appSettingsStore.save(appSettingsStore.load().copy(chatDisplayMode = normalized))
        refreshShellSettings()
    }

    fun applyProvider(providerId: String): Boolean {
        val preset = ProviderPresets.find(providerId) ?: return false
        appSettingsStore.save(
            appSettingsStore.load().copy(
                provider = preset.id,
                baseUrl = preset.baseUrl,
                model = preset.modelHint,
            ),
        )
        refreshShellSettings()
        return true
    }

    fun applyModel(modelName: String): Boolean {
        val normalized = modelName.trim()
        if (normalized.isBlank()) return false
        appSettingsStore.save(appSettingsStore.load().copy(model = normalized))
        refreshShellSettings()
        return true
    }

    fun setActions(actions: List<ShellActionItem>) {
        currentActions = actions
        if (actions.isEmpty()) {
            showActionSheet = false
        }
    }

    fun navigateToSection(section: AppSection) {
        refreshShellSettings()
        showNavigationDrawer = false
        if (section != currentSection) {
            sectionContentReady = section == AppSection.Hermes
            showActionSheet = false
        }
        currentSection = section
    }

    val pageBottomClearance = 24.dp

    LaunchedEffect(currentSection) {
        setActions(emptyList())
    }

    LaunchedEffect(currentSection, sectionContentReady) {
        if (!sectionContentReady) {
            withFrameNanos { }
            sectionContentReady = true
        }
    }

    HermesTheme(
        config = HermesThemeConfig(
            primaryHex = shellSettings.themePrimaryHex,
            secondaryHex = shellSettings.themeSecondaryHex,
            backgroundHex = shellSettings.themeBackgroundHex,
            surfaceHex = shellSettings.themeSurfaceHex,
            surfaceVariantHex = shellSettings.themeSurfaceVariantHex,
            cardShape = shellSettings.themeCardShape,
        ),
    ) {
        CompositionLocalProvider(LocalHermesStrings provides strings) {
            Box(modifier = Modifier.fillMaxSize()) {
                val showShellNavigation = currentSection != AppSection.Hermes
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    containerColor = MaterialTheme.colorScheme.background,
                    topBar = {
                        if (showShellNavigation) {
                            HermesTopBar(
                                section = currentSection,
                                bootUiState = bootUiState,
                                onOpenNavigationMenu = { showNavigationDrawer = true },
                            )
                        }
                    },
                ) { innerPadding ->
                    Surface(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding),
                        color = MaterialTheme.colorScheme.background,
                    ) {
                        if (!sectionContentReady) {
                            SectionWarmupPane(
                                modifier = Modifier.fillMaxSize(),
                                section = currentSection,
                            )
                        } else when (currentSection) {
                            AppSection.Hermes -> {
                                val authViewModel: AuthViewModel = viewModel()
                                val chatViewModel: ChatViewModel = viewModel()
                                ChatScreen(
                                    modifier = Modifier.fillMaxSize(),
                                    viewModel = chatViewModel,
                                    chatDisplayMode = shellSettings.chatDisplayMode,
                                    keywordHighlightingEnabled = shellSettings.keywordHighlightingEnabled,
                                    authViewModel = authViewModel,
                                    onNavigateToSection = ::navigateToSection,
                                    onContextActionsChanged = ::setActions,
                                    onOpenNavigationMenu = { showNavigationDrawer = true },
                                    onOpenContextActions = { showActionSheet = true },
                                    onToggleChatDisplayMode = {
                                        updateChatDisplayMode(
                                            if (shellSettings.chatDisplayMode == "compact") "expanded" else "compact",
                                        )
                                    },
                                    onApplyProvider = ::applyProvider,
                                    onApplyModel = ::applyModel,
                                )
                            }

                            AppSection.Accounts -> {
                                val authViewModel: AuthViewModel = viewModel()
                                AuthScreen(
                                    modifier = Modifier.fillMaxSize(),
                                    viewModel = authViewModel,
                                    extraBottomSpacing = pageBottomClearance,
                                    onOpenSettings = { navigateToSection(AppSection.Settings) },
                                    onContextActionsChanged = ::setActions,
                                )
                            }

                            AppSection.NousPortal -> {
                                val portalViewModel: NousPortalViewModel = viewModel()
                                NousPortalScreen(
                                    modifier = Modifier.fillMaxSize(),
                                    viewModel = portalViewModel,
                                    extraBottomSpacing = pageBottomClearance,
                                    onContextActionsChanged = ::setActions,
                                )
                            }

                            AppSection.Device -> {
                                val deviceViewModel: DeviceViewModel = viewModel()
                                DeviceScreen(
                                    modifier = Modifier.fillMaxSize(),
                                    viewModel = deviceViewModel,
                                    extraBottomSpacing = pageBottomClearance,
                                    onContextActionsChanged = ::setActions,
                                )
                            }

                            AppSection.Settings -> {
                                val settingsViewModel: SettingsViewModel = viewModel()
                                LaunchedEffect(settingsViewModel) {
                                    settingsViewModel.reload()
                                }
                                SettingsScreen(
                                    modifier = Modifier.fillMaxSize(),
                                    viewModel = settingsViewModel,
                                    extraBottomSpacing = pageBottomClearance,
                                    onContextActionsChanged = ::setActions,
                                )
                            }
                        }
                    }
                }
                if (showNavigationDrawer) {
                    ShellNavigationDrawerOverlay(
                        currentSection = currentSection,
                        navigationSections = shellDrawerNavigationSections(),
                        actions = currentActions,
                        onDismiss = { showNavigationDrawer = false },
                        onSelectSection = ::navigateToSection,
                        onSelectAction = { action ->
                            showNavigationDrawer = false
                            action.onClick()
                        },
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

@Composable
private fun HermesTopBar(
    section: AppSection,
    bootUiState: BootUiState,
    onOpenNavigationMenu: () -> Unit,
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
                ShellTopBarDrawerButton(
                    onOpenNavigationMenu = onOpenNavigationMenu,
                )
                Image(
                    painter = painterResource(id = R.drawable.hermes_agent_fork_logo),
                    contentDescription = strings.hermesLogoDescription,
                    modifier = Modifier.size(34.dp),
                )
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = section.title(strings),
                        style = MaterialTheme.typography.titleLarge,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Text(
                        text = subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TopBarStatusBadge(
                        text = strings.alphaBadge,
                        containerColor = MaterialTheme.colorScheme.secondary,
                        contentColor = MaterialTheme.colorScheme.onSecondary,
                    )
                    TopBarStatusBadge(
                        text = strings.forkBadge(),
                        containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                        contentColor = MaterialTheme.colorScheme.onTertiaryContainer,
                    )
                }
            }
        }
    }
}

@Composable
private fun TopBarStatusBadge(
    text: String,
    containerColor: Color,
    contentColor: Color,
) {
    Surface(
        color = containerColor,
        shape = MaterialTheme.shapes.small,
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 6.dp),
            color = contentColor,
            style = MaterialTheme.typography.labelMedium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun ShellTopBarDrawerButton(
    onOpenNavigationMenu: () -> Unit,
) {
    IconButton(
        onClick = onOpenNavigationMenu,
        modifier = Modifier
            .size(40.dp)
            .semantics { contentDescription = "Open navigation menu" }
            .testTag("HermesShellDrawerButton"),
    ) {
        ShellHamburgerMenuIcon()
    }
}

@Composable
private fun ShellHamburgerMenuIcon() {
    Column(
        modifier = Modifier.width(18.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        repeat(3) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(2.dp)
                    .clip(RoundedCornerShape(1.dp))
                    .background(MaterialTheme.colorScheme.primary),
            )
        }
    }
}

@Composable
private fun ShellNavigationDrawerOverlay(
    currentSection: AppSection,
    navigationSections: List<AppSection>,
    actions: List<ShellActionItem>,
    onDismiss: () -> Unit,
    onSelectSection: (AppSection) -> Unit,
    onSelectAction: (ShellActionItem) -> Unit,
) {
    val strings = LocalHermesStrings.current
    Box(
        modifier = Modifier
            .fillMaxSize()
            .zIndex(10f)
            .testTag("HermesNavigationDrawerOverlay"),
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.42f))
                .clickable(onClick = onDismiss)
                .testTag("HermesNavigationDrawerScrim"),
        )
        Surface(
            modifier = Modifier
                .align(Alignment.TopStart)
                .statusBarsPadding()
                .padding(start = 16.dp, top = 64.dp, end = 16.dp)
                .widthIn(min = 260.dp, max = 340.dp)
                .testTag("HermesShellDrawerMenu"),
            color = MaterialTheme.colorScheme.surface,
            shape = MaterialTheme.shapes.medium,
            tonalElevation = 8.dp,
            shadowElevation = 8.dp,
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(10.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 10.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Image(
                        painter = painterResource(id = R.drawable.hermes_agent_fork_logo),
                        contentDescription = null,
                        modifier = Modifier.size(30.dp),
                    )
                    Text(
                        text = "Hermes Fork",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                HorizontalDivider()
                navigationSections.forEach { section ->
                    ShellNavigationDrawerItem(
                        label = section.navigationLabel(strings),
                        iconRes = section.iconRes,
                        selected = section == currentSection,
                        testTag = "HermesNav${section.name}",
                        onClick = { onSelectSection(section) },
                    )
                }
                if (actions.isNotEmpty()) {
                    HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
                    actions.forEachIndexed { index, action ->
                        ShellNavigationDrawerItem(
                            label = action.label,
                            iconRes = action.iconRes,
                            selected = false,
                            testTag = "HermesShellDrawerAction$index",
                            onClick = { onSelectAction(action) },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ShellNavigationDrawerItem(
    label: String,
    iconRes: Int,
    selected: Boolean,
    testTag: String,
    onClick: () -> Unit,
) {
    val containerColor = if (selected) {
        MaterialTheme.colorScheme.primary.copy(alpha = 0.16f)
    } else {
        Color.Transparent
    }
    val contentColor = if (selected) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.onSurface
    }
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.small)
            .clickable(onClick = onClick)
            .testTag(testTag),
        color = containerColor,
        shape = MaterialTheme.shapes.small,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                painter = painterResource(id = iconRes),
                contentDescription = null,
                tint = contentColor,
                modifier = Modifier.size(22.dp),
            )
            Text(
                text = label,
                color = contentColor,
                style = MaterialTheme.typography.bodyLarge,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun SectionWarmupPane(
    section: AppSection,
    modifier: Modifier = Modifier,
) {
    val strings = LocalHermesStrings.current
    Box(
        modifier = modifier.padding(24.dp),
        contentAlignment = Alignment.TopCenter,
    ) {
        Text(
            text = section.title(strings),
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onBackground,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
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
                    description = strings.accountsActionDescription(),
                    iconRes = R.drawable.ic_nav_accounts,
                    onClick = onOpenAccounts,
                ),
                ShellActionItem(
                    label = strings.settings.ifBlank { "Settings" },
                    description = strings.settingsActionDescription(),
                    iconRes = R.drawable.ic_nav_settings,
                    onClick = onOpenSettings,
                ),
                // label = "Provider Portal"
                ShellActionItem(
                    label = strings.portalTitle.ifBlank { "Provider Portal" },
                    description = strings.portalActionDescription(),
                    iconRes = R.drawable.ic_nav_portal,
                    onClick = onOpenPortal,
                ),
                // label = "Device"
                ShellActionItem(
                    label = strings.sectionDevice.ifBlank { "Device" },
                    description = strings.deviceActionDescription(),
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
            painter = painterResource(id = R.drawable.hermes_agent_fork_logo),
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
            Text(strings.retryHermes())
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
                Text(strings.gettingStartedTitle(), style = MaterialTheme.typography.titleMedium)
                Text(strings.gettingStartedStep(1))
                Text(strings.gettingStartedStep(2))
                Text(strings.gettingStartedStep(3))
                Text(strings.gettingStartedStep(4))
            }
        }
    }
}
