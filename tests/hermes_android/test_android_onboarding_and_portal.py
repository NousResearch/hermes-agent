from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_hermes_home_includes_getting_started_actions():
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")

    assert 'Text(strings.gettingStartedTitle()' in app_shell
    assert 'Text(strings.gettingStartedStep(1))' in app_shell
    assert 'Text(strings.gettingStartedStep(4))' in app_shell
    assert 'label = "Provider Portal"' in app_shell
    assert 'label = "Device"' in app_shell


def test_settings_screen_includes_new_user_guidance():
    settings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")

    assert 'Text(strings.settingsNewHereTitle' in settings
    assert 'Text(strings.settingsHelpAccounts)' in settings
    assert 'Text(strings.currentProviderProfile(providerLabel))' in settings
    assert 'strings.apiKeyHelp()' in settings
    assert 'Hermes Agent Fork' in strings
    assert 'forkDisclosure()' in strings
    assert 'Getting started' in strings
    assert 'Hermes chat: use voice input, chat commands, or the cog button' in strings
    assert 'Use Accounts for Corr3xt app sign-in with email, phone, or Google' in strings
    assert 'Choose the provider you want Hermes to call directly.' in strings
    assert 'Paste the API key or access token for the selected provider, then tap Save' in strings
    assert 'rememberScrollState()' in settings
    assert 'verticalScroll(' in settings


def test_portal_screen_auto_loads_and_uses_contextual_actions():
    portal = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/portal/NousPortalScreen.kt").read_text(encoding="utf-8")

    assert 'onContextActionsChanged' in portal
    assert 'label = "Refresh portal"' in portal
    assert 'label = "Open externally"' in portal
    assert 'loadUrl(uiState.portalUrl)' in portal
    assert 'The embedded portal now auto-loads on this page.' in portal
    assert 'extraBottomSpacing' in portal
    assert 'Full screen portal' in portal
    assert 'Minimize portal' in portal
    assert 'Try embedded preview' not in portal
    assert 'Reload preview' not in portal


def test_portal_python_refresh_is_deferred_until_portal_is_visible():
    portal = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/portal/NousPortalScreen.kt").read_text(encoding="utf-8")
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/mobilefork/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")

    portal_view_model = portal.split("class NousPortalViewModel", 1)[1].split("@Composable", 1)[0]

    assert "init {" not in portal_view_model
    assert "HermesRuntimeManager.ensurePythonStarted(getApplication())" in portal
    assert "withContext(Dispatchers.IO)" in portal
    assert "LaunchedEffect(strings.language) {" in portal
    portal_branch = app_shell.split("AppSection.NousPortal -> {", 1)[1].split("AppSection.Device ->", 1)[0]

    assert "val portalViewModel: NousPortalViewModel = viewModel()" in portal_branch
    assert "val portalViewModel: NousPortalViewModel = viewModel()" not in app_shell.split("val settingsState", 1)[0]
