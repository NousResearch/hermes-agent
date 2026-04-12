from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_hermes_home_includes_getting_started_actions():
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")

    assert 'Text("Getting started"' in app_shell
    assert 'Accounts: connect ChatGPT, Claude, Gemini, email, phone, or Google.' in app_shell
    assert 'Device: grant shared-folder access if you want Hermes to edit real mobile files directly.' in app_shell
    assert 'Hermes chat: use voice input, chat commands, or the cog button' in app_shell
    assert 'label = "Nous Portal"' in app_shell
    assert 'label = "Device"' in app_shell


def test_settings_screen_includes_new_user_guidance():
    settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")

    assert 'Text(strings.settingsNewHereTitle' in settings
    assert 'Text(strings.settingsHelpAccounts)' in settings
    assert 'Text(strings.currentProviderProfile(providerLabel))' in settings
    assert 'strings.apiKeyHelp()' in settings
    assert 'New here?' in strings
    assert 'Use Accounts if you want Corr3xt-based sign-in flows' in strings
    assert 'Choose the provider you want Hermes to call directly.' in strings
    assert 'Paste the key for the selected provider, then tap Save' in strings
    assert 'rememberScrollState()' in settings
    assert 'verticalScroll(' in settings


def test_portal_screen_auto_loads_and_uses_contextual_actions():
    portal = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/portal/NousPortalScreen.kt").read_text(encoding="utf-8")

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
