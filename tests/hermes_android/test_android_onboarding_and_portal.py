from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_hermes_home_includes_getting_started_actions():
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")

    assert 'Text("Getting started"' in app_shell
    assert 'Accounts: connect ChatGPT, Claude, Gemini, email, phone, or Google.' in app_shell
    assert 'Settings: choose a provider, confirm the base URL/model, and save your API key.' in app_shell
    assert 'Nous Portal: open the full portal experience in your browser' in app_shell
    assert 'Text("Open Nous Portal")' in app_shell
    assert 'currentSection = AppSection.Accounts' in app_shell
    assert 'currentSection = AppSection.Settings' in app_shell
    assert 'currentSection = AppSection.NousPortal' in app_shell


def test_settings_screen_includes_new_user_guidance():
    settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")

    assert 'Text("New here?"' in settings
    assert 'Use Accounts if you want Corr3xt-based sign-in flows' in settings
    assert 'Choose the provider you want Hermes to call directly.' in settings
    assert 'Paste the key for the selected provider, then tap Save' in settings


def test_portal_screen_defaults_to_external_guidance_and_hardens_webview():
    portal = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/portal/NousPortalScreen.kt").read_text(encoding="utf-8")

    assert 'Text("Portal access"' in portal
    assert 'Best experience in alpha: open Nous Portal in your browser.' in portal
    assert 'Text("Open externally")' in portal
    assert 'Hide embedded preview' in portal
    assert 'Try embedded preview' in portal
    assert 'CookieManager.getInstance()' in portal
    assert 'setAcceptThirdPartyCookies' in portal
    assert 'PORTAL_EMBED_USER_AGENT' in portal
    assert 'Embedded preview is experimental in this alpha.' in portal
