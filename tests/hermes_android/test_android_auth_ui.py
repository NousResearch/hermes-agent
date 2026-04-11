from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_app_shell_has_accounts_tab_and_auth_screen():
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")

    assert 'Accounts("Accounts")' in app_shell
    assert 'AppSection.Accounts -> AuthScreen' in app_shell


def test_auth_screen_lists_requested_sign_in_methods():
    auth_models = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/AuthModels.kt").read_text(encoding="utf-8")
    auth_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthScreen.kt").read_text(encoding="utf-8")

    for label in ["Email", "Google", "Phone", "ChatGPT", "Claude", "Gemini"]:
        assert label in auth_models
    assert 'Corr3xt auth base URL' in auth_screen
    assert 'Sign in' in auth_screen


def test_main_activity_and_manifest_handle_auth_callbacks():
    main_activity = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/MainActivity.kt").read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "android/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")

    assert 'consumeAuthCallback' in main_activity
    assert 'AuthRuntimeApplier.apply' in main_activity
    assert 'android.intent.action.VIEW' in manifest
    assert 'android:scheme="hermesagent"' in manifest
    assert 'android:host="auth"' in manifest
    assert 'android:pathPrefix="/callback"' in manifest


def test_provider_presets_include_chatgpt_claude_and_gemini():
    presets = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/ProviderPresets.kt").read_text(encoding="utf-8")

    assert 'id = "chatgpt-web"' in presets
    assert 'id = "anthropic"' in presets
    assert 'id = "gemini"' in presets
