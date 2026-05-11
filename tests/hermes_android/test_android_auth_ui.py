from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_app_shell_has_accounts_tab_and_auth_screen():
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")
    shell_models = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/ShellModels.kt").read_text(encoding="utf-8")

    assert 'Accounts(' in shell_models
    assert 'label = "Accounts"' in shell_models
    accounts_branch = app_shell.split("AppSection.Accounts -> {", 1)[1].split("AppSection.NousPortal ->", 1)[0]
    assert 'val authViewModel: AuthViewModel = viewModel()' in accounts_branch
    assert 'AuthScreen(' in accounts_branch


def test_auth_screen_lists_requested_sign_in_methods_and_pending_fallback_ui():
    auth_models = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/AuthModels.kt").read_text(encoding="utf-8")
    auth_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthScreen.kt").read_text(encoding="utf-8")
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")

    for label in ["Email", "Google", "Phone", "ChatGPT", "Claude", "Gemini", "Qwen Cloud", "Qwen OAuth", "Z.AI"]:
        assert label in auth_models
    assert 'Corr3xt auth base URL' in auth_screen
    assert 'Pending Corr3xt sign-in' in auth_screen
    assert 'strings.cancelPendingSignIn()' in auth_screen
    assert 'strings.authRefreshDescription()' in auth_screen
    assert 'strings.authCancelPendingDescription()' in auth_screen
    assert 'strings.authWaitingCallbackFor(uiState.pendingMethodLabel)' in auth_screen
    assert 'viewModel::copyPendingSignInUrl' in auth_screen
    assert 'strings.copyAuthSignInUrl()' in auth_screen
    assert 'LaunchedEffect(strings.language)' in auth_screen
    assert 'secure callback' in auth_screen
    assert 'Sign in' in auth_screen
    assert 'option.supportsApiKeySetup' in auth_screen
    assert 'option.supportsBrowserSignIn' in auth_screen
    assert 'strings.useApiKeyInSettings()' in auth_screen
    assert 'strings.setUpApiKeyFor(option.label)' in auth_screen
    assert 'option.providerSetupUrl.isNotBlank()' in auth_screen
    assert 'viewModel.openProviderSetupPage(option.id)' in auth_screen
    assert 'viewModel.copyProviderSetupUrl(option.id)' in auth_screen
    assert 'strings.openProviderKeyPage(option.label)' in auth_screen
    assert 'strings.copyProviderSetupUrl()' in auth_screen
    assert 'AuthProviderCopySetup-${option.id}' in auth_screen
    assert 'viewModel.prepareApiKeySetup(option.id)' in auth_screen
    assert 'onOpenSettings()' in auth_screen
    assert 'FlowRow' in auth_screen
    assert 'settingsViewModel.reload()' in app_shell
    assert 'fun reload()' in settings_view_model
    assert 'extraBottomSpacing' in auth_screen


def test_main_activity_and_manifest_handle_auth_callbacks():
    main_activity = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/MainActivity.kt").read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "android/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")

    assert 'consumeAuthCallback' in main_activity
    assert 'AuthRuntimeApplier.apply' in main_activity
    assert 'android.intent.action.VIEW' in manifest
    assert 'android:scheme="hermesagent"' in manifest
    assert 'android:host="auth"' in manifest
    assert 'android:pathPrefix="/callback"' in manifest
    assert 'android:resizeableActivity="true"' in manifest


def test_provider_presets_include_chatgpt_claude_gemini_qwen_and_zai():
    presets = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/ProviderPresets.kt").read_text(encoding="utf-8")

    assert 'id = "chatgpt-web"' in presets
    assert 'id = "anthropic"' in presets
    assert 'id = "gemini"' in presets
    assert 'id = "alibaba"' in presets
    assert 'id = "qwen-oauth"' in presets
    assert 'id = "zai"' in presets
    assert 'apiKeyUrl = "https://openrouter.ai/keys"' in presets
    assert 'apiKeyUrl = "https://platform.openai.com/settings/organization/api-keys"' in presets
    assert 'apiKeyUrl = "https://home.qwencloud.com/api-keys"' in presets
    assert 'apiKeyUrl = "https://z.ai/manage-apikey/apikey-list"' in presets
    assert 'fallbackSetupUrls = listOf(' in presets
    assert 'https://docs.qwencloud.com/api-reference/preparation/api-key' in presets
    assert 'https://docs.z.ai/guides/' in presets
    assert 'fun setupClipboardText(providerId: String): String' in presets
    assert 'fun providerIdForSetupUrl(url: String): String?' in presets
    assert 'fun runtimeConfigBaseUrl(providerId: String, baseUrl: String): String' in presets
    assert 'providerId == "zai" && normalized == presetDefault -> ""' in presets


def test_auth_callback_hardening_strings_and_base_url_validation_exist():
    auth_session_store = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/AuthSessionStore.kt").read_text(encoding="utf-8")
    auth_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthViewModel.kt").read_text(encoding="utf-8")
    browser_launcher = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesExternalBrowserLauncher.kt").read_text(encoding="utf-8")
    corr3xt_auth_client = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/auth/Corr3xtAuthClient.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")

    assert 'Auth callback rejected: no pending sign-in request' in auth_session_store
    assert 'Auth callback expired. Start sign-in again.' in auth_session_store
    assert 'Auth callback rejected: method mismatch' in auth_session_store
    assert 'Auth callback rejected: provider mismatch' in auth_session_store
    assert 'Auth callback rejected: no provider credentials were returned' in auth_session_store
    assert 'Auth callback rejected: no account identity returned' in auth_session_store
    assert 'currentStrings().authBaseUrlMustBeValid()' in auth_view_model
    assert 'currentStrings().authConfigureCorr3xtFirst()' in auth_view_model
    assert 'currentStrings().authSavedBaseUrl()' in auth_view_model
    assert 'viewModelScope.launch' in auth_view_model
    assert 'Corr3xtAuthClient.probeStartUri' in auth_view_model
    assert 'currentStrings().authCheckingCorr3xt(option.label)' in auth_view_model
    assert 'authAppSignInHostCouldNotBeResolved' in auth_view_model
    assert 'authAppSignInPageCouldNotBeReached' in auth_view_model
    assert 'currentStrings().authHostCouldNotBeResolved(probe.host)' in auth_view_model
    assert 'currentStrings().authPageCouldNotBeReached(probe.errorName)' in auth_view_model
    assert 'fun prepareApiKeySetup' in auth_view_model
    assert 'if (!option.browserSignInSupported && option.scope == AuthScope.RuntimeProvider)' in auth_view_model
    assert 'authApiKeySetupReady(option.label)' in auth_view_model
    assert 'currentStrings().authOpenedCorr3xt(option.label)' in auth_view_model
    assert 'HermesExternalBrowserLauncher.open' in auth_view_model
    assert 'Intent.createChooser' in browser_launcher
    assert 'putExtra(Browser.EXTRA_APPLICATION_ID' in browser_launcher
    assert 'copyAuthStartUrl(pendingRequest.startUrl, updateStatus = false)' in auth_view_model
    assert 'fun copyPendingSignInUrl()' in auth_view_model
    assert 'ClipData.newPlainText("Hermes Corr3xt sign-in URL", target)' in auth_view_model
    assert 'currentStrings().authNoBrowser()' in auth_view_model
    assert 'addCategory(Intent.CATEGORY_BROWSABLE)' in browser_launcher
    assert 'pendingStartUrl = pending?.startUrl.orEmpty()' in auth_view_model
    assert 'authBaseUrlMustBeValid' in strings
    assert 'authConfigureCorr3xtFirst' in strings
    assert 'Configure a reachable Corr3xt URL to enable app sign-in' in strings
    assert 'authOpenedCorr3xt' in strings
    assert 'copyAuthSignInUrl' in strings
    assert 'authCopiedSignInUrl' in strings
    assert 'Copy sign-in URL' in strings
    assert 'Copied sign-in URL.' in strings
    assert 'If your browser stalls, copy the sign-in URL' in strings
    assert 'authCheckingCorr3xt' in strings
    assert 'authHostCouldNotBeResolved' in strings
    assert 'authPageCouldNotBeReached' in strings
    assert 'authAppSignInHostCouldNotBeResolved' in strings
    assert 'App sign-in is unavailable until a reachable Corr3xt URL is set' in strings
    assert 'Use API key in Settings' in strings
    assert 'secure API-key setup' in strings
    assert 'Unable to open Corr3xt: no browser is available' in strings
    assert 'callback_contract' in corr3xt_auth_client
    assert 'ui_locales' in corr3xt_auth_client
    assert 'locale' in corr3xt_auth_client
    assert 'lang' in corr3xt_auth_client
    assert 'normalizeConfiguredBaseUrl' in corr3xt_auth_client
    assert 'throw IllegalArgumentException("Corr3xt base URL is not configured")' in corr3xt_auth_client
    assert 'probeStartUri' in corr3xt_auth_client
    assert 'probeHttpUri(probeUri, host, timeoutMs)' in corr3xt_auth_client
    assert 'probeHttpUri(uri, host, timeoutMs)' in corr3xt_auth_client
    assert 'status = "query_required"' in corr3xt_auth_client
    assert 'UnknownHostException' in corr3xt_auth_client
    assert 'status = "unknown_host"' in corr3xt_auth_client
    assert 'status = "network_error"' in corr3xt_auth_client
    assert 'encodedQuery(null)' in corr3xt_auth_client


def test_runtime_provider_accounts_use_key_setup_instead_of_dead_corr3xt_default():
    auth_models = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/AuthModels.kt").read_text(encoding="utf-8")
    auth_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthScreen.kt").read_text(encoding="utf-8")
    auth_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthViewModel.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")

    provider_presets = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/ProviderPresets.kt").read_text(encoding="utf-8")

    for provider in ["openrouter", "chatgpt", "claude", "gemini", "qwen", "qwen-oauth", "zai"]:
        block = auth_models.split(f'id = "{provider}"', 1)[1].split("AuthOption(", 1)[0]
        assert "browserSignInSupported = false" in block

    qwen_block = auth_models.split('id = "qwen"', 1)[1].split("AuthOption(", 1)[0]
    assert 'runtimeProvider = "alibaba"' in qwen_block
    assert 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1' in qwen_block
    qwen_oauth_block = auth_models.split('id = "qwen-oauth"', 1)[1].split("AuthOption(", 1)[0]
    assert 'runtimeProvider = "qwen-oauth"' in qwen_oauth_block
    assert 'https://portal.qwen.ai/v1' in qwen_oauth_block
    assert "if (option.supportsBrowserSignIn)" in auth_screen
    assert "enabled = option.browserSignInEnabled" in auth_screen
    assert "browserSignInEnabled = option.scope != AuthScope.AppAccount || corr3xtConfigured" in auth_view_model
    assert "providerSetupUrl = ProviderPresets.find(option.runtimeProvider)?.apiKeyUrl.orEmpty()" in auth_view_model
    assert "fun openProviderSetupPage(methodId: String)" in auth_view_model
    assert "prepareApiKeySetup(methodId)\n            openProviderSetupPage(methodId)" in auth_view_model
    assert "HermesExternalBrowserLauncher.open" in auth_view_model
    assert "fun copyProviderSetupUrl(methodId: String)" in auth_view_model
    assert "ProviderPresets.setupClipboardText(option.runtimeProvider)" in auth_view_model
    assert 'ClipData.newPlainText("Hermes ${option.label} setup URLs", setupText)' in auth_view_model
    assert '" and 1 alternate official page"' in auth_view_model
    assert '" and $fallbackCount alternate official pages"' in auth_view_model
    assert "strings.setUpApiKeyFor(option.label)" in auth_screen
    assert "prepareApiKeySetup(methodId)" in auth_view_model
    assert "providers use secure API keys or tokens in Settings" in strings
    assert "Qwen OAuth / Qwen Chat token" in provider_presets


def test_settings_opens_official_provider_key_pages():
    settings_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    browser_launcher = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesExternalBrowserLauncher.kt").read_text(encoding="utf-8")
    provider_presets = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/ProviderPresets.kt").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt").read_text(encoding="utf-8")

    assert "providerPreset?.apiKeyUrl" in settings_screen
    assert "viewModel::openProviderKeyPage" in settings_screen
    assert "viewModel::copyProviderKeyPage" in settings_screen
    assert "Intent.ACTION_VIEW" in browser_launcher
    assert "Uri.parse(target)" in settings_view_model
    assert "HermesExternalBrowserLauncher.open" in settings_view_model
    assert "Intent.createChooser" in browser_launcher
    assert "putExtra(Browser.EXTRA_APPLICATION_ID" in browser_launcher
    assert "ClipboardManager" in settings_view_model
    assert "ClipData.newPlainText" in settings_view_model
    assert "ProviderPresets.providerIdForSetupUrl(target)" in settings_view_model
    assert "ProviderPresets.setupClipboardText(it)" in settings_view_model
    assert 'ClipData.newPlainText("Hermes provider setup URLs", setupText)' in settings_view_model
    assert "addCategory(Intent.CATEGORY_BROWSABLE)" in browser_launcher
    assert "openProviderKeyPage(providerLabel)" in settings_screen
    assert "copyProviderSetupUrl()" in settings_screen
    assert "importSavedProviderCredential()" in settings_screen
    assert "Use saved Hermes credential" in strings
    assert "Open $providerLabel setup page" in strings
    assert "Copy setup URL" in strings
    assert "ProviderPresets.androidSettingsDefaults.forEach" in settings_screen
    assert "androidSettingsDefaults = defaults" in provider_presets
    assert "PasswordVisualTransformation()" in settings_screen
    assert "KeyboardType.Password" in settings_screen


def test_settings_can_import_saved_python_provider_credentials_without_blank_overwrite():
    settings_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    auth_bridge = (REPO_ROOT / "hermes_android/auth_bridge.py").read_text(encoding="utf-8")

    assert "onImportProviderCredential = viewModel::importSavedProviderCredential" in settings_screen
    assert "status = uiState.status" in settings_screen
    assert "if (status.isNotBlank())" in settings_screen
    assert "fun importSavedProviderCredential()" in settings_view_model
    assert "read_provider_auth_bundle_json" in settings_view_model
    assert "HermesRuntimeManager.ensurePythonStarted(app)" in settings_view_model
    assert "secretsStore.saveApiKey(snapshot.provider, apiKey)" in settings_view_model
    assert "val providerApiKey = snapshot.apiKey.trim()" in settings_view_model
    assert "if (providerApiKey.isNotBlank())" in settings_view_model
    assert "Blank API key field left existing Hermes credentials untouched" in settings_view_model
    assert "write_provider_auth_bundle" in settings_view_model
    assert "write_runtime_config" in settings_view_model
    assert "No saved Hermes credential found for $providerLabel" in settings_view_model
    assert "Imported saved Hermes credential for $providerLabel" in settings_view_model
    assert "def read_provider_auth_bundle_json(provider: str) -> str:" in auth_bridge
    assert '"reason": "blank_api_key_preserved"' in auth_bridge
    assert '"zai": {' in auth_bridge
    assert 'if normalized == "qwen-oauth":' in auth_bridge


def test_settings_provider_switch_applies_selected_provider_defaults():
    settings_view_model = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt").read_text(encoding="utf-8")
    auth_runtime_applier = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/auth/AuthRuntimeApplier.kt").read_text(encoding="utf-8")
    runtime_manager = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/HermesRuntimeManager.kt").read_text(encoding="utf-8")

    assert "val providerChanged = provider != it.provider" in settings_view_model
    assert 'baseUrl = if (providerChanged && provider != "custom") preset?.baseUrl.orEmpty() else it.baseUrl' in settings_view_model
    assert 'model = if (providerChanged && provider != "custom") preset?.modelHint.orEmpty() else it.model' in settings_view_model
    assert 'ProviderPresets.runtimeConfigBaseUrl(snapshot.provider, snapshot.baseUrl)' in settings_view_model
    assert 'val runtimeConfigBaseUrl = ProviderPresets.runtimeConfigBaseUrl(session.runtimeProvider, resolvedBaseUrl)' in auth_runtime_applier
    assert 'runtimeConfigBaseUrl,' in auth_runtime_applier
    assert 'import com.nousresearch.hermesagent.data.ProviderPresets' in runtime_manager
    assert 'ProviderPresets.runtimeConfigBaseUrl(settings.provider, settings.baseUrl)' in runtime_manager


def test_android_wheel_task_tracks_python_auth_sources():
    build_gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")

    assert 'tasks.register<Exec>("prepareHermesAndroidWheel")' in build_gradle
    assert 'inputs.file(repoRoot.resolve("pyproject.toml"))' in build_gradle
    assert 'inputs.files(fileTree(repoRoot.resolve(packageDir))' in build_gradle
    assert '"hermes_android"' in build_gradle
    assert '"hermes_cli"' in build_gradle
    assert 'include("**/*.py")' in build_gradle
