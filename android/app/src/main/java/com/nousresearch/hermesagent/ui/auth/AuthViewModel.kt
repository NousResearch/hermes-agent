package com.nousresearch.hermesagent.ui.auth

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nousresearch.hermesagent.auth.AuthRuntimeApplier
import com.nousresearch.hermesagent.auth.Corr3xtAuthClient
import com.nousresearch.hermesagent.data.AppSettings
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.AuthCatalog
import com.nousresearch.hermesagent.data.AuthOption
import com.nousresearch.hermesagent.data.AuthScope
import com.nousresearch.hermesagent.data.AuthSession
import com.nousresearch.hermesagent.data.AuthSessionStore
import com.nousresearch.hermesagent.data.PendingAuthRequest
import com.nousresearch.hermesagent.data.ProviderPresets
import com.nousresearch.hermesagent.data.ProviderSetupTarget
import com.nousresearch.hermesagent.device.HermesExternalBrowserLauncher
import com.nousresearch.hermesagent.ui.i18n.AppLanguage
import com.nousresearch.hermesagent.ui.i18n.HermesStrings
import com.nousresearch.hermesagent.ui.i18n.hermesStringsFor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

data class AuthOptionUiState(
    val id: String,
    val label: String,
    val description: String,
    val scope: AuthScope,
    val runtimeProvider: String = "",
    val signedIn: Boolean = false,
    val status: String = "Not signed in",
    val accountHint: String = "",
    val supportsApiKeySetup: Boolean = false,
    val supportsBrowserSignIn: Boolean = true,
    val browserSignInEnabled: Boolean = true,
    val providerSetupUrl: String = "",
)

data class AuthUiState(
    val corr3xtBaseUrl: String = "",
    val corr3xtConfigured: Boolean = false,
    val globalStatus: String = "Configure a reachable Corr3xt URL for app sign-in; providers use secure API keys or tokens in Settings.",
    val pendingMethodLabel: String = "",
    val hasPendingRequest: Boolean = false,
    val apiKeyFallbackMethodId: String = "",
    val apiKeyFallbackLabel: String = "",
    val pendingStartUrl: String = "",
    val options: List<AuthOptionUiState> = emptyList(),
)

class AuthViewModel(application: Application) : AndroidViewModel(application) {
    private val appSettingsStore = AppSettingsStore(application)
    private val authSessionStore = AuthSessionStore(application)
    private val providerSetupOpenIndexes = mutableMapOf<String, Int>()
    private val signedOutStatuses by lazy {
        buildSet {
            add("Not signed in")
            AppLanguage.entries.forEach { language ->
                add(hermesStringsFor(language).authNotSignedIn())
            }
        }
    }

    private fun currentStrings(): HermesStrings {
        val settings = appSettingsStore.load()
        return hermesStringsFor(AppLanguage.fromTag(settings.languageTag))
    }

    private val _uiState = MutableStateFlow(buildState())
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    fun refresh() {
        _uiState.value = buildState()
    }

    fun updateCorr3xtBaseUrl(value: String) {
        _uiState.update { it.copy(corr3xtBaseUrl = value) }
    }

    fun saveCorr3xtBaseUrl() {
        val candidate = _uiState.value.corr3xtBaseUrl.trim()
        if (candidate.isBlank()) {
            _uiState.update {
                it.copy(globalStatus = currentStrings().authConfigureCorr3xtFirst())
            }
            return
        }
        val normalized = Corr3xtAuthClient.normalizeConfiguredBaseUrl(candidate)
        if (normalized == null) {
            _uiState.update {
                it.copy(globalStatus = currentStrings().authBaseUrlMustBeValid())
            }
            return
        }

        val existing = appSettingsStore.load()
        appSettingsStore.save(
            AppSettings(
                provider = existing.provider,
                baseUrl = existing.baseUrl,
                model = existing.model,
                corr3xtBaseUrl = normalized,
                dataSaverMode = existing.dataSaverMode,
                onDeviceBackend = existing.onDeviceBackend,
                languageTag = existing.languageTag,
            )
        )
        _uiState.update {
            it.copy(
                corr3xtBaseUrl = normalized,
                corr3xtConfigured = true,
                globalStatus = currentStrings().authSavedBaseUrl(),
            )
        }
    }

    fun startAuth(methodId: String): Boolean {
        val option = AuthCatalog.find(methodId) ?: return false
        if (!option.browserSignInSupported && option.scope == AuthScope.RuntimeProvider) {
            prepareApiKeySetup(methodId)
            openProviderSetupPage(methodId)
            return true
        }
        val candidateBaseUrl = _uiState.value.corr3xtBaseUrl.trim()
        if (candidateBaseUrl.isBlank()) {
            _uiState.update {
                it.copy(globalStatus = currentStrings().authConfigureCorr3xtFirst())
            }
            return false
        }
        val normalizedBaseUrl = Corr3xtAuthClient.normalizeConfiguredBaseUrl(candidateBaseUrl)
        if (normalizedBaseUrl == null) {
            _uiState.update {
                it.copy(globalStatus = currentStrings().authBaseUrlMustBeValid())
            }
            return false
        }

        val settings = appSettingsStore.load()
        val state = UUID.randomUUID().toString()
        val pendingRequest = PendingAuthRequest(
            state = state,
            methodId = option.id,
            startUrl = Corr3xtAuthClient.buildStartUri(
                baseUrl = normalizedBaseUrl,
                option = option,
                state = state,
                languageTag = settings.languageTag,
            ).toString(),
        )
        val startUri = Uri.parse(pendingRequest.startUrl)

        viewModelScope.launch {
            _uiState.update { it.copy(globalStatus = currentStrings().authCheckingCorr3xt(option.label)) }
            val probe = withContext(Dispatchers.IO) {
                Corr3xtAuthClient.probeStartUri(android.net.Uri.parse(pendingRequest.startUrl))
            }
            if (!probe.reachable) {
                authSessionStore.clearPendingRequest()
                val apiKeyFallbackAvailable = option.scope == AuthScope.RuntimeProvider &&
                    option.runtimeProvider.isNotBlank()
                val failureStatus = when (probe.status) {
                    "unknown_host" -> if (option.scope == AuthScope.AppAccount) {
                        currentStrings().authAppSignInHostCouldNotBeResolved(probe.host)
                    } else {
                        currentStrings().authHostCouldNotBeResolved(probe.host)
                    }
                    "network_error" -> if (option.scope == AuthScope.AppAccount) {
                        currentStrings().authAppSignInPageCouldNotBeReached(probe.errorName)
                    } else {
                        currentStrings().authPageCouldNotBeReached(probe.errorName)
                    }
                    else -> probe.status.ifBlank { currentStrings().authTryAgain() }
                }.let { status ->
                    if (apiKeyFallbackAvailable) {
                        "$status ${currentStrings().authApiKeyFallbackAvailable(option.label)}"
                    } else {
                        status
                    }
                }
                _uiState.update {
                    it.copy(
                        corr3xtBaseUrl = normalizedBaseUrl,
                        globalStatus = failureStatus,
                        pendingMethodLabel = "",
                        hasPendingRequest = false,
                        pendingStartUrl = "",
                        apiKeyFallbackMethodId = if (apiKeyFallbackAvailable) option.id else "",
                        apiKeyFallbackLabel = if (apiKeyFallbackAvailable) option.label else "",
                    )
                }
                return@launch
            }

            authSessionStore.savePendingRequest(pendingRequest)
            val launch = HermesExternalBrowserLauncher.open(
                context = getApplication(),
                uri = startUri,
                title = "Open ${option.label} sign-in",
            )
            if (launch.success) {
                _uiState.update { current ->
                    current.copy(
                        corr3xtBaseUrl = normalizedBaseUrl,
                        globalStatus = currentStrings().authOpenedCorr3xt(option.label),
                        pendingMethodLabel = option.label,
                        hasPendingRequest = true,
                        pendingStartUrl = pendingRequest.startUrl,
                        apiKeyFallbackMethodId = "",
                        apiKeyFallbackLabel = "",
                    )
                }
            } else {
                authSessionStore.clearPendingRequest()
                copyAuthStartUrl(pendingRequest.startUrl, updateStatus = false)
                val statusPrefix = if (launch.errorName == "ActivityNotFoundException") {
                    currentStrings().authNoBrowser()
                } else {
                    "${currentStrings().authTryAgain()} (${launch.errorName.ifBlank { "browser_error" }})"
                }
                _uiState.update {
                    it.copy(
                        globalStatus = "$statusPrefix ${currentStrings().authCopiedSignInUrl()}",
                        pendingStartUrl = pendingRequest.startUrl,
                        apiKeyFallbackMethodId = if (option.scope == AuthScope.RuntimeProvider) option.id else "",
                        apiKeyFallbackLabel = if (option.scope == AuthScope.RuntimeProvider) option.label else "",
                    )
                }
            }
        }
        return true
    }

    fun copyPendingSignInUrl() {
        val startUrl = _uiState.value.pendingStartUrl.ifBlank {
            authSessionStore.loadPendingRequest()?.startUrl.orEmpty()
        }
        copyAuthStartUrl(startUrl, updateStatus = true)
    }

    private fun copyAuthStartUrl(startUrl: String, updateStatus: Boolean) {
        val target = startUrl.trim()
        if (target.isBlank()) {
            return
        }
        val clipboard = getApplication<Application>().getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
        clipboard?.setPrimaryClip(ClipData.newPlainText("Hermes Corr3xt sign-in URL", target))
        if (updateStatus) {
            _uiState.update { it.copy(globalStatus = currentStrings().authCopiedSignInUrl()) }
        }
    }

    fun prepareApiKeySetup(methodId: String) {
        val option = AuthCatalog.find(methodId) ?: return
        if (option.runtimeProvider.isBlank()) {
            return
        }
        val existing = appSettingsStore.load()
        appSettingsStore.save(
            AppSettings(
                provider = option.runtimeProvider,
                baseUrl = option.defaultBaseUrl,
                model = option.defaultModel,
                corr3xtBaseUrl = existing.corr3xtBaseUrl,
                dataSaverMode = existing.dataSaverMode,
                onDeviceBackend = existing.onDeviceBackend,
                languageTag = existing.languageTag,
            )
        )
        _uiState.update {
            it.copy(
                globalStatus = currentStrings().authApiKeySetupReady(option.label),
                apiKeyFallbackMethodId = "",
                apiKeyFallbackLabel = "",
                pendingStartUrl = "",
            )
        }
    }

    fun openProviderSetupPage(methodId: String) {
        val option = AuthCatalog.find(methodId) ?: return
        val target = nextProviderSetupTarget(option.runtimeProvider) ?: return
        val uri = Uri.parse(target.url)
        if (uri.scheme !in setOf("http", "https")) {
            _uiState.update { it.copy(globalStatus = "Provider setup URL must start with https:// or http://") }
            return
        }
        val launch = HermesExternalBrowserLauncher.open(
            context = getApplication(),
            uri = uri,
            title = "Open ${option.label} setup page",
        )
        if (launch.success) {
            copyProviderSetupUrl(methodId, updateStatus = false)
            _uiState.update {
                it.copy(globalStatus = providerSetupOpenedStatus(option.label, option.runtimeProvider, target))
            }
        } else {
            copyProviderSetupUrl(methodId, updateStatus = false)
            _uiState.update {
                it.copy(globalStatus = "Unable to open browser (${launch.errorName.ifBlank { "browser_error" }}); copied the ${option.label} setup URLs.")
            }
        }
    }

    private fun nextProviderSetupTarget(providerId: String): ProviderSetupTarget? {
        val nextIndex = providerSetupOpenIndexes[providerId] ?: 0
        val target = ProviderPresets.setupTarget(providerId, nextIndex) ?: return null
        providerSetupOpenIndexes[providerId] = target.nextIndex
        return target
    }

    private fun providerSetupOpenedStatus(
        optionLabel: String,
        providerId: String,
        target: ProviderSetupTarget,
    ): String {
        val cycleHint = if (target.total > 1) {
            " ${target.displayIndex}/${target.total}; copied all official setup URLs. Tap Open again for the next fallback if this page stalls."
        } else {
            ". If your browser stalls, copy the setup URL and paste it into another browser."
        }
        val qwenLegacyHint = if (providerId == "qwen-oauth") {
            " Qwen OAuth is legacy; choose Qwen Cloud for new API-key setup."
        } else {
            ""
        }
        return "Opened $optionLabel setup page$cycleHint$qwenLegacyHint"
    }

    fun copyProviderSetupUrl(methodId: String) {
        copyProviderSetupUrl(methodId, updateStatus = true)
    }

    private fun copyProviderSetupUrl(methodId: String, updateStatus: Boolean) {
        val option = AuthCatalog.find(methodId) ?: return
        val setupText = ProviderPresets.setupClipboardText(option.runtimeProvider)
        if (setupText.isBlank()) {
            return
        }
        val clipboard = getApplication<Application>().getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
        clipboard?.setPrimaryClip(ClipData.newPlainText("Hermes ${option.label} setup URLs", setupText))
        if (updateStatus) {
            val fallbackCount = ProviderPresets.setupUrls(option.runtimeProvider).size - 1
            val suffix = when (fallbackCount) {
                0 -> ""
                1 -> " and 1 alternate official page"
                else -> " and $fallbackCount alternate official pages"
            }
            _uiState.update { it.copy(globalStatus = "Copied ${option.label} setup URL$suffix.") }
        }
    }

    fun cancelPendingRequest() {
        authSessionStore.clearPendingRequest()
        _uiState.update {
            it.copy(
                pendingMethodLabel = "",
                hasPendingRequest = false,
                apiKeyFallbackMethodId = "",
                apiKeyFallbackLabel = "",
                pendingStartUrl = "",
                globalStatus = currentStrings().authCanceled(),
            )
        }
    }

    fun signOut(methodId: String) {
        val session = authSessionStore.loadSession(methodId)
        authSessionStore.clearSession(methodId)
        if (session != null && session.runtimeProvider.isNotBlank()) {
            runCatching {
                val python = com.chaquo.python.Python.getInstance()
                python.getModule("hermes_android.auth_bridge")
                    .callAttr("clear_provider_auth_bundle", session.runtimeProvider)
            }
        }
        refresh()
    }

    private fun buildState(): AuthUiState {
        val settings = appSettingsStore.load()
        val strings = hermesStringsFor(AppLanguage.fromTag(settings.languageTag))
        val persistedPending = authSessionStore.loadPendingRequest()
        val pending = persistedPending?.takeUnless { AuthSessionStore.isPendingRequestExpired(it) }
        if (persistedPending != null && pending == null) {
            authSessionStore.clearPendingRequest()
        }

        val corr3xtBaseUrl = Corr3xtAuthClient.normalizedBaseUrl(settings.corr3xtBaseUrl)
        val corr3xtConfigured = corr3xtBaseUrl.isNotBlank()
        val sessions = authSessionStore.loadSessions()
        val sessionsById = sessions.associateBy { it.methodId }
        val options = AuthCatalog.options.map { option ->
            val session = sessionsById[option.id] ?: defaultSession(option)
            val localizedStatus = when {
                session.signedIn -> strings.authSignedInWith(option.label)
                isSignedOutStatus(session.status) -> strings.authNotSignedIn()
                else -> session.status
            }
            AuthOptionUiState(
                id = option.id,
                label = option.label,
                description = strings.authDescription(option.id, option.description),
                scope = option.scope,
                runtimeProvider = session.runtimeProvider,
                signedIn = session.signedIn,
                status = localizedStatus,
                supportsApiKeySetup = option.scope == AuthScope.RuntimeProvider && option.runtimeProvider.isNotBlank(),
                supportsBrowserSignIn = option.browserSignInSupported,
                accountHint = listOf(session.displayName, session.email, session.phone)
                    .firstOrNull { it.isNotBlank() }
                    .orEmpty(),
                browserSignInEnabled = option.scope != AuthScope.AppAccount || corr3xtConfigured,
                providerSetupUrl = ProviderPresets.find(option.runtimeProvider)?.apiKeyUrl.orEmpty(),
            )
        }
        val signedInAccounts = options.count { it.signedIn }
        val latestSessionStatus = sessions
            .filter { session ->
                session.updatedAtEpochMs > 0 &&
                    session.status.isNotBlank() &&
                    !isSignedOutStatus(session.status)
            }
            .maxByOrNull { it.updatedAtEpochMs }
            ?.status
        val pendingMethodLabel = pending?.methodId
            ?.let { AuthCatalog.find(it)?.label ?: it }
            .orEmpty()
        val globalStatus = when {
            pending != null -> strings.authWaitingCallback(pendingMethodLabel)
            !latestSessionStatus.isNullOrBlank() -> latestSessionStatus
            signedInAccounts > 0 -> strings.authConnectedMethods(signedInAccounts)
            !corr3xtConfigured -> strings.authConfigureCorr3xtFirst()
            else -> strings.authGlobalStatusDefault()
        }

        return AuthUiState(
            corr3xtBaseUrl = corr3xtBaseUrl,
            corr3xtConfigured = corr3xtConfigured,
            globalStatus = globalStatus,
            pendingMethodLabel = pendingMethodLabel,
            hasPendingRequest = pending != null,
            pendingStartUrl = pending?.startUrl.orEmpty(),
            apiKeyFallbackMethodId = "",
            apiKeyFallbackLabel = "",
            options = options,
        )
    }

    fun applyConsumedCallbackIfPresent() {
        val pending = authSessionStore.loadPendingRequest() ?: return
        val storedSession = authSessionStore.loadSession(pending.methodId) ?: return
        if (!storedSession.signedIn) {
            refresh()
            return
        }
        AuthRuntimeApplier.apply(getApplication(), storedSession)
        authSessionStore.clearPendingRequest()
        refresh()
    }

    private fun defaultSession(option: AuthOption): AuthSession {
        return AuthSession(
            methodId = option.id,
            label = option.label,
            scope = option.scope,
            runtimeProvider = option.runtimeProvider,
            status = currentStrings().authNotSignedIn(),
            updatedAtEpochMs = 0,
        )
    }

    private fun isSignedOutStatus(status: String): Boolean {
        return status.trim() in signedOutStatuses
    }
}
