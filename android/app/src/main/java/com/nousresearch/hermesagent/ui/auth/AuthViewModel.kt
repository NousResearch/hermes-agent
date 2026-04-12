package com.nousresearch.hermesagent.ui.auth

import android.app.Application
import android.content.ActivityNotFoundException
import android.content.Intent
import androidx.lifecycle.AndroidViewModel
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
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
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
)

data class AuthUiState(
    val corr3xtBaseUrl: String = Corr3xtAuthClient.DEFAULT_BASE_URL,
    val globalStatus: String = "Use Corr3xt to sign into the app or connect provider accounts.",
    val pendingMethodLabel: String = "",
    val hasPendingRequest: Boolean = false,
    val options: List<AuthOptionUiState> = emptyList(),
)

class AuthViewModel(application: Application) : AndroidViewModel(application) {
    private val appSettingsStore = AppSettingsStore(application)
    private val authSessionStore = AuthSessionStore(application)

    private val _uiState = MutableStateFlow(buildState())
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    fun refresh() {
        _uiState.value = buildState()
    }

    fun updateCorr3xtBaseUrl(value: String) {
        _uiState.update { it.copy(corr3xtBaseUrl = value) }
    }

    fun saveCorr3xtBaseUrl() {
        val normalized = Corr3xtAuthClient.normalizeConfiguredBaseUrl(_uiState.value.corr3xtBaseUrl)
        if (normalized == null) {
            _uiState.update {
                it.copy(globalStatus = "Corr3xt base URL must be a valid http(s) URL")
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
                globalStatus = "Saved Corr3xt base URL",
            )
        }
    }

    fun startAuth(methodId: String) {
        val option = AuthCatalog.find(methodId) ?: return
        val normalizedBaseUrl = Corr3xtAuthClient.normalizeConfiguredBaseUrl(_uiState.value.corr3xtBaseUrl)
        if (normalizedBaseUrl == null) {
            _uiState.update {
                it.copy(globalStatus = "Corr3xt base URL must be a valid http(s) URL")
            }
            return
        }

        val state = UUID.randomUUID().toString()
        val pendingRequest = PendingAuthRequest(
            state = state,
            methodId = option.id,
            startUrl = Corr3xtAuthClient.buildStartUri(normalizedBaseUrl, option, state).toString(),
        )
        val browserIntent = Intent(Intent.ACTION_VIEW, android.net.Uri.parse(pendingRequest.startUrl)).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }

        authSessionStore.savePendingRequest(pendingRequest)
        try {
            getApplication<Application>().startActivity(browserIntent)
            _uiState.update { current ->
                current.copy(
                    corr3xtBaseUrl = normalizedBaseUrl,
                    globalStatus = "Opened Corr3xt for ${option.label} sign-in",
                    pendingMethodLabel = option.label,
                    hasPendingRequest = true,
                )
            }
        } catch (_: ActivityNotFoundException) {
            authSessionStore.clearPendingRequest()
            _uiState.update {
                it.copy(globalStatus = "Unable to open Corr3xt: no browser is available")
            }
        } catch (_: RuntimeException) {
            authSessionStore.clearPendingRequest()
            _uiState.update {
                it.copy(globalStatus = "Unable to open Corr3xt. Check the auth URL and try again.")
            }
        }
    }

    fun cancelPendingRequest() {
        authSessionStore.clearPendingRequest()
        _uiState.update {
            it.copy(
                pendingMethodLabel = "",
                hasPendingRequest = false,
                globalStatus = "Canceled pending Corr3xt sign-in",
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
        val persistedPending = authSessionStore.loadPendingRequest()
        val pending = persistedPending?.takeUnless { AuthSessionStore.isPendingRequestExpired(it) }
        if (persistedPending != null && pending == null) {
            authSessionStore.clearPendingRequest()
        }

        val corr3xtBaseUrl = Corr3xtAuthClient.normalizedBaseUrl(settings.corr3xtBaseUrl)
        val sessions = authSessionStore.loadSessions()
        val sessionsById = sessions.associateBy { it.methodId }
        val options = AuthCatalog.options.map { option ->
            val session = sessionsById[option.id] ?: defaultSession(option)
            AuthOptionUiState(
                id = option.id,
                label = option.label,
                description = option.description,
                scope = option.scope,
                runtimeProvider = session.runtimeProvider,
                signedIn = session.signedIn,
                status = session.status,
                accountHint = listOf(session.displayName, session.email, session.phone)
                    .firstOrNull { it.isNotBlank() }
                    .orEmpty(),
            )
        }
        val signedInAccounts = options.count { it.signedIn }
        val latestSessionStatus = sessions
            .filter { it.updatedAtEpochMs > 0 && it.status.isNotBlank() && it.status != "Not signed in" }
            .maxByOrNull { it.updatedAtEpochMs }
            ?.status
        val pendingMethodLabel = pending?.methodId
            ?.let { AuthCatalog.find(it)?.label ?: it }
            .orEmpty()
        val globalStatus = when {
            pending != null -> "Waiting for Corr3xt callback for $pendingMethodLabel"
            !latestSessionStatus.isNullOrBlank() -> latestSessionStatus
            signedInAccounts > 0 -> "$signedInAccounts sign-in methods connected"
            else -> "Use Corr3xt to sign into the app or connect provider accounts."
        }

        return AuthUiState(
            corr3xtBaseUrl = corr3xtBaseUrl,
            globalStatus = globalStatus,
            pendingMethodLabel = pendingMethodLabel,
            hasPendingRequest = pending != null,
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
            status = "Not signed in",
            updatedAtEpochMs = 0,
        )
    }
}
