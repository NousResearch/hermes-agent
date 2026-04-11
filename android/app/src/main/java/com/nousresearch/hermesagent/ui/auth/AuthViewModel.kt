package com.nousresearch.hermesagent.ui.auth

import android.app.Application
import android.content.Intent
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
        val snapshot = _uiState.value
        val existing = appSettingsStore.load()
        appSettingsStore.save(
            AppSettings(
                provider = existing.provider,
                baseUrl = existing.baseUrl,
                model = existing.model,
                corr3xtBaseUrl = snapshot.corr3xtBaseUrl.trim(),
            )
        )
        _uiState.update { it.copy(globalStatus = "Saved Corr3xt base URL") }
    }

    fun startAuth(methodId: String) {
        val option = AuthCatalog.find(methodId) ?: return
        val corr3xtBaseUrl = Corr3xtAuthClient.normalizedBaseUrl(_uiState.value.corr3xtBaseUrl)
        val state = UUID.randomUUID().toString()
        val startUri = Corr3xtAuthClient.buildStartUri(corr3xtBaseUrl, option, state)
        authSessionStore.savePendingRequest(
            com.nousresearch.hermesagent.data.PendingAuthRequest(
                state = state,
                methodId = option.id,
                startUrl = startUri.toString(),
            )
        )
        val browserIntent = Intent(Intent.ACTION_VIEW, startUri).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        getApplication<Application>().startActivity(browserIntent)
        _uiState.update { current ->
            current.copy(globalStatus = "Opened Corr3xt for ${option.label} sign-in")
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
        val corr3xtBaseUrl = Corr3xtAuthClient.normalizedBaseUrl(settings.corr3xtBaseUrl)
        val sessionsById = authSessionStore.loadSessions().associateBy { it.methodId }
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
        return AuthUiState(
            corr3xtBaseUrl = corr3xtBaseUrl,
            globalStatus = if (signedInAccounts > 0) {
                "$signedInAccounts sign-in methods connected"
            } else {
                "Use Corr3xt to sign into the app or connect provider accounts."
            },
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
        )
    }
}
