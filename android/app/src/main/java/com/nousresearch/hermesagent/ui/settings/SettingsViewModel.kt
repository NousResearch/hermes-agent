package com.nousresearch.hermesagent.ui.settings

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.data.AppSettings
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.ProviderPresets
import com.nousresearch.hermesagent.data.SecureSecretsStore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class SettingsUiState(
    val provider: String = "openrouter",
    val baseUrl: String = "",
    val model: String = "",
    val apiKey: String = "",
    val dataSaverMode: Boolean = false,
    val status: String = "",
)

class SettingsViewModel(application: Application) : AndroidViewModel(application) {
    private val settingsStore = AppSettingsStore(application)
    private val secretsStore = SecureSecretsStore(application)

    private val _uiState = MutableStateFlow(loadInitialState())
    val uiState: StateFlow<SettingsUiState> = _uiState.asStateFlow()

    private fun loadInitialState(): SettingsUiState {
        val stored = settingsStore.load()
        return SettingsUiState(
            provider = stored.provider,
            baseUrl = stored.baseUrl,
            model = stored.model,
            apiKey = secretsStore.loadApiKey(stored.provider),
            dataSaverMode = stored.dataSaverMode,
        )
    }

    fun updateProvider(provider: String) {
        val preset = ProviderPresets.find(provider)
        _uiState.update {
            it.copy(
                provider = provider,
                baseUrl = if (it.baseUrl.isBlank()) preset?.baseUrl.orEmpty() else it.baseUrl,
                model = if (it.model.isBlank()) preset?.modelHint.orEmpty() else it.model,
                apiKey = if (provider == it.provider) it.apiKey else secretsStore.loadApiKey(provider),
            )
        }
    }

    fun updateBaseUrl(value: String) = _uiState.update { it.copy(baseUrl = value) }
    fun updateModel(value: String) = _uiState.update { it.copy(model = value) }
    fun updateApiKey(value: String) = _uiState.update { it.copy(apiKey = value) }
    fun updateDataSaverMode(enabled: Boolean) = _uiState.update { it.copy(dataSaverMode = enabled) }

    fun save() {
        val snapshot = _uiState.value
        viewModelScope.launch {
            val existingSettings = settingsStore.load()
            settingsStore.save(
                AppSettings(
                    provider = snapshot.provider,
                    baseUrl = snapshot.baseUrl,
                    model = snapshot.model,
                    corr3xtBaseUrl = existingSettings.corr3xtBaseUrl,
                    dataSaverMode = snapshot.dataSaverMode,
                )
            )
            secretsStore.saveApiKey(snapshot.provider, snapshot.apiKey)

            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(getApplication()))
            }
            Python.getInstance().getModule("hermes_android.config_bridge").callAttr(
                "write_runtime_config",
                snapshot.provider,
                snapshot.model,
                snapshot.baseUrl,
            )
            Python.getInstance().getModule("hermes_android.auth_bridge").callAttr(
                "write_provider_api_key",
                snapshot.provider,
                snapshot.apiKey,
            )
            HermesRuntimeManager.stop()
            HermesRuntimeManager.ensureStarted(getApplication())
            _uiState.update {
                it.copy(
                    status = if (snapshot.dataSaverMode) {
                        "Settings saved. Data saver mode now keeps heavy downloads on Wi‑Fi / unmetered networks."
                    } else {
                        "Settings saved and backend restarted"
                    },
                )
            }
        }
    }
}
