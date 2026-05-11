package com.nousresearch.hermesagent.ui.settings

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Browser
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.nousresearch.hermesagent.backend.BackendKind
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
import com.nousresearch.hermesagent.data.AppSettings
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.ProviderPresets
import com.nousresearch.hermesagent.data.SecureSecretsStore
import com.nousresearch.hermesagent.ui.i18n.AppLanguage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

data class SettingsUiState(
    val provider: String = "openrouter",
    val baseUrl: String = "",
    val model: String = "",
    val apiKey: String = "",
    val dataSaverMode: Boolean = false,
    val onDeviceBackend: String = BackendKind.NONE.persistedValue,
    val languageTag: String = AppLanguage.ENGLISH.tag,
    val onDeviceSummary: String = "Remote provider mode",
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
            onDeviceBackend = stored.onDeviceBackend,
            languageTag = AppLanguage.fromTag(stored.languageTag).tag,
            onDeviceSummary = OnDeviceBackendManager.preferredDownloadSummary(getApplication(), stored.onDeviceBackend),
        )
    }

    fun reload() {
        _uiState.value = loadInitialState()
    }

    fun updateProvider(provider: String) {
        val preset = ProviderPresets.find(provider)
        _uiState.update {
            val providerChanged = provider != it.provider
            it.copy(
                provider = provider,
                baseUrl = if (providerChanged && provider != "custom") preset?.baseUrl.orEmpty() else it.baseUrl,
                model = if (providerChanged && provider != "custom") preset?.modelHint.orEmpty() else it.model,
                apiKey = if (provider == it.provider) it.apiKey else secretsStore.loadApiKey(provider),
            )
        }
    }

    fun updateBaseUrl(value: String) = _uiState.update { it.copy(baseUrl = value) }
    fun updateModel(value: String) = _uiState.update { it.copy(model = value) }
    fun updateApiKey(value: String) = _uiState.update { it.copy(apiKey = value) }
    fun updateDataSaverMode(enabled: Boolean) = _uiState.update { it.copy(dataSaverMode = enabled) }

    fun updateOnDeviceBackend(value: String) {
        _uiState.update {
            it.copy(
                onDeviceBackend = value,
                onDeviceSummary = OnDeviceBackendManager.preferredDownloadSummary(getApplication(), value),
            )
        }
    }

    fun syncOnDeviceBackendWithRuntimeFlavor(runtimeFlavor: String) {
        val backendValue = when (runtimeFlavor) {
            "GGUF" -> BackendKind.LLAMA_CPP.persistedValue
            "LiteRT-LM" -> BackendKind.LITERT_LM.persistedValue
            else -> BackendKind.NONE.persistedValue
        }
        updateOnDeviceBackend(backendValue)
    }

    fun openProviderKeyPage(url: String) {
        val target = url.trim()
        if (target.isBlank()) {
            return
        }
        val uri = Uri.parse(target)
        if (uri.scheme !in setOf("http", "https")) {
            _uiState.update { it.copy(status = "Provider setup URL must start with https:// or http://") }
            return
        }
        val intent = Intent(Intent.ACTION_VIEW, uri).apply {
            addCategory(Intent.CATEGORY_BROWSABLE)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            putExtra(Browser.EXTRA_APPLICATION_ID, getApplication<Application>().packageName)
        }
        runCatching {
            getApplication<Application>().startActivity(intent)
        }.onSuccess {
            _uiState.update { it.copy(status = "Opened provider setup page. If your browser stalls, copy the setup URL and paste it into another browser.") }
        }.onFailure { error ->
            copyProviderKeyPage(target, updateSuccessStatus = false)
            _uiState.update {
                it.copy(status = "Unable to open browser (${error::class.java.simpleName}); copied the provider setup URL.")
            }
        }
    }

    fun copyProviderKeyPage(url: String) {
        copyProviderKeyPage(url, updateSuccessStatus = true)
    }

    fun importSavedProviderCredential() {
        val snapshot = _uiState.value
        val preset = ProviderPresets.find(snapshot.provider)
        val providerLabel = preset?.label ?: snapshot.provider
        if (snapshot.provider.isBlank() || snapshot.provider == "custom") {
            _uiState.update { it.copy(status = "Choose a saved provider before importing a Hermes credential.") }
            return
        }
        viewModelScope.launch {
            _uiState.update { it.copy(status = "Checking saved Hermes credential for $providerLabel…") }
            val bundleResult = runCatching {
                withContext(Dispatchers.IO) {
                    val app = getApplication<Application>()
                    HermesRuntimeManager.ensurePythonStarted(app)
                    Python.getInstance()
                        .getModule("hermes_android.auth_bridge")
                        .callAttr("read_provider_auth_bundle_json", snapshot.provider)
                        .toString()
                }
            }
            val payload = bundleResult.getOrElse { error ->
                _uiState.update {
                    it.copy(status = "Unable to read saved Hermes credential (${error::class.java.simpleName}).")
                }
                return@launch
            }
            val json = runCatching { JSONObject(payload) }.getOrElse {
                _uiState.update { it.copy(status = "Saved Hermes credential for $providerLabel could not be decoded.") }
                return@launch
            }
            val apiKey = listOf(
                json.optString("api_key"),
                json.optString("access_token"),
                json.optString("session_token"),
            ).firstOrNull { it.isNotBlank() }.orEmpty()
            val configured = json.optBoolean("configured", false) || apiKey.isNotBlank()
            if (!configured || apiKey.isBlank()) {
                _uiState.update { it.copy(status = "No saved Hermes credential found for $providerLabel.") }
                return@launch
            }

            val resolvedBaseUrl = json.optString("base_url")
                .ifBlank { snapshot.baseUrl }
                .ifBlank { preset?.baseUrl.orEmpty() }
            val resolvedModel = snapshot.model.ifBlank { preset?.modelHint.orEmpty() }
            val runtimeConfigBaseUrl = ProviderPresets.runtimeConfigBaseUrl(snapshot.provider, resolvedBaseUrl)
            val existingSettings = settingsStore.load()
            val updatedSettings = AppSettings(
                provider = snapshot.provider,
                baseUrl = resolvedBaseUrl,
                model = resolvedModel,
                corr3xtBaseUrl = existingSettings.corr3xtBaseUrl,
                dataSaverMode = existingSettings.dataSaverMode,
                onDeviceBackend = existingSettings.onDeviceBackend,
                languageTag = existingSettings.languageTag,
            )
            runCatching {
                withContext(Dispatchers.IO) {
                    val app = getApplication<Application>()
                    HermesRuntimeManager.ensurePythonStarted(app)
                    val python = Python.getInstance()
                    python.getModule("hermes_android.auth_bridge").callAttr(
                        "write_provider_auth_bundle",
                        snapshot.provider,
                        apiKey,
                        json.optString("access_token"),
                        json.optString("session_token"),
                        json.optString("refresh_token"),
                        resolvedBaseUrl,
                    )
                    python.getModule("hermes_android.config_bridge").callAttr(
                        "write_runtime_config",
                        snapshot.provider,
                        resolvedModel,
                        runtimeConfigBaseUrl,
                    )
                }
                settingsStore.save(updatedSettings)
                secretsStore.saveApiKey(snapshot.provider, apiKey)
                HermesRuntimeManager.stop()
                HermesRuntimeManager.ensureStarted(getApplication())
            }.onSuccess {
                _uiState.update {
                    it.copy(
                        baseUrl = resolvedBaseUrl,
                        model = resolvedModel,
                        apiKey = apiKey,
                        status = "Imported saved Hermes credential for $providerLabel and restarted the runtime.",
                    )
                }
            }.onFailure { error ->
                _uiState.update {
                    it.copy(status = "Saved Hermes credential import failed (${error::class.java.simpleName}).")
                }
            }
        }
    }

    private fun copyProviderKeyPage(url: String, updateSuccessStatus: Boolean) {
        val target = url.trim()
        if (target.isBlank()) {
            return
        }
        val providerId = ProviderPresets.providerIdForSetupUrl(target)
        val setupText = providerId?.let { ProviderPresets.setupClipboardText(it) }
            .orEmpty()
            .ifBlank { target }
        val fallbackCount = providerId?.let { ProviderPresets.setupUrls(it).size - 1 } ?: 0
        val clipboard = getApplication<Application>().getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
        clipboard?.setPrimaryClip(ClipData.newPlainText("Hermes provider setup URLs", setupText))
        if (updateSuccessStatus) {
            val suffix = when (fallbackCount) {
                0 -> ""
                1 -> " and 1 alternate official page"
                else -> " and $fallbackCount alternate official pages"
            }
            _uiState.update { it.copy(status = "Copied provider setup URL$suffix") }
        }
    }

    fun startLocalRuntimeForFlavor(runtimeFlavor: String) {
        val backendValue = when (runtimeFlavor) {
            "GGUF" -> BackendKind.LLAMA_CPP.persistedValue
            "LiteRT-LM" -> BackendKind.LITERT_LM.persistedValue
            else -> BackendKind.NONE.persistedValue
        }
        _uiState.update {
            it.copy(
                provider = "custom",
                baseUrl = "",
                model = "",
                onDeviceBackend = backendValue,
                onDeviceSummary = OnDeviceBackendManager.preferredDownloadSummary(getApplication(), backendValue),
                status = "Starting local Hermes runtime…",
            )
        }
        save()
    }

    fun selectLanguage(language: AppLanguage) {
        val normalized = language.tag
        settingsStore.save(settingsStore.load().copy(languageTag = normalized))
        val strings = com.nousresearch.hermesagent.ui.i18n.hermesStringsFor(language)
        _uiState.update {
            it.copy(
                languageTag = normalized,
                status = strings.languageSwitchedTo(language.nativeLabel),
            )
        }
    }

    fun save() {
        val snapshot = _uiState.value
        viewModelScope.launch {
            val existingSettings = settingsStore.load()
            val updatedSettings = AppSettings(
                provider = snapshot.provider,
                baseUrl = snapshot.baseUrl,
                model = snapshot.model,
                corr3xtBaseUrl = existingSettings.corr3xtBaseUrl,
                dataSaverMode = snapshot.dataSaverMode,
                onDeviceBackend = snapshot.onDeviceBackend,
                languageTag = snapshot.languageTag,
            )
            settingsStore.save(updatedSettings)

            val app = getApplication<Application>()
            val localBackendStatus = OnDeviceBackendManager.ensureConfigured(app, snapshot.onDeviceBackend)
            val backendKind = BackendKind.fromPersistedValue(snapshot.onDeviceBackend)

            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(app))
            }
            val useLocalBackend = localBackendStatus.started
            val effectiveProvider = if (useLocalBackend) "custom" else snapshot.provider
            val effectiveModel = if (useLocalBackend) localBackendStatus.modelName else snapshot.model
            val effectiveBaseUrl = if (useLocalBackend) {
                localBackendStatus.baseUrl
            } else {
                ProviderPresets.runtimeConfigBaseUrl(snapshot.provider, snapshot.baseUrl)
            }
            Python.getInstance().getModule("hermes_android.config_bridge").callAttr(
                "write_runtime_config",
                effectiveProvider,
                effectiveModel,
                effectiveBaseUrl,
            )
            val providerApiKey = snapshot.apiKey.trim()
            val preservedBlankCredential = providerApiKey.isBlank() && snapshot.provider != "custom"
            if (providerApiKey.isNotBlank()) {
                secretsStore.saveApiKey(snapshot.provider, providerApiKey)
                Python.getInstance().getModule("hermes_android.auth_bridge").callAttr(
                    "write_provider_api_key",
                    snapshot.provider,
                    providerApiKey,
                )
            }
            HermesRuntimeManager.stop()
            HermesRuntimeManager.ensureStarted(app)
            _uiState.update {
                val backendSummary = if (localBackendStatus.started) {
                    "${localBackendStatus.backendKind.persistedValue} ready · ${localBackendStatus.modelName}"
                } else {
                    OnDeviceBackendManager.preferredDownloadSummary(app, snapshot.onDeviceBackend)
                }
                val statusMessage = when {
                    useLocalBackend -> "On-device backend ready and Hermes runtime restarted"
                    backendKind != BackendKind.NONE -> "${localBackendStatus.statusMessage}. Hermes stayed on your saved remote provider."
                    snapshot.dataSaverMode -> "Settings saved. Data saver mode now keeps heavy downloads on Wi‑Fi / unmetered networks."
                    preservedBlankCredential -> "Settings saved and backend restarted. Blank API key field left existing Hermes credentials untouched."
                    else -> "Settings saved and backend restarted"
                }
                it.copy(
                    onDeviceSummary = backendSummary,
                    apiKey = providerApiKey.ifBlank { it.apiKey },
                    status = statusMessage,
                )
            }
        }
    }
}
