package com.nousresearch.hermesagent.ui.boot

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URL

data class BootUiState(
    val status: String = "Booting Hermes runtime…",
    val ready: Boolean = false,
    val probeResult: String = "",
    val baseUrl: String = "",
    val error: String = "",
)

class BootViewModel(application: Application) : AndroidViewModel(application) {
    private var firstRefresh = true
    private val _uiState = MutableStateFlow(BootUiState())
    val uiState: StateFlow<BootUiState> = _uiState.asStateFlow()

    fun refresh() {
        _uiState.value = BootUiState(status = "Booting Hermes runtime…")
        val startupDelayMillis = if (firstRefresh) FIRST_LAUNCH_RUNTIME_WARMUP_DELAY_MS else 0L
        firstRefresh = false
        viewModelScope.launch {
            if (startupDelayMillis > 0L) {
                delay(startupDelayMillis)
            }
            val runtime = withContext(Dispatchers.IO) {
                HermesRuntimeManager.ensureStarted(getApplication())
            }
            if (!runtime.started || runtime.baseUrl.isNullOrBlank()) {
                _uiState.value = BootUiState(
                    status = "Hermes backend failed to start",
                    error = runtime.error.orEmpty(),
                    probeResult = runtime.probeResult.orEmpty(),
                    baseUrl = runtime.baseUrl.orEmpty(),
                )
                return@launch
            }

            runCatching {
                withContext(Dispatchers.IO) {
                    checkHealth(runtime.baseUrl, runtime.apiKey)
                }
            }.onSuccess { healthOk ->
                _uiState.value = if (healthOk) {
                    BootUiState(
                        status = "Hermes backend is ready",
                        ready = true,
                        probeResult = runtime.probeResult.orEmpty(),
                        baseUrl = runtime.baseUrl.orEmpty(),
                    )
                } else {
                    BootUiState(
                        status = "Hermes backend did not pass /health",
                        probeResult = runtime.probeResult.orEmpty(),
                        baseUrl = runtime.baseUrl.orEmpty(),
                        error = "GET /health did not return HTTP 200",
                    )
                }
            }.onFailure { error ->
                _uiState.value = BootUiState(
                    status = "Hermes backend health check failed",
                    probeResult = runtime.probeResult.orEmpty(),
                    baseUrl = runtime.baseUrl.orEmpty(),
                    error = error.message ?: error.javaClass.simpleName,
                )
            }
        }
    }

    private fun checkHealth(baseUrl: String, apiKey: String?): Boolean {
        val connection = (URL(hermesHealthUrl(baseUrl)).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 5000
            readTimeout = 5000
            if (!apiKey.isNullOrBlank()) {
                setRequestProperty("Authorization", "Bearer $apiKey")
            }
        }
        return try {
            connection.responseCode == 200
        } finally {
            connection.disconnect()
        }
    }
}

private const val FIRST_LAUNCH_RUNTIME_WARMUP_DELAY_MS = 30_000L

internal fun hermesHealthUrl(baseUrl: String): String {
    return baseUrl.trimEnd('/').removeSuffix("/v1") + "/health"
}
