package com.mobilefork.hermesagent.ui.boot

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class BootUiState(
    val status: String = "Opening Hermes…",
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
        _uiState.value = BootUiState(status = "Opening Hermes…")
        val startupDelayMillis = if (firstRefresh) FIRST_SHELL_REFRESH_DELAY_MS else 0L
        firstRefresh = false
        viewModelScope.launch {
            if (startupDelayMillis > 0L) {
                delay(startupDelayMillis)
            }
            _uiState.value = BootUiState(status = "Hermes shell ready", ready = true)
        }
    }

    companion object {
        private const val FIRST_SHELL_REFRESH_DELAY_MS = 150L
    }
}
