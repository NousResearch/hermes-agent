package com.mobilefork.hermesagent.ui.boot

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.withFrameNanos
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilefork.hermesagent.ui.shell.AppShellScreen

@Composable
fun BootScreen(viewModel: BootViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsState()
    LaunchedEffect(Unit) {
        withFrameNanos { }
        viewModel.refresh()
    }
    AppShellScreen(
        bootUiState = uiState,
        onRetryHermes = viewModel::refresh,
    )
}
