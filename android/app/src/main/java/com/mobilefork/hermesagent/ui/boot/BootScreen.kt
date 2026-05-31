package com.mobilefork.hermesagent.ui.boot

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.mobilefork.hermesagent.R
import com.mobilefork.hermesagent.ui.shell.AppShellScreen

@Composable
fun BootScreen() {
    var shellVisible by rememberSaveable { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        withFrameNanos { }
        shellVisible = true
    }
    if (!shellVisible) {
        StartupFirstFrame()
        return
    }
    val viewModel: BootViewModel = viewModel()
    val uiState by viewModel.uiState.collectAsState()
    LaunchedEffect(viewModel) {
        viewModel.refresh()
    }
    AppShellScreen(
        bootUiState = uiState,
        onRetryHermes = viewModel::refresh,
    )
}

@Composable
private fun StartupFirstFrame() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF05070A)),
        contentAlignment = Alignment.Center,
    ) {
        Image(
            painter = painterResource(id = R.drawable.hermes_agent_fork_logo),
            contentDescription = null,
            modifier = Modifier.size(88.dp),
        )
    }
}
