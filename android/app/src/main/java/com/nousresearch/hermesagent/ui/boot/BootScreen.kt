package com.nousresearch.hermesagent.ui.boot

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nousresearch.hermesagent.ui.chat.ChatScreen

@Composable
fun BootScreen(viewModel: BootViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsState()

    if (uiState.ready) {
        ChatScreen()
        return
    }

    MaterialTheme {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(text = uiState.status, style = MaterialTheme.typography.headlineSmall)
                if (uiState.baseUrl.isNotBlank()) {
                    Text(text = uiState.baseUrl, modifier = Modifier.padding(top = 12.dp))
                }
                if (uiState.probeResult.isNotBlank()) {
                    Text(text = uiState.probeResult, modifier = Modifier.padding(top = 12.dp))
                }
                if (uiState.error.isNotBlank()) {
                    Text(text = uiState.error, modifier = Modifier.padding(top = 12.dp))
                }
                Button(onClick = viewModel::refresh, modifier = Modifier.padding(top = 20.dp)) {
                    Text("Retry")
                }
            }
        }
    }
}
