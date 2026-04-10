package com.nousresearch.hermesagent.ui.settings

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

private val ENABLED_TOOLS = listOf(
    "web_search",
    "web_extract",
    "vision_analyze",
    "image_generate",
    "skills_list",
    "skill_view",
    "skill_manage",
    "todo",
    "memory",
    "session_search",
)

private val BLOCKED_TOOL_CLASSES = listOf(
    "terminal / process",
    "local file editing",
    "browser automation",
    "execute_code",
    "delegate_task",
    "cronjob",
    "voice / transcription",
)

@Composable
fun ToolProfileCard() {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Android MVP Tool Profile", style = MaterialTheme.typography.titleMedium)
            Text(
                "Enabled: ${ENABLED_TOOLS.joinToString()}",
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                "Not included in the first mobile release: ${BLOCKED_TOOL_CLASSES.joinToString()}",
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    }
}
