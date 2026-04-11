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
    "android_device_status",
    "read_file",
    "search_files",
    "write_file",
    "patch",
    "web_search",
    "web_extract",
    "vision_analyze",
    "skills_list",
    "skill_view",
    "skill_manage",
    "todo",
    "memory",
    "session_search",
)

private val BLOCKED_TOOL_CLASSES = listOf(
    "terminal / process",
    "browser automation",
    "execute_code",
    "delegate_task",
    "cronjob",
    "image generation (deferred)",
    "voice / transcription",
)

@Composable
fun ToolProfileCard() {
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Android alpha Tool Profile", style = MaterialTheme.typography.titleMedium)
            Text(
                "Enabled: ${ENABLED_TOOLS.joinToString()}",
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                "Workspace file tools target the Hermes Device workspace. Use android_device_status to discover the current path before asking Hermes to edit files.",
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                "Still excluded from the mobile runtime: ${BLOCKED_TOOL_CLASSES.joinToString()}",
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    }
}
