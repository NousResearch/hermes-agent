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
    "terminal",
    "process",
    "android_device_status",
    "android_shared_folder_list",
    "android_shared_folder_read",
    "android_shared_folder_write",
    "android_ui_snapshot",
    "android_ui_action",
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
                "Hermes now has a local Linux command suite in the Android app, so terminal/process can execute real CLI commands while shared-folder tools handle direct document edits.",
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                "Accessibility targeting is available through android_ui_snapshot + android_ui_action after you enable the Hermes accessibility service.",
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                "The Android command suite is extracted into an app-private prefix and exposed through terminal/process for the same style of local CLI usage Hermes already supports in Termux.",
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                "Still excluded from the mobile runtime: ${BLOCKED_TOOL_CLASSES.joinToString()}",
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    }
}
