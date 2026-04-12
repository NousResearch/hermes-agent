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
import com.nousresearch.hermesagent.ui.i18n.LocalHermesStrings

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
    val strings = LocalHermesStrings.current
    OutlinedCard(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(strings.toolProfileTitle(), style = MaterialTheme.typography.titleMedium)
            Text(
                strings.toolProfileEnabledSummary(ENABLED_TOOLS.joinToString()),
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                strings.toolProfileLinuxSummary(),
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                strings.toolProfileAccessibilitySummary(),
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                strings.toolProfileCommandSuiteSummary(),
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                strings.toolProfileExcludedSummary(BLOCKED_TOOL_CLASSES.joinToString()),
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    }
}
