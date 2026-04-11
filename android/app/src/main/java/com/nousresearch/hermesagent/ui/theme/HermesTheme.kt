package com.nousresearch.hermesagent.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val HermesLightColors = lightColorScheme(
    primary = Color(0xFF5B2E8C),
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFEBDDFF),
    onPrimaryContainer = Color(0xFF241038),
    secondary = Color(0xFFE8A93B),
    onSecondary = Color(0xFF2B1A00),
    secondaryContainer = Color(0xFFFFE2B3),
    onSecondaryContainer = Color(0xFF2B1A00),
    background = Color(0xFFF6F2FB),
    onBackground = Color(0xFF1F1A24),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF1F1A24),
    surfaceVariant = Color(0xFFE9E0F2),
    onSurfaceVariant = Color(0xFF4E445A),
    error = Color(0xFFB3261E),
    onError = Color(0xFFFFFFFF),
)

@Composable
fun HermesTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = HermesLightColors,
        content = content,
    )
}
