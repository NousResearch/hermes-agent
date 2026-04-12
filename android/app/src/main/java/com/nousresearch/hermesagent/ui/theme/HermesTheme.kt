package com.nousresearch.hermesagent.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val HermesLightColors = lightColorScheme(
    primary = Color(0xFF4D2FA4),
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFE7E0FF),
    onPrimaryContainer = Color(0xFF24104A),
    secondary = Color(0xFFE8A93B),
    onSecondary = Color(0xFF2B1A00),
    secondaryContainer = Color(0xFFFFE6BD),
    onSecondaryContainer = Color(0xFF2D1B00),
    background = Color(0xFFF7F4FC),
    onBackground = Color(0xFF1F1A24),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF1F1A24),
    surfaceVariant = Color(0xFFF0E8F8),
    onSurfaceVariant = Color(0xFF4C4559),
    outlineVariant = Color(0xFFD9CDE8),
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
