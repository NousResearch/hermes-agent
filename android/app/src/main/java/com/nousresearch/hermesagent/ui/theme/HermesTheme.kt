package com.nousresearch.hermesagent.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val HermesDarkColors = darkColorScheme(
    primary = Color(0xFF8C7BFF),
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFF1A1D29),
    onPrimaryContainer = Color(0xFFE9E4FF),
    secondary = Color(0xFFC6A15B),
    onSecondary = Color(0xFF1D1407),
    secondaryContainer = Color(0xFF2B2214),
    onSecondaryContainer = Color(0xFFF5E5C6),
    background = Color(0xFF090B10),
    onBackground = Color(0xFFF2F3F5),
    surface = Color(0xFF11141C),
    onSurface = Color(0xFFF2F3F5),
    surfaceVariant = Color(0xFF1B202B),
    onSurfaceVariant = Color(0xFFD7DBE4),
    outline = Color(0xFF394150),
    outlineVariant = Color(0xFF232A36),
    error = Color(0xFFFF6B6B),
    onError = Color(0xFF2A0C0C),
)

@Composable
fun HermesTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = HermesDarkColors,
        content = content,
    )
}
