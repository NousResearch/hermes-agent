package com.nousresearch.hermesagent.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.unit.dp

data class HermesThemeConfig(
    val primaryHex: String = "#8C7BFF",
    val secondaryHex: String = "#C6A15B",
    val backgroundHex: String = "#090B10",
    val surfaceHex: String = "#11141C",
    val surfaceVariantHex: String = "#1B202B",
    val cardShape: String = "rounded",
)

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
fun HermesTheme(
    config: HermesThemeConfig = HermesThemeConfig(),
    content: @Composable () -> Unit,
) {
    val primary = parseThemeColor(config.primaryHex, HermesDarkColors.primary)
    val secondary = parseThemeColor(config.secondaryHex, HermesDarkColors.secondary)
    val background = parseThemeColor(config.backgroundHex, HermesDarkColors.background)
    val surface = parseThemeColor(config.surfaceHex, HermesDarkColors.surface)
    val surfaceVariant = parseThemeColor(config.surfaceVariantHex, HermesDarkColors.surfaceVariant)
    val colors = darkColorScheme(
        primary = primary,
        onPrimary = readableOn(primary),
        primaryContainer = surfaceVariant,
        onPrimaryContainer = readableOn(surfaceVariant),
        secondary = secondary,
        onSecondary = readableOn(secondary),
        secondaryContainer = secondary.copy(alpha = 0.18f),
        onSecondaryContainer = HermesDarkColors.onSecondaryContainer,
        background = background,
        onBackground = HermesDarkColors.onBackground,
        surface = surface,
        onSurface = HermesDarkColors.onSurface,
        surfaceVariant = surfaceVariant,
        onSurfaceVariant = HermesDarkColors.onSurfaceVariant,
        outline = HermesDarkColors.outline,
        outlineVariant = HermesDarkColors.outlineVariant,
        error = HermesDarkColors.error,
        onError = HermesDarkColors.onError,
    )
    MaterialTheme(
        colorScheme = colors,
        shapes = hermesShapes(config.cardShape),
        content = content,
    )
}

fun normalizeThemeHex(value: String, fallback: String): String {
    val trimmed = value.trim()
    val raw = if (trimmed.startsWith("#")) trimmed.drop(1) else trimmed
    if (!Regex("[0-9a-fA-F]{6}|[0-9a-fA-F]{8}").matches(raw)) {
        return fallback
    }
    return "#${raw.uppercase()}"
}

private fun parseThemeColor(value: String, fallback: Color): Color {
    val normalized = normalizeThemeHex(value, "")
    if (normalized.isBlank()) return fallback
    return runCatching { Color(android.graphics.Color.parseColor(normalized)) }.getOrDefault(fallback)
}

private fun readableOn(color: Color): Color {
    return if (color.luminance() > 0.54f) Color(0xFF121318) else Color.White
}

private fun hermesShapes(cardShape: String): Shapes {
    val normalized = cardShape.trim().lowercase()
    val small = when (normalized) {
        "square", "squared" -> 2.dp
        "soft" -> 8.dp
        else -> 12.dp
    }
    val medium = when (normalized) {
        "square", "squared" -> 4.dp
        "soft" -> 14.dp
        else -> 20.dp
    }
    val large = when (normalized) {
        "square", "squared" -> 6.dp
        "soft" -> 18.dp
        else -> 28.dp
    }
    return Shapes(
        extraSmall = RoundedCornerShape(small / 2),
        small = RoundedCornerShape(small),
        medium = RoundedCornerShape(medium),
        large = RoundedCornerShape(large),
        extraLarge = RoundedCornerShape(large + 6.dp),
    )
}
