package com.nousresearch.hermesagent.ui.i18n

enum class AppLanguage(
    val tag: String,
    val flag: String,
    val nativeLabel: String,
) {
    ENGLISH("en", "🇬🇧", "English"),
    CHINESE("zh", "🇨🇳", "中文"),
    SPANISH("es", "🇪🇸", "Español"),
    GERMAN("de", "🇩🇪", "Deutsch"),
    PORTUGUESE("pt", "🇵🇹", "Português"),
    FRENCH("fr", "🇫🇷", "Français");

    companion object {
        fun fromTag(tag: String?): AppLanguage {
            val normalized = tag.orEmpty().trim().lowercase()
            return entries.firstOrNull { it.tag == normalized } ?: ENGLISH
        }
    }
}
