package com.nousresearch.hermesagent.ui.shell

import androidx.annotation.DrawableRes
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.ui.i18n.HermesStrings

enum class AppSection(
    @DrawableRes val iconRes: Int,
) {
    Hermes(iconRes = R.drawable.ic_nav_hermes),
    // label = "Accounts"
    Accounts(iconRes = R.drawable.ic_nav_accounts),
    // label = "Nous Portal"
    NousPortal(iconRes = R.drawable.ic_nav_portal),
    Device(iconRes = R.drawable.ic_nav_device),
    Settings(iconRes = R.drawable.ic_nav_settings);

    fun label(strings: HermesStrings): String {
        return when (this) {
            Hermes -> strings.sectionHermes
            Accounts -> strings.sectionAccounts
            NousPortal -> strings.sectionPortal
            Device -> strings.sectionDevice
            Settings -> strings.sectionSettings
        }
    }

    fun navigationLabel(strings: HermesStrings): String {
        return when (this) {
            Device -> when (strings.language) {
                com.nousresearch.hermesagent.ui.i18n.AppLanguage.SPANISH -> "Equipo"
                com.nousresearch.hermesagent.ui.i18n.AppLanguage.PORTUGUESE -> "Aparelho"
                com.nousresearch.hermesagent.ui.i18n.AppLanguage.FRENCH -> "Appareil"
                else -> label(strings)
            }
            else -> label(strings)
        }
    }

    fun title(strings: HermesStrings): String {
        return when (this) {
            Hermes -> strings.sectionHermes
            Accounts -> strings.sectionAccounts
            NousPortal -> strings.portalTitle
            Device -> strings.sectionDevice
            Settings -> strings.sectionSettings
        }
    }

    fun subtitle(strings: HermesStrings): String {
        return when (this) {
            Hermes -> strings.subtitleHermes
            Accounts -> strings.subtitleAccounts
            NousPortal -> strings.subtitlePortal
            Device -> strings.subtitleDevice
            Settings -> strings.subtitleSettings
        }
    }
}

data class ShellActionItem(
    val label: String,
    val description: String = "",
    @DrawableRes val iconRes: Int,
    val onClick: () -> Unit,
)
