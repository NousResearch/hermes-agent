package com.nousresearch.hermesagent.ui.shell

import androidx.annotation.DrawableRes
import com.nousresearch.hermesagent.R

enum class AppSection(
    val label: String,
    val title: String,
    val subtitle: String,
    @DrawableRes val iconRes: Int,
) {
    Hermes(
        label = "Hermes",
        title = "Hermes",
        subtitle = "Chat, commands, and voice",
        iconRes = R.drawable.ic_nav_hermes,
    ),
    Accounts(
        label = "Accounts",
        title = "Accounts",
        subtitle = "Corr3xt sign-in and provider access",
        iconRes = R.drawable.ic_nav_accounts,
    ),
    NousPortal(
        label = "Portal",
        title = "Nous Portal",
        subtitle = "Portal preview and browser fallback",
        iconRes = R.drawable.ic_nav_portal,
    ),
    Device(
        label = "Device",
        title = "Device",
        subtitle = "Files, Linux suite, and phone controls",
        iconRes = R.drawable.ic_nav_device,
    ),
    Settings(
        label = "Settings",
        title = "Settings",
        subtitle = "Runtime provider and API configuration",
        iconRes = R.drawable.ic_nav_settings,
    ),
}

data class ShellActionItem(
    val label: String,
    val description: String = "",
    @DrawableRes val iconRes: Int,
    val onClick: () -> Unit,
)
