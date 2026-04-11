package com.nousresearch.hermesagent.device

import android.accessibilityservice.AccessibilityService
import android.content.ComponentName
import android.content.Context
import android.provider.Settings

enum class HermesGlobalAction(val label: String, val actionId: Int) {
    Home("Home", AccessibilityService.GLOBAL_ACTION_HOME),
    Back("Back", AccessibilityService.GLOBAL_ACTION_BACK),
    Recents("Recents", AccessibilityService.GLOBAL_ACTION_RECENTS),
    Notifications("Notifications", AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS),
    QuickSettings("Quick settings", AccessibilityService.GLOBAL_ACTION_QUICK_SETTINGS),
}

object HermesAccessibilityController {
    @Volatile
    private var service: HermesAccessibilityService? = null

    fun bind(service: HermesAccessibilityService) {
        this.service = service
        DeviceStateWriter.write(service.applicationContext)
    }

    fun unbind(service: HermesAccessibilityService) {
        if (this.service === service) {
            this.service = null
        }
        DeviceStateWriter.write(service.applicationContext)
    }

    fun isServiceConnected(): Boolean = service != null

    fun isServiceEnabled(context: Context): Boolean {
        val enabledServices = Settings.Secure.getString(
            context.contentResolver,
            Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES,
        ).orEmpty()
        val expected = ComponentName(context, HermesAccessibilityService::class.java).flattenToString()
        return enabledServices.split(':').any { it.equals(expected, ignoreCase = true) }
    }

    fun performAction(action: HermesGlobalAction): Boolean {
        return service?.performGlobalAction(action.actionId) == true
    }
}
