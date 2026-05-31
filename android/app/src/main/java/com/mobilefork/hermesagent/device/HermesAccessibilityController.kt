package com.mobilefork.hermesagent.device

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.ComponentName
import android.content.Context
import android.graphics.Path
import android.provider.Settings

enum class HermesGlobalAction(val label: String, val actionId: Int) {
    Home("Home", AccessibilityService.GLOBAL_ACTION_HOME),
    Back("Back", AccessibilityService.GLOBAL_ACTION_BACK),
    Recents("Recents", AccessibilityService.GLOBAL_ACTION_RECENTS),
    Notifications("Notifications", AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS),
    QuickSettings("Quick settings", AccessibilityService.GLOBAL_ACTION_QUICK_SETTINGS),
}

data class HermesScreenMetrics(
    val width: Int,
    val height: Int,
    val density: Float,
)

object HermesAccessibilityController {
    @Volatile
    private var service: HermesAccessibilityService? = null
    @Volatile
    private var lastForegroundPackageName: String = ""

    fun bind(service: HermesAccessibilityService) {
        this.service = service
    }

    fun unbind(service: HermesAccessibilityService) {
        if (this.service === service) {
            this.service = null
        }
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

    fun screenMetrics(): HermesScreenMetrics? {
        val metrics = service?.resources?.displayMetrics ?: return null
        return HermesScreenMetrics(
            width = metrics.widthPixels,
            height = metrics.heightPixels,
            density = metrics.density,
        )
    }

    fun performTap(x: Float, y: Float, durationMs: Long): Boolean {
        val path = Path().apply {
            moveTo(x, y)
        }
        return dispatchGesture(path, durationMs.coerceAtLeast(1L))
    }

    fun performSwipe(startX: Float, startY: Float, endX: Float, endY: Float, durationMs: Long): Boolean {
        val path = Path().apply {
            moveTo(startX, startY)
            lineTo(endX, endY)
        }
        return dispatchGesture(path, durationMs.coerceAtLeast(1L))
    }

    fun currentService(): HermesAccessibilityService? = service

    fun rememberForegroundPackage(packageName: String): Boolean {
        val trimmed = packageName.trim()
        if (trimmed.isBlank() || trimmed == lastForegroundPackageName) {
            return false
        }
        lastForegroundPackageName = trimmed
        return true
    }

    fun currentForegroundPackageName(): String = lastForegroundPackageName

    private fun dispatchGesture(path: Path, durationMs: Long): Boolean {
        val connectedService = service ?: return false
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path, 0L, durationMs))
            .build()
        return connectedService.dispatchGesture(gesture, null, null)
    }
}
