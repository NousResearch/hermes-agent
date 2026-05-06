package com.nousresearch.hermesagent.device

import android.content.ComponentName
import android.content.Context
import android.provider.Settings

object HermesNotificationController {
    @Volatile
    private var service: HermesNotificationListenerService? = null
    @Volatile
    private var lastPostedPackageName: String = ""
    @Volatile
    private var lastPostedTitle: String = ""
    @Volatile
    private var lastPostedText: String = ""

    fun bind(service: HermesNotificationListenerService) {
        this.service = service
        DeviceStateWriter.write(service.applicationContext)
    }

    fun unbind(service: HermesNotificationListenerService) {
        if (this.service === service) {
            this.service = null
        }
        DeviceStateWriter.write(service.applicationContext)
    }

    fun isListenerConnected(): Boolean = service != null

    fun isListenerEnabled(context: Context): Boolean {
        val expected = ComponentName(context, HermesNotificationListenerService::class.java)
        val enabledListeners = Settings.Secure.getString(
            context.contentResolver,
            "enabled_notification_listeners",
        ).orEmpty()
        return enabledListeners.split(':')
            .mapNotNull { ComponentName.unflattenFromString(it) }
            .any { component ->
                component.packageName.equals(expected.packageName, ignoreCase = true) &&
                    component.className.equals(expected.className, ignoreCase = true)
            }
    }

    fun rememberPostedNotification(packageName: String, title: String, text: String) {
        lastPostedPackageName = packageName.trim()
        lastPostedTitle = title.take(MAX_STATE_CHARS)
        lastPostedText = text.take(MAX_STATE_CHARS)
    }

    fun currentPackageName(): String = lastPostedPackageName

    fun currentTitle(): String = lastPostedTitle

    fun currentText(): String = lastPostedText

    private const val MAX_STATE_CHARS = 500
}
