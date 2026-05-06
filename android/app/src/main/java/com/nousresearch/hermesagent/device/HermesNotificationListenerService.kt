package com.nousresearch.hermesagent.device

import android.app.Notification
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class HermesNotificationListenerService : NotificationListenerService() {
    private val automationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onListenerConnected() {
        super.onListenerConnected()
        HermesNotificationController.bind(this)
    }

    override fun onListenerDisconnected() {
        HermesNotificationController.unbind(this)
        super.onListenerDisconnected()
    }

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        val packageName = sbn?.packageName?.trim().orEmpty()
        if (packageName.isBlank()) {
            return
        }
        val extras = sbn?.notification?.extras
        val title = extras?.getCharSequence(Notification.EXTRA_TITLE)?.toString().orEmpty()
        val text = extras?.getCharSequence(Notification.EXTRA_TEXT)?.toString().orEmpty()
        HermesNotificationController.rememberPostedNotification(packageName, title, text)
        DeviceStateWriter.write(applicationContext)
        automationScope.launch {
            HermesAutomationBridge.runNotificationPostedTriggerJson(applicationContext, packageName, title, text)
        }
    }

    override fun onDestroy() {
        automationScope.cancel()
        HermesNotificationController.unbind(this)
        super.onDestroy()
    }
}
