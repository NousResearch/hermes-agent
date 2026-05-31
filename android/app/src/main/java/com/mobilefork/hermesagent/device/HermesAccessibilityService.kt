package com.mobilefork.hermesagent.device

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class HermesAccessibilityService : AccessibilityService() {
    private val automationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onServiceConnected() {
        super.onServiceConnected()
        HermesAccessibilityController.bind(this)
        automationScope.launch {
            DeviceStateWriter.write(applicationContext)
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event?.eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            return
        }
        val packageName = event.packageName?.toString()?.trim().orEmpty()
        if (packageName.isBlank()) {
            return
        }
        if (!HermesAccessibilityController.rememberForegroundPackage(packageName)) {
            return
        }
        automationScope.launch {
            DeviceStateWriter.write(applicationContext)
            if (packageName != applicationContext.packageName) {
                HermesAutomationBridge.runAppForegroundTriggerJson(applicationContext, packageName)
            }
        }
    }

    override fun onInterrupt() {
        // No-op for now.
    }

    override fun onDestroy() {
        automationScope.cancel()
        HermesAccessibilityController.unbind(this)
        super.onDestroy()
    }
}
