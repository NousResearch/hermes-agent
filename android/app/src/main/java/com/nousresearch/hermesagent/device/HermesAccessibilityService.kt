package com.nousresearch.hermesagent.device

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
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        DeviceStateWriter.write(applicationContext)
        if (event?.eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            return
        }
        val packageName = event.packageName?.toString()?.trim().orEmpty()
        if (!HermesAccessibilityController.rememberForegroundPackage(packageName)) {
            return
        }
        automationScope.launch {
            HermesAutomationBridge.runAppForegroundTriggerJson(applicationContext, packageName)
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
