package com.nousresearch.hermesagent.device

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent

class HermesAccessibilityService : AccessibilityService() {
    override fun onServiceConnected() {
        super.onServiceConnected()
        HermesAccessibilityController.bind(this)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // Alpha scaffold: event inspection hooks can be added here later.
    }

    override fun onInterrupt() {
        // No-op for now.
    }

    override fun onDestroy() {
        HermesAccessibilityController.unbind(this)
        super.onDestroy()
    }
}
