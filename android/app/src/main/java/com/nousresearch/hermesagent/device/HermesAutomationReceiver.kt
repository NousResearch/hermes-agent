package com.nousresearch.hermesagent.device

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class HermesAutomationReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            HermesAutomationScheduler.scheduleAll(context.applicationContext)
            return
        }
        if (intent.action != HermesAutomationScheduler.ACTION_RUN_AUTOMATION) {
            return
        }
        val automationId = intent.getStringExtra(HermesAutomationScheduler.EXTRA_AUTOMATION_ID).orEmpty()
        if (automationId.isBlank()) {
            return
        }
        val pendingResult = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                HermesAutomationBridge.runAutomationJson(context.applicationContext, automationId, "alarm")
            } finally {
                pendingResult.finish()
            }
        }
    }
}
