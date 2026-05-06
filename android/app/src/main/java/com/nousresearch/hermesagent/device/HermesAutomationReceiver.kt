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
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED -> runAsync {
                HermesAutomationScheduler.scheduleAll(context.applicationContext)
                HermesAutomationBridge.runTriggerJson(context.applicationContext, TRIGGER_BOOT)
            }
            Intent.ACTION_POWER_CONNECTED -> runAsync {
                HermesAutomationBridge.runTriggerJson(context.applicationContext, TRIGGER_POWER_CONNECTED)
            }
            Intent.ACTION_POWER_DISCONNECTED -> runAsync {
                HermesAutomationBridge.runTriggerJson(context.applicationContext, TRIGGER_POWER_DISCONNECTED)
            }
            Intent.ACTION_BATTERY_LOW -> runAsync {
                HermesAutomationBridge.runTriggerJson(context.applicationContext, TRIGGER_BATTERY_LOW)
            }
            Intent.ACTION_BATTERY_OKAY -> runAsync {
                HermesAutomationBridge.runTriggerJson(context.applicationContext, TRIGGER_BATTERY_OKAY)
            }
            HermesAutomationScheduler.ACTION_RUN_AUTOMATION -> {
                val automationId = intent.getStringExtra(HermesAutomationScheduler.EXTRA_AUTOMATION_ID).orEmpty()
                if (automationId.isBlank()) {
                    return
                }
                runAsync {
                    HermesAutomationBridge.runAutomationJson(context.applicationContext, automationId, "alarm")
                }
            }
        }
    }

    private fun runAsync(block: () -> Unit) {
        val pendingResult = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                block()
            } finally {
                pendingResult.finish()
            }
        }
    }
}
