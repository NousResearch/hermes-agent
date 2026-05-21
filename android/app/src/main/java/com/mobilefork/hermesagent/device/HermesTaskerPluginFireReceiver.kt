package com.mobilefork.hermesagent.device

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class HermesTaskerPluginFireReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != HermesTaskerPluginBridge.ACTION_FIRE_SETTING) {
            return
        }
        val bundle = HermesTaskerPluginBridge.bundleFromIntent(intent) ?: return
        val senderPackage = if (Build.VERSION.SDK_INT >= 34) sentFromPackage.orEmpty() else ""
        val pendingResult = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                HermesTaskerPluginBridge.runPluginBundleJson(context.applicationContext, bundle, senderPackage)
            } finally {
                pendingResult.finish()
            }
        }
    }
}
