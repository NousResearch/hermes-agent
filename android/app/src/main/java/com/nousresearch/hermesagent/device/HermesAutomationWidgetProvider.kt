package com.nousresearch.hermesagent.device

import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.Context
import android.content.Intent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class HermesAutomationWidgetProvider : AppWidgetProvider() {
    override fun onUpdate(context: Context, appWidgetManager: AppWidgetManager, appWidgetIds: IntArray) {
        appWidgetIds.forEach { appWidgetId ->
            HermesAutomationWidgetBridge.updateWidget(context, appWidgetManager, appWidgetId)
        }
    }

    override fun onDeleted(context: Context, appWidgetIds: IntArray) {
        HermesAutomationWidgetBridge.removeWidgetConfigs(context, appWidgetIds)
        super.onDeleted(context, appWidgetIds)
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == HermesAutomationWidgetBridge.ACTION_RUN_AUTOMATION_WIDGET) {
            val appWidgetId = intent.getIntExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, AppWidgetManager.INVALID_APPWIDGET_ID)
                .takeIf { it != AppWidgetManager.INVALID_APPWIDGET_ID }
            val pending = goAsync()
            CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
                try {
                    HermesAutomationWidgetBridge.runConfiguredAutomationJson(context.applicationContext, appWidgetId)
                    withContext(Dispatchers.Main) {
                        HermesAutomationWidgetBridge.updateWidgets(context.applicationContext)
                    }
                } finally {
                    pending.finish()
                }
            }
            return
        }
        super.onReceive(context, intent)
    }
}
