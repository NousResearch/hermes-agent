package com.nousresearch.hermesagent.device

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.SystemClock

object HermesAutomationScheduler {
    fun schedule(context: Context, record: HermesAutomationRecord) {
        if (!record.enabled || record.triggerType != TRIGGER_INTERVAL) {
            cancel(context, record.id)
            return
        }
        val intervalMinutes = record.intervalMinutes ?: return
        if (intervalMinutes < MIN_INTERVAL_MINUTES) {
            cancel(context, record.id)
            return
        }
        val intervalMillis = intervalMinutes * 60_000L
        val alarmManager = context.applicationContext.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val pendingIntent = pendingIntent(context, record.id)
        alarmManager.setInexactRepeating(
            AlarmManager.ELAPSED_REALTIME_WAKEUP,
            SystemClock.elapsedRealtime() + intervalMillis,
            intervalMillis,
            pendingIntent,
        )
    }

    fun scheduleAll(context: Context) {
        HermesAutomationStore(context).list().forEach { record ->
            schedule(context, record)
        }
    }

    fun cancel(context: Context, id: String) {
        val alarmManager = context.applicationContext.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        alarmManager.cancel(pendingIntent(context, id))
    }

    private fun pendingIntent(context: Context, id: String): PendingIntent {
        val appContext = context.applicationContext
        val intent = Intent(appContext, HermesAutomationReceiver::class.java)
            .setAction(ACTION_RUN_AUTOMATION)
            .putExtra(EXTRA_AUTOMATION_ID, id)
        return PendingIntent.getBroadcast(
            appContext,
            id.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    const val MIN_INTERVAL_MINUTES = 15
    const val ACTION_RUN_AUTOMATION = "com.nousresearch.hermesagent.RUN_AUTOMATION"
    const val EXTRA_AUTOMATION_ID = "automation_id"
}
