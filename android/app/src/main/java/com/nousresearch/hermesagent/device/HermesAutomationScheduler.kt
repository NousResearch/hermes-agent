package com.nousresearch.hermesagent.device

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.SystemClock
import java.util.Calendar

object HermesAutomationScheduler {
    fun schedule(context: Context, record: HermesAutomationRecord) {
        if (!record.enabled) {
            cancel(context, record.id)
            return
        }
        when (record.triggerType) {
            TRIGGER_INTERVAL -> scheduleInterval(context, record)
            TRIGGER_TIME -> scheduleTime(context, record)
            else -> cancel(context, record.id)
        }
    }

    private fun scheduleInterval(context: Context, record: HermesAutomationRecord) {
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

    private fun scheduleTime(context: Context, record: HermesAutomationRecord) {
        val triggerAtMillis = nextTimeTriggerAtMillis(
            nowEpochMs = System.currentTimeMillis(),
            timeMinutes = record.triggerTimeMinutes ?: return,
            daysOfWeekCsv = record.triggerDaysOfWeek,
        ) ?: return cancel(context, record.id)
        val alarmManager = context.applicationContext.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        alarmManager.set(
            AlarmManager.RTC_WAKEUP,
            triggerAtMillis,
            pendingIntent(context, record.id),
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

    internal fun nextTimeTriggerAtMillis(
        nowEpochMs: Long,
        timeMinutes: Int,
        daysOfWeekCsv: String = "",
    ): Long? {
        if (timeMinutes !in 0..1439) {
            return null
        }
        val allowedDays = parseDaysOfWeek(daysOfWeekCsv)
        val now = Calendar.getInstance().apply {
            timeInMillis = nowEpochMs
        }
        for (dayOffset in 0..7) {
            val candidate = Calendar.getInstance().apply {
                timeInMillis = nowEpochMs
                add(Calendar.DAY_OF_YEAR, dayOffset)
                set(Calendar.HOUR_OF_DAY, timeMinutes / 60)
                set(Calendar.MINUTE, timeMinutes % 60)
                set(Calendar.SECOND, 0)
                set(Calendar.MILLISECOND, 0)
            }
            if (candidate.timeInMillis <= now.timeInMillis) {
                continue
            }
            if (allowedDays.isEmpty() || candidate.get(Calendar.DAY_OF_WEEK) in allowedDays) {
                return candidate.timeInMillis
            }
        }
        return null
    }

    private fun parseDaysOfWeek(daysOfWeekCsv: String): Set<Int> {
        if (daysOfWeekCsv.isBlank()) {
            return emptySet()
        }
        return daysOfWeekCsv
            .split(',')
            .mapNotNull { token -> DAY_TO_CALENDAR[token.trim().uppercase()] }
            .toSet()
    }

    const val MIN_INTERVAL_MINUTES = 15
    const val ACTION_RUN_AUTOMATION = "com.nousresearch.hermesagent.RUN_AUTOMATION"
    const val EXTRA_AUTOMATION_ID = "automation_id"

    private val DAY_TO_CALENDAR = mapOf(
        "SUN" to Calendar.SUNDAY,
        "MON" to Calendar.MONDAY,
        "TUE" to Calendar.TUESDAY,
        "WED" to Calendar.WEDNESDAY,
        "THU" to Calendar.THURSDAY,
        "FRI" to Calendar.FRIDAY,
        "SAT" to Calendar.SATURDAY,
    )
}
