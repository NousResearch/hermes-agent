package com.nousresearch.hermesagent.device

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import org.json.JSONArray
import org.json.JSONObject

object HermesVibrationActionBridge {
    fun vibrateJson(context: Context, payload: JSONObject): JSONObject {
        val durationMs = payload.optLong("duration_ms", 0L)
        val pattern = payload.optJSONArray("pattern_ms")
        val vibrator = defaultVibrator(context)
            ?: return errorJson("Android vibrator service is unavailable")
        val hardwareAvailable = runCatching { vibrator.hasVibrator() }.getOrDefault(true)
        return runCatching {
            if (hardwareAvailable) {
                if (pattern != null && pattern.length() > 1) {
                    vibratePattern(vibrator, pattern)
                } else {
                    vibrateOneShot(vibrator, durationMs)
                }
            }
            JSONObject()
                .put("success", true)
                .put("exit_code", 0)
                .put("action", "vibrate")
                .put("hardware_available", hardwareAvailable)
                .put("duration_ms", durationMs)
                .put("pattern_ms", pattern ?: JSONArray())
                .put(
                    "message",
                    if (hardwareAvailable) {
                        "Vibration requested"
                    } else {
                        "Vibration request accepted, but no vibrator hardware was reported"
                    },
                )
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    @Suppress("DEPRECATION")
    private fun vibrateOneShot(vibrator: Vibrator, durationMs: Long) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(durationMs, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            vibrator.vibrate(durationMs)
        }
    }

    @Suppress("DEPRECATION")
    private fun vibratePattern(vibrator: Vibrator, pattern: JSONArray) {
        val timings = LongArray(pattern.length()) { index -> pattern.optLong(index, 0L) }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(timings, -1))
        } else {
            vibrator.vibrate(timings, -1)
        }
    }

    private fun defaultVibrator(context: Context): Vibrator? {
        val appContext = context.applicationContext
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            appContext.getSystemService(VibratorManager::class.java)?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            appContext.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
    }

    private fun errorJson(message: String): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 1)
            .put("action", "vibrate")
            .put("error", message)
    }
}
