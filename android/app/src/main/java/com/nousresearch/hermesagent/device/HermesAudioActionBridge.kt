package com.nousresearch.hermesagent.device

import android.app.NotificationManager
import android.content.Context
import android.media.AudioManager
import android.os.Build
import android.provider.Settings
import org.json.JSONArray
import org.json.JSONObject

object HermesAudioActionBridge {
    fun performAudioActionJson(context: Context, payload: JSONObject): JSONObject {
        val action = normalizeAudioAction(payload.optString("audio_action"))
            ?: return errorJson("Unsupported audio action: ${payload.optString("audio_action")}")
        val audioManager = context.applicationContext.getSystemService(Context.AUDIO_SERVICE) as? AudioManager
            ?: return errorJson("Android audio service is unavailable")
        return runCatching {
            when (action) {
                AUDIO_ACTION_SET_VOLUME -> setVolume(audioManager, payload)
                AUDIO_ACTION_SET_RINGER_MODE -> setRingerMode(context, audioManager, payload)
                AUDIO_ACTION_SET_MICROPHONE_MUTE -> setMicrophoneMute(audioManager, payload)
                AUDIO_ACTION_SET_SPEAKERPHONE -> setSpeakerphone(audioManager, payload)
                else -> errorJson("Unsupported audio action after normalization: $action")
            }
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    fun normalizeAudioAction(value: String): String? {
        return when (value.trim().lowercase().replace("-", "_").replace(" ", "_")) {
            "set_volume", "volume", "set_stream_volume", "stream_volume" -> AUDIO_ACTION_SET_VOLUME
            "set_ringer_mode", "ringer_mode", "sound_mode", "set_sound_mode" -> AUDIO_ACTION_SET_RINGER_MODE
            "set_microphone_mute", "microphone_mute", "mic_mute", "mute_microphone" -> AUDIO_ACTION_SET_MICROPHONE_MUTE
            "set_speakerphone", "speakerphone", "speaker_phone" -> AUDIO_ACTION_SET_SPEAKERPHONE
            else -> null
        }
    }

    fun normalizeAudioStream(value: String): String? {
        return when (value.trim().lowercase().replace("-", "_").replace(" ", "_")) {
            "alarm", "alarms" -> "alarm"
            "ring", "ringer", "ringtone" -> "ringer"
            "notification", "notifications" -> "notification"
            "call", "voice_call", "in_call", "incall" -> "voice_call"
            "media", "music" -> "media"
            "system" -> "system"
            "dtmf", "tone" -> "dtmf"
            "bluetooth", "bluetooth_sco", "bt_voice", "bt_sco" -> "voice_call"
            "accessibility" -> "accessibility"
            else -> null
        }
    }

    fun normalizeRingerMode(value: String): String? {
        return when (value.trim().lowercase().replace("-", "_").replace(" ", "_")) {
            "normal", "sound", "ring", "on" -> "normal"
            "vibrate", "vibration" -> "vibrate"
            "silent", "mute", "off" -> "silent"
            else -> null
        }
    }

    fun supportedStreamsJson(): JSONArray = JSONArray(SUPPORTED_STREAMS)

    private fun setVolume(audioManager: AudioManager, payload: JSONObject): JSONObject {
        val streamName = normalizeAudioStream(payload.optString("stream"))
            ?: return errorJson("set_volume requires a supported stream")
        val level = payload.optInt("level", -1)
        if (level < 0) {
            return errorJson("set_volume requires level >= 0")
        }
        val stream = streamId(streamName)
            ?: return errorJson("Audio stream is not supported on this Android version: $streamName")
        val max = audioManager.getStreamMaxVolume(stream)
        val applied = level.coerceIn(0, max)
        audioManager.setStreamVolume(stream, applied, 0)
        return JSONObject()
            .put("success", true)
            .put("exit_code", 0)
            .put("action", AUDIO_ACTION_SET_VOLUME)
            .put("stream", streamName)
            .put("requested_level", level)
            .put("applied_level", applied)
            .put("max_level", max)
            .put("message", "Audio stream volume updated")
    }

    private fun setRingerMode(context: Context, audioManager: AudioManager, payload: JSONObject): JSONObject {
        val modeName = normalizeRingerMode(payload.optString("ringer_mode"))
            ?: return errorJson("set_ringer_mode requires normal, vibrate, or silent")
        val mode = when (modeName) {
            "normal" -> AudioManager.RINGER_MODE_NORMAL
            "vibrate" -> AudioManager.RINGER_MODE_VIBRATE
            else -> AudioManager.RINGER_MODE_SILENT
        }
        try {
            audioManager.ringerMode = mode
        } catch (error: SecurityException) {
            return JSONObject()
                .put("success", false)
                .put("exit_code", 2)
                .put("action", AUDIO_ACTION_SET_RINGER_MODE)
                .put("ringer_mode", modeName)
                .put("requires_notification_policy_access", true)
                .put("settings_action", Settings.ACTION_NOTIFICATION_POLICY_ACCESS_SETTINGS)
                .put("error", error.message ?: "Notification policy access is required to change ringer mode")
        }
        return JSONObject()
            .put("success", true)
            .put("exit_code", 0)
            .put("action", AUDIO_ACTION_SET_RINGER_MODE)
            .put("ringer_mode", modeName)
            .put("notification_policy_access_granted", notificationPolicyAccessGranted(context))
            .put("message", "Ringer mode updated")
    }

    @Suppress("DEPRECATION")
    private fun setMicrophoneMute(audioManager: AudioManager, payload: JSONObject): JSONObject {
        val enabled = payload.optBoolean("target_enabled", false)
        audioManager.isMicrophoneMute = enabled
        return JSONObject()
            .put("success", true)
            .put("exit_code", 0)
            .put("action", AUDIO_ACTION_SET_MICROPHONE_MUTE)
            .put("target_enabled", enabled)
            .put("message", "Microphone mute state requested")
    }

    @Suppress("DEPRECATION")
    private fun setSpeakerphone(audioManager: AudioManager, payload: JSONObject): JSONObject {
        val enabled = payload.optBoolean("target_enabled", false)
        audioManager.isSpeakerphoneOn = enabled
        return JSONObject()
            .put("success", true)
            .put("exit_code", 0)
            .put("action", AUDIO_ACTION_SET_SPEAKERPHONE)
            .put("target_enabled", enabled)
            .put("message", "Speakerphone state requested")
    }

    private fun notificationPolicyAccessGranted(context: Context): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) {
            return true
        }
        val notificationManager = context.applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as? NotificationManager
            ?: return false
        return notificationManager.isNotificationPolicyAccessGranted
    }

    private fun streamId(name: String): Int? {
        return when (name) {
            "alarm" -> AudioManager.STREAM_ALARM
            "ringer" -> AudioManager.STREAM_RING
            "notification" -> AudioManager.STREAM_NOTIFICATION
            "voice_call" -> AudioManager.STREAM_VOICE_CALL
            "media" -> AudioManager.STREAM_MUSIC
            "system" -> AudioManager.STREAM_SYSTEM
            "dtmf" -> AudioManager.STREAM_DTMF
            "accessibility" -> if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) AudioManager.STREAM_ACCESSIBILITY else null
            else -> null
        }
    }

    private fun errorJson(message: String): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 1)
            .put("action", "audio_action")
            .put("error", message)
            .put("supported_streams", supportedStreamsJson())
    }

    private val SUPPORTED_STREAMS = listOf(
        "alarm",
        "ringer",
        "notification",
        "voice_call",
        "media",
        "system",
        "dtmf",
        "accessibility",
    )

    private const val AUDIO_ACTION_SET_VOLUME = "set_volume"
    private const val AUDIO_ACTION_SET_RINGER_MODE = "set_ringer_mode"
    private const val AUDIO_ACTION_SET_MICROPHONE_MUTE = "set_microphone_mute"
    private const val AUDIO_ACTION_SET_SPEAKERPHONE = "set_speakerphone"
}
