package com.nousresearch.hermesagent.backend

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject

object HermesRuntimeManager {
    data class RuntimeState(
        val started: Boolean,
        val baseUrl: String? = null,
        val apiKey: String? = null,
        val hermesHome: String? = null,
        val modelName: String? = null,
        val probeResult: String? = null,
        val error: String? = null,
    )

    @Volatile
    private var currentState: RuntimeState = RuntimeState(started = false)

    @Synchronized
    fun ensureStarted(context: Context): RuntimeState {
        if (currentState.started && currentState.error == null) {
            return currentState
        }

        return try {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context.applicationContext))
            }
            val probeResult = PythonBootProbe.readProbe(context.applicationContext)
            val statusJson = Python.getInstance()
                .getModule("hermes_android.server_bridge")
                .callAttr("ensure_server", context.filesDir.absolutePath)
                .toString()
            val status = JSONObject(statusJson)
            currentState = RuntimeState(
                started = status.optBoolean("started", false),
                baseUrl = status.optString("base_url").ifBlank { null },
                apiKey = status.optString("api_server_key").ifBlank { null },
                hermesHome = status.optString("hermes_home").ifBlank { null },
                modelName = status.optString("api_server_model_name").ifBlank { null },
                probeResult = probeResult,
            )
            currentState
        } catch (exc: Throwable) {
            currentState = RuntimeState(
                started = false,
                error = exc.message ?: exc.toString(),
            )
            currentState
        }
    }

    @Synchronized
    fun stop(): RuntimeState {
        return try {
            if (Python.isStarted()) {
                Python.getInstance()
                    .getModule("hermes_android.server_bridge")
                    .callAttr("stop_server")
            }
            currentState = RuntimeState(started = false)
            currentState
        } catch (exc: Throwable) {
            currentState = RuntimeState(started = false, error = exc.message ?: exc.toString())
            currentState
        }
    }

    fun currentState(): RuntimeState = currentState
}
