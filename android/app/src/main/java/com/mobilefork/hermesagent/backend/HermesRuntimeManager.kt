package com.mobilefork.hermesagent.backend

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.mobilefork.hermesagent.data.AppSettingsStore
import com.mobilefork.hermesagent.data.ProviderPresets
import com.mobilefork.hermesagent.device.HermesLinuxSubsystemBridge
import java.io.File
import org.json.JSONObject

object HermesRuntimeManager {
    private val pythonStartLock = Any()

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

    fun ensurePythonStarted(context: Context) {
        if (Python.isStarted()) {
            return
        }

        val appContext = context.applicationContext
        synchronized(pythonStartLock) {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(appContext))
            }
        }
    }

    @Synchronized
    fun ensureStarted(context: Context): RuntimeState {
        val appContext = context.applicationContext
        val settings = AppSettingsStore(appContext).load()
        val selectedLocalBackend = BackendKind.fromPersistedValue(settings.onDeviceBackend)
        if (
            selectedLocalBackend == BackendKind.NONE &&
            currentState.started &&
            currentState.error == null &&
            !currentState.baseUrl.isNullOrBlank() &&
            !currentState.apiKey.isNullOrBlank()
        ) {
            return currentState
        }

        return try {
            HermesLinuxSubsystemBridge.ensureInstalled(appContext)
            refreshPythonRuntimeEnvironment(appContext)
            val localBackendStatus = OnDeviceBackendManager.ensureConfigured(
                appContext,
                settings.onDeviceBackend,
            )
            if (localBackendStatus.started) {
                currentState = RuntimeState(
                    started = true,
                    baseUrl = localBackendStatus.baseUrl,
                    hermesHome = File(appContext.filesDir, "hermes-home").absolutePath,
                    modelName = localBackendStatus.modelName,
                    probeResult = "native-android-litert-lm",
                )
                return currentState
            }
            if (settings.offlineAirplaneMode) {
                currentState = RuntimeState(
                    started = false,
                    hermesHome = File(appContext.filesDir, "hermes-home").absolutePath,
                    error = localBackendStatus.statusMessage.ifBlank {
                        "Offline airplane mode is on and no on-device backend is ready."
                    },
                )
                return currentState
            }
            val localBackendFallbackWarning =
                localBackendFallbackWarning(selectedLocalBackend, localBackendStatus)

            ensurePythonStarted(appContext)
            refreshPythonRuntimeEnvironment(appContext)
            val effectiveProvider = settings.provider
            val effectiveModel = settings.model
            val effectiveBaseUrl = ProviderPresets.runtimeConfigBaseUrl(settings.provider, settings.baseUrl)
            Python.getInstance().getModule("hermes_android.config_bridge").callAttr(
                "write_runtime_config",
                effectiveProvider,
                effectiveModel,
                effectiveBaseUrl,
            )
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
                probeResult = probeResult.withLocalBackendWarning(localBackendFallbackWarning),
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

    private fun refreshPythonRuntimeEnvironment(context: Context) {
        if (!Python.isStarted()) {
            return
        }
        runCatching {
            Python.getInstance()
                .getModule("hermes_android.runtime_env")
                .callAttr("prepare_runtime_env", context.filesDir.absolutePath)
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

    internal fun localBackendFallbackWarning(
        selectedLocalBackend: BackendKind,
        localBackendStatus: LocalBackendStatus,
    ): String? {
        if (selectedLocalBackend == BackendKind.NONE || localBackendStatus.started) {
            return null
        }
        val reason = localBackendStatus.statusMessage.ifBlank {
            "Selected local backend ${selectedLocalBackend.persistedValue} did not start."
        }
        return "Local ${selectedLocalBackend.persistedValue} backend unavailable: $reason. " +
            "Using saved remote provider."
    }

    internal fun String?.withLocalBackendWarning(warning: String?): String? {
        val trimmedWarning = warning.orEmpty().trim()
        val trimmedProbe = orEmpty().trim()
        return when {
            trimmedWarning.isBlank() -> trimmedProbe.ifBlank { null }
            trimmedProbe.isBlank() -> trimmedWarning
            else -> "$trimmedProbe\n$trimmedWarning"
        }
    }
}
