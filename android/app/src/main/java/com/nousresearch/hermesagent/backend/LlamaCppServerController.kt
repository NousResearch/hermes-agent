package com.nousresearch.hermesagent.backend

import android.content.Context
import android.system.Os
import android.system.OsConstants
import com.nousresearch.hermesagent.device.HermesLinuxSubsystemBridge
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.util.Locale
import java.util.concurrent.TimeUnit

object LlamaCppServerController {
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(750, TimeUnit.MILLISECONDS)
        .readTimeout(750, TimeUnit.MILLISECONDS)
        .writeTimeout(750, TimeUnit.MILLISECONDS)
        .build()

    @Volatile private var process: Process? = null
    @Volatile private var activeModelPath: String = ""
    @Volatile private var activeModelName: String = ""
    @Volatile private var recentLog: String = ""

    @Synchronized
    fun ensureRunning(
        context: Context,
        modelPath: String,
        requestedModelName: String,
        port: Int,
    ): LocalBackendStatus {
        val currentProcess = process
        if (currentProcess != null && currentProcess.isAlive && activeModelPath == modelPath && checkReady(port)) {
            return LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = true,
                baseUrl = "http://127.0.0.1:$port/v1",
                modelName = actualModelName(port, requestedModelName),
                sourceModelPath = modelPath,
                statusMessage = "llama.cpp is serving locally from the embedded Linux suite",
            )
        }

        stop()
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(context)
        val shellPath = linuxState.optString("shell_path").ifBlank { linuxState.optString("bash_path") }
        val prefixPath = linuxState.optString("prefix_path")
        val homePath = linuxState.optString("home_path")
        val llamaServerPath = selectLlamaServerPath(context, linuxState)
        if (shellPath.isBlank() || prefixPath.isBlank()) {
            return LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = false,
                sourceModelPath = modelPath,
                statusMessage = "The embedded Linux suite is not ready yet for llama.cpp",
            )
        }
        if (!File(llamaServerPath).canExecute()) {
            val fallbackReason = linuxState.optString("fallback_reason").ifBlank {
                "embedded Linux shell could not be launched"
            }
            val shellModeHint = if (linuxState.optString("execution_mode") == "android_system_shell") {
                " Native Android shell fallback reason: $fallbackReason."
            } else {
                ""
            }
            return LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = false,
                sourceModelPath = modelPath,
                statusMessage = "llama.cpp executable is not available at $llamaServerPath.$shellModeHint Use LiteRT-LM .litertlm models for fully native local inference.",
            )
        }

        val command = buildString {
            append("exec ")
            append(shellQuote(llamaServerPath))
            append(" ")
            append("--model ")
            append(shellQuote(modelPath))
            append(" --host 127.0.0.1 --port ")
            append(port)
            append(" ")
            append(launchOptionsForModel(modelPath))
        }

        return try {
            val shellArgs = if (shellPath.endsWith("/sh")) {
                listOf(shellPath, "-c", command)
            } else {
                listOf(shellPath, "-lc", command)
            }
            val startedProcess = ProcessBuilder(shellArgs)
                .directory(File(homePath.ifBlank { prefixPath }))
                .redirectErrorStream(true)
                .apply {
                    environment().putAll(HermesLinuxSubsystemBridge.buildRunEnvironment(linuxState))
                }
                .start()
            process = startedProcess
            activeModelPath = modelPath
            activeModelName = requestedModelName
            drainLogs(startedProcess)
            if (!waitUntilReady(port)) {
                val errorTail = recentLog.takeLast(600)
                stop()
                return LocalBackendStatus(
                    backendKind = BackendKind.LLAMA_CPP,
                    started = false,
                    sourceModelPath = modelPath,
                    statusMessage = if (errorTail.isBlank()) {
                        "llama.cpp failed to become ready"
                    } else {
                        "llama.cpp failed to become ready: $errorTail"
                    },
                )
            }
            LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = true,
                baseUrl = "http://127.0.0.1:$port/v1",
                modelName = actualModelName(port, requestedModelName),
                sourceModelPath = modelPath,
                statusMessage = "llama.cpp is serving locally from ${llamaServerOriginLabel(linuxState)}${llamaServerCompatibilitySuffix(llamaServerPath)}",
            )
        } catch (error: Throwable) {
            stop()
            LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = false,
                sourceModelPath = modelPath,
                statusMessage = error.message ?: error.javaClass.simpleName,
            )
        }
    }

    @Synchronized
    fun stop() {
        process?.let { current ->
            runCatching {
                current.destroy()
                if (!current.waitFor(1200, TimeUnit.MILLISECONDS)) {
                    current.destroyForcibly()
                    current.waitFor(1200, TimeUnit.MILLISECONDS)
                }
            }
        }
        process = null
        activeModelPath = ""
        activeModelName = ""
        recentLog = ""
    }

    private fun shellQuote(value: String): String {
        return "'" + value.replace("'", "'\\''") + "'"
    }

    private fun selectLlamaServerPath(context: Context, linuxState: JSONObject): String {
        val defaultPath = linuxState.optString("native_llama_server_path").ifBlank { "llama-server" }
        val bionicSpawnPath = bionicLlamaServerPath(context, linuxState)
        if (linuxState.optString("execution_mode") == "android_system_shell" && bionicSpawnPath.isFile) {
            return bionicSpawnPath.absolutePath
        }
        val pageSize = devicePageSizeBytes()
        if (pageSize < ANDROID_16K_PAGE_SIZE_BYTES) {
            return defaultPath
        }
        return if (bionicSpawnPath.isFile) bionicSpawnPath.absolutePath else defaultPath
    }

    private fun bionicLlamaServerPath(context: Context, linuxState: JSONObject): File {
        return File(
            linuxState.optString("bionic_llama_server_path").ifBlank {
                val nativeDir = linuxState.optString("native_library_dir")
                    .ifBlank { context.applicationInfo.nativeLibraryDir.orEmpty() }
                File(nativeDir, LEGACY_BIONIC_SPAWN_LLAMA_SERVER_LIBRARY_NAME).absolutePath
            }
        )
    }

    private fun llamaServerOriginLabel(linuxState: JSONObject): String {
        return if (linuxState.optString("execution_mode") == "android_system_shell") {
            "Android's extracted native-library directory"
        } else {
            "the embedded Linux suite"
        }
    }

    private fun llamaServerCompatibilitySuffix(llamaServerPath: String): String {
        return if (llamaServerPath.endsWith(BIONIC_LLAMA_SERVER_NAME) ||
            llamaServerPath.endsWith(LEGACY_BIONIC_SPAWN_LLAMA_SERVER_LIBRARY_NAME)
        ) {
            " using the Android 16 KB page-size libc posix_spawn compatibility launcher"
        } else {
            ""
        }
    }

    private fun drainLogs(startedProcess: Process) {
        Thread {
            runCatching {
                BufferedReader(InputStreamReader(startedProcess.inputStream)).use { reader ->
                    while (true) {
                        val line = reader.readLine() ?: break
                        recentLog = (recentLog + "\n" + line).takeLast(4000)
                    }
                }
            }
        }.start()
    }

    private fun waitUntilReady(port: Int): Boolean {
        repeat(LLAMA_CPP_READY_CHECKS) {
            if (checkReady(port)) {
                return true
            }
            Thread.sleep(250)
        }
        return false
    }

    private fun checkReady(port: Int): Boolean {
        val request = Request.Builder().url("http://127.0.0.1:$port/v1/models").get().build()
        return runCatching {
            httpClient.newCall(request).execute().use { response ->
                response.isSuccessful
            }
        }.getOrDefault(false)
    }

    private fun actualModelName(port: Int, fallback: String): String {
        val request = Request.Builder().url("http://127.0.0.1:$port/v1/models").get().build()
        return runCatching {
            httpClient.newCall(request).execute().use { response ->
                val body = response.body?.string().orEmpty()
                if (!response.isSuccessful) {
                    return@use fallback
                }
                val data = JSONObject(body).optJSONArray("data")
                data?.optJSONObject(0)?.optString("id")?.ifBlank { fallback } ?: fallback
            }
        }.getOrDefault(fallback)
    }

    internal fun launchOptionsForModel(
        modelPath: String,
        availableProcessors: Int = Runtime.getRuntime().availableProcessors(),
    ): String {
        val lower = modelPath.lowercase(Locale.US)
        val ctxSize = when {
            "0.8b" in lower || "0-8b" in lower || "0_8b" in lower -> 1024
            "0.6b" in lower || "0-6b" in lower || "0_6b" in lower -> 1024
            else -> 2048
        }
        val threads = availableProcessors.coerceIn(1, 4)
        return "--ctx-size $ctxSize --parallel 1 --threads $threads --batch-size 64 --ubatch-size 64 --no-warmup"
    }

    private fun devicePageSizeBytes(): Long {
        return runCatching { Os.sysconf(OsConstants._SC_PAGESIZE) }.getOrDefault(4096L)
    }

    private const val ANDROID_16K_PAGE_SIZE_BYTES = 16_384L
    private const val LLAMA_CPP_READY_CHECKS = 720
    private const val BIONIC_LLAMA_SERVER_NAME = "llama-server-bionic"
    private const val LEGACY_BIONIC_SPAWN_LLAMA_SERVER_LIBRARY_NAME = "libhermes_android_llama_server_bionic_spawn.so"
}
