package com.nousresearch.hermesagent.backend

import android.content.Context
import com.nousresearch.hermesagent.device.HermesLinuxSubsystemBridge
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
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
        if (linuxState.optString("execution_mode") == "android_system_shell") {
            val fallbackReason = linuxState.optString("fallback_reason").ifBlank {
                "embedded Linux shell could not be launched"
            }
            return LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = false,
                sourceModelPath = modelPath,
                statusMessage = "llama.cpp is not available in native Android shell mode: $fallbackReason. Use LiteRT-LM .litertlm models for fully native local inference.",
            )
        }
        val bashPath = linuxState.optString("bash_path")
        val prefixPath = linuxState.optString("prefix_path")
        val homePath = linuxState.optString("home_path")
        val llamaServerPath = linuxState.optString("native_llama_server_path").ifBlank { "llama-server" }
        if (bashPath.isBlank() || prefixPath.isBlank()) {
            return LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = false,
                sourceModelPath = modelPath,
                statusMessage = "The embedded Linux suite is not ready yet for llama.cpp",
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
            append(" --ctx-size 4096 --parallel 1")
        }

        return try {
            val startedProcess = ProcessBuilder(listOf(bashPath, "-lc", command))
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
                statusMessage = "llama.cpp is serving locally from the embedded Linux suite",
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
        repeat(360) {
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
}
