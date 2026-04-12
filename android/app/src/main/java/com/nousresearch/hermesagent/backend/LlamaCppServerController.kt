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
        .connectTimeout(2, TimeUnit.SECONDS)
        .readTimeout(2, TimeUnit.SECONDS)
        .writeTimeout(2, TimeUnit.SECONDS)
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
        val bashPath = linuxState.optString("bash_path")
        val prefixPath = linuxState.optString("prefix_path")
        val binPath = linuxState.optString("bin_path")
        val libPath = linuxState.optString("lib_path")
        val homePath = linuxState.optString("home_path")
        val tmpPath = linuxState.optString("tmp_path")
        if (bashPath.isBlank() || prefixPath.isBlank()) {
            return LocalBackendStatus(
                backendKind = BackendKind.LLAMA_CPP,
                started = false,
                sourceModelPath = modelPath,
                statusMessage = "The embedded Linux suite is not ready yet for llama.cpp",
            )
        }

        val command = buildString {
            append("exec llama-server ")
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
                    environment().putAll(
                        buildRunEnvironment(
                            prefixPath = prefixPath,
                            binPath = binPath,
                            libPath = libPath,
                            homePath = homePath,
                            tmpPath = tmpPath,
                        )
                    )
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

    private fun buildRunEnvironment(
        prefixPath: String,
        binPath: String,
        libPath: String,
        homePath: String,
        tmpPath: String,
    ): Map<String, String> {
        val env = mutableMapOf<String, String>()
        env["PREFIX"] = prefixPath
        env["TERMUX_PREFIX"] = prefixPath
        env["PATH"] = listOf(binPath, System.getenv("PATH").orEmpty()).filter { it.isNotBlank() }.joinToString(":")
        env["LD_LIBRARY_PATH"] = listOf(libPath, System.getenv("LD_LIBRARY_PATH").orEmpty()).filter { it.isNotBlank() }.joinToString(":")
        env["HOME"] = homePath.ifBlank { prefixPath }
        env["TMPDIR"] = tmpPath.ifBlank { homePath.ifBlank { prefixPath } }
        env["TERM"] = "xterm-256color"
        env["LANG"] = "C.UTF-8"
        return env
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
        repeat(80) {
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
