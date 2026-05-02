package com.nousresearch.hermesagent.device

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

object NativeAndroidShellTool {
    fun run(
        context: Context,
        command: String,
        timeoutSeconds: Long = 60,
    ): JSONObject {
        val state = HermesLinuxSubsystemBridge.ensureInstalled(context.applicationContext)
        val homeDir = File(state.getString("home_path")).apply { mkdirs() }
        val tmpDir = File(state.getString("tmp_path")).apply { mkdirs() }
        val shellPath = state.optString("shell_path", "/system/bin/sh").ifBlank { "/system/bin/sh" }

        val process = ProcessBuilder(shellPath, "-c", command)
            .directory(homeDir)
            .apply {
                environment().putAll(HermesLinuxSubsystemBridge.buildRunEnvironment(state))
            }
            .start()

        val executor = Executors.newFixedThreadPool(2)
        val stdout = executor.submit(Callable {
            process.inputStream.bufferedReader().use { it.readText() }
        })
        val stderr = executor.submit(Callable {
            process.errorStream.bufferedReader().use { it.readText() }
        })

        val completed = process.waitFor(timeoutSeconds, TimeUnit.SECONDS)
        if (!completed) {
            process.destroy()
        }
        val exitCode = if (completed) process.exitValue() else 124
        val output = stdout.get(1, TimeUnit.SECONDS)
        val error = stderr.get(1, TimeUnit.SECONDS)
        executor.shutdownNow()

        return JSONObject()
            .put("exit_code", exitCode)
            .put("output", output)
            .put("error", error)
            .put("cwd", homeDir.absolutePath)
            .put("shell", shellPath)
    }

}
