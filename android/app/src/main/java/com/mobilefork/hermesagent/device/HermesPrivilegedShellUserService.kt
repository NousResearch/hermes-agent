package com.mobilefork.hermesagent.device

import android.content.Context
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

class HermesPrivilegedShellUserService : IHermesPrivilegedShellService.Stub {
    constructor() : super()
    constructor(@Suppress("UNUSED_PARAMETER") context: Context) : super()

    init {
        scheduleProcessExit(IDLE_PROCESS_EXIT_DELAY_MS, "hermes-shizuku-service-idle-exit")
    }

    override fun runCommand(command: String, timeoutSeconds: Int): String {
        val normalizedCommand = command.trim()
        if (normalizedCommand.isBlank()) {
            return finish(
                JSONObject()
                .put("success", false)
                .put("exit_code", 2)
                .put("error", "run_privileged_shell requires a non-empty command"),
            )
        }
        if (normalizedCommand.indexOf('\u0000') >= 0) {
            return finish(
                JSONObject()
                .put("success", false)
                .put("exit_code", 2)
                .put("error", "run_privileged_shell command must not contain NUL bytes"),
            )
        }

        val timeout = timeoutSeconds.coerceIn(1, MAX_TIMEOUT_SECONDS)
        return finish(runCatching {
            val process = ProcessBuilder("/system/bin/sh", "-c", normalizedCommand).start()
            val stdout = AtomicReference("")
            val stderr = AtomicReference("")
            val readersDone = CountDownLatch(2)
            Thread {
                try {
                    stdout.set(readLimited(process.inputStream))
                } finally {
                    readersDone.countDown()
                }
            }.apply {
                name = "hermes-shizuku-stdout"
                isDaemon = true
                start()
            }
            Thread {
                try {
                    stderr.set(readLimited(process.errorStream))
                } finally {
                    readersDone.countDown()
                }
            }.apply {
                name = "hermes-shizuku-stderr"
                isDaemon = true
                start()
            }

            val finished = process.waitFor(timeout.toLong(), TimeUnit.SECONDS)
            if (!finished) {
                process.destroy()
            }
            readersDone.await(2, TimeUnit.SECONDS)
            val exitCode = if (finished) process.exitValue() else 124
            JSONObject()
                .put("success", finished && exitCode == 0)
                .put("exit_code", exitCode)
                .put("output", stdout.get())
                .put("error", stderr.get())
                .put("timed_out", !finished)
                .put("uid", android.os.Process.myUid())
                .put("privilege_context", "shizuku_user_service")
        }.getOrElse { error ->
            JSONObject()
                .put("success", false)
                .put("exit_code", -1)
                .put("error", error.message ?: error.javaClass.simpleName)
                .put("privilege_context", "shizuku_user_service")
        })
    }

    private fun finish(payload: JSONObject): String {
        scheduleProcessExit(PROCESS_EXIT_DELAY_MS, "hermes-shizuku-service-exit")
        return payload.toString()
    }

    private fun scheduleProcessExit(delayMs: Long, threadName: String) {
        Thread {
            try {
                Thread.sleep(delayMs)
            } finally {
                android.os.Process.killProcess(android.os.Process.myPid())
            }
        }.apply {
            name = threadName
            isDaemon = true
            start()
        }
    }

    private fun readLimited(input: InputStream): String {
        return input.use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            val output = ByteArrayOutputStream()
            var truncated = false
            while (true) {
                val read = stream.read(buffer)
                if (read < 0) break
                val remaining = MAX_CAPTURE_BYTES - output.size()
                if (remaining <= 0) {
                    truncated = true
                    break
                }
                output.write(buffer, 0, minOf(read, remaining))
                if (read > remaining) {
                    truncated = true
                    break
                }
            }
            buildString {
                append(output.toByteArray().toString(Charsets.UTF_8))
                if (truncated) {
                    append("\n[truncated]")
                }
            }
        }
    }

    private companion object {
        private const val MAX_CAPTURE_BYTES = 16_384
        private const val MAX_TIMEOUT_SECONDS = 120
        private const val PROCESS_EXIT_DELAY_MS = 750L
        private const val IDLE_PROCESS_EXIT_DELAY_MS = 180_000L
    }
}
