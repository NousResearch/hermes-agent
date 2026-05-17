package com.nousresearch.hermesagent

import android.app.Application
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
import com.nousresearch.hermesagent.ui.chat.NativeToolCallingChatClient
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.FileInputStream
import java.net.ServerSocket

@RunWith(AndroidJUnit4::class)
class HermesGoalAppLaunchInstrumentedTest {
    private val app: Application
        get() = ApplicationProvider.getApplicationContext()

    @After
    fun tearDown() {
        HermesRuntimeManager.stop()
        OnDeviceBackendManager.stopAll()
    }

    @Test
    fun nativeToolLoopCanLaunchGoalAppsWithoutPosting() {
        val targets = listOf(
            "com.google.android.gm",
            "com.zhiliaoapp.musically",
            "com.instagram.android",
        )

        targets.forEach { packageName ->
            executeShell("input keyevent HOME")
            SystemClock.sleep(500)

            val port = freePort()
            val server = AppLaunchChatServer(port, packageName)
            server.start(30_000, false)
            try {
                val result = NativeToolCallingChatClient(app).send(
                    baseUrl = "http://127.0.0.1:$port",
                    modelName = "scripted-app-launch-tool-model",
                    sessionId = "goal-app-launch-$packageName",
                    userText = "Open $packageName using android_ui_tool launch_app, without typing, posting, sending, or clicking inside it.",
                )

                assertEquals("Expected exactly one app-launch tool call for $packageName", 1, result.executedToolCalls)
                assertTrue(result.content, result.content.contains("opened $packageName"))
                assertTrue(server.requests.toString(), server.requests.last().toString().contains("Opened $packageName"))
                val foregroundPackage = waitForForegroundPackage(packageName)
                assertEquals("Expected $packageName in the foreground", packageName, foregroundPackage)
            } finally {
                server.stop()
            }
        }
    }

    private fun waitForForegroundPackage(expectedPackage: String, timeoutMs: Long = 10_000): String {
        val deadline = SystemClock.uptimeMillis() + timeoutMs
        var latest = currentForegroundPackage()
        while (SystemClock.uptimeMillis() < deadline) {
            if (latest == expectedPackage) {
                return latest
            }
            SystemClock.sleep(250)
            latest = currentForegroundPackage().ifBlank { latest }
        }
        return latest
    }

    private fun currentForegroundPackage(): String {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val rootPackage = instrumentation.uiAutomation.rootInActiveWindow?.packageName?.toString().orEmpty()
        if (rootPackage.isNotBlank()) {
            return rootPackage
        }
        val windowState = executeShell("dumpsys window")
        Regex("""mCurrentFocus=.*?\s([A-Za-z0-9_.]+)/""")
            .find(windowState)
            ?.groups
            ?.get(1)
            ?.value
            ?.takeIf { it.isNotBlank() }
            ?.let { return it }
        return Regex("""mFocusedApp=.*?\s([A-Za-z0-9_.]+)/""")
            .find(windowState)
            ?.groups
            ?.get(1)
            ?.value
            .orEmpty()
    }

    private fun executeShell(command: String): String {
        val descriptor = InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand(command)
        return FileInputStream(descriptor.fileDescriptor).bufferedReader().use { reader ->
            reader.readText()
        }.also {
            descriptor.close()
        }
    }

    private fun freePort(): Int {
        return ServerSocket(0).use { it.localPort }
    }

    private class AppLaunchChatServer(
        port: Int,
        private val packageName: String,
    ) : NanoHTTPD("127.0.0.1", port) {
        val requests = mutableListOf<JSONObject>()

        override fun serve(session: IHTTPSession): Response {
            return if (session.method == Method.POST && session.uri == "/v1/chat/completions") {
                val files = HashMap<String, String>()
                session.parseBody(files)
                requests += JSONObject(files["postData"].orEmpty().ifBlank { "{}" })
                val payload = if (requests.size == 1) toolCallPayload() else finalPayload()
                newFixedLengthResponse(Response.Status.OK, "application/json", payload.toString())
            } else {
                newFixedLengthResponse(Response.Status.NOT_FOUND, "application/json", JSONObject().put("error", "not found").toString())
            }
        }

        private fun toolCallPayload(): JSONObject {
            val message = JSONObject()
                .put("role", "assistant")
                .put("content", JSONObject.NULL)
                .put(
                    "tool_calls",
                    JSONArray().put(
                        toolCall(
                            id = "call_launch_${packageName.replace('.', '_')}",
                            name = "android_ui_tool",
                            arguments = JSONObject()
                                .put("action", "launch_app")
                                .put("package_name", packageName),
                        ),
                    ),
                )
            return completionPayload(message, "tool_calls")
        }

        private fun finalPayload(): JSONObject {
            return completionPayload(
                JSONObject()
                    .put("role", "assistant")
                    .put("content", "opened $packageName"),
                "stop",
            )
        }

        private fun toolCall(id: String, name: String, arguments: JSONObject): JSONObject {
            return JSONObject()
                .put("id", id)
                .put("type", "function")
                .put(
                    "function",
                    JSONObject()
                        .put("name", name)
                        .put("arguments", arguments.toString()),
                )
        }

        private fun completionPayload(message: JSONObject, finishReason: String): JSONObject {
            return JSONObject()
                .put("id", "chatcmpl-goal-app-launch")
                .put("object", "chat.completion")
                .put("created", System.currentTimeMillis() / 1000)
                .put("model", "scripted-app-launch-tool-model")
                .put(
                    "choices",
                    JSONArray().put(
                        JSONObject()
                            .put("index", 0)
                            .put("message", message)
                            .put("finish_reason", finishReason),
                    ),
                )
        }
    }
}
