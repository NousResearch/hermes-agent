package com.nousresearch.hermesagent

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
import com.nousresearch.hermesagent.device.HermesLinuxSubsystemBridge
import com.nousresearch.hermesagent.device.HermesSystemControlBridge
import com.nousresearch.hermesagent.ui.chat.NativeToolCallingChatClient
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.net.ServerSocket

@RunWith(AndroidJUnit4::class)
class NativeAgentToolAccessInstrumentedTest {
    private val app: Application
        get() = ApplicationProvider.getApplicationContext()

    @After
    fun tearDown() {
        HermesRuntimeManager.stop()
        OnDeviceBackendManager.stopAll()
    }

    @Test
    fun nativeChatToolLoopCanCreateDeleteAndReadSystemStatus() {
        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(app)
        val workspace = File(linuxState.getString("home_path"))
        val createFile = File(workspace, "hermes-tool-access-create.txt").apply { delete() }
        val deleteFile = File(workspace, "hermes-tool-access-delete.txt").apply { delete() }

        val port = freePort()
        val server = ScriptedChatServer(port)
        server.start(30_000, false)
        try {
            val result = NativeToolCallingChatClient(app).send(
                baseUrl = "http://127.0.0.1:$port",
                modelName = "scripted-tool-model",
                sessionId = "native-tool-access-smoke",
                userText = "Create a file, delete a file, and inspect Android phone status.",
            )

            assertEquals(3, result.executedToolCalls)
            assertTrue(result.content, result.content.contains("native tool access ok"))
            assertTrue("Expected ${createFile.absolutePath}", createFile.isFile)
            assertEquals("native-create-ok", createFile.readText())
            assertFalse("Expected ${deleteFile.absolutePath} to be deleted", deleteFile.exists())

            val followUpPayload = server.requests.last()
            val toolMessages = followUpPayload.getJSONArray("messages")
            assertTrue(toolMessages.toString(), toolMessages.toString().contains("native-create-ok"))
            assertTrue(toolMessages.toString(), toolMessages.toString().contains("deleted-ok"))
            assertTrue(toolMessages.toString(), toolMessages.toString().contains("available_system_actions"))
            assertTrue(toolMessages.toString(), toolMessages.toString().contains("active_network_label"))
        } finally {
            server.stop()
        }
    }

    @Test
    fun cronLifecycleAndBackgroundRuntimeActionsWorkInsideAndroidRuntime() {
        HermesLinuxSubsystemBridge.ensureInstalled(app)
        HermesRuntimeManager.ensurePythonStarted(app)
        Python.getInstance()
            .getModule("hermes_android.runtime_env")
            .callAttr("prepare_runtime_env", app.filesDir.absolutePath)

        val cronModule = Python.getInstance().getModule("tools.cronjob_tools")
        var jobId: String? = null
        try {
            val created = cron(
                cronModule,
                "create",
                null,
                "Android native cron lifecycle smoke prompt",
                "*/15 * * * *",
                "android-native-cron-smoke",
                1,
                "local",
            )
            assertTrue(created.toString(), created.getBoolean("success"))
            jobId = created.getString("job_id")

            val listed = cron(cronModule, "list")
            assertTrue(listed.toString(), listed.toString().contains(jobId!!))

            val paused = cron(cronModule, "pause", jobId!!)
            assertTrue(paused.toString(), paused.getBoolean("success"))

            val resumed = cron(cronModule, "resume", jobId!!)
            assertTrue(resumed.toString(), resumed.getBoolean("success"))

            val triggered = cron(cronModule, "trigger", jobId!!)
            assertTrue(triggered.toString(), triggered.getBoolean("success"))
        } finally {
            jobId?.let {
                val removed = cron(cronModule, "remove", it)
                assertTrue(removed.toString(), removed.getBoolean("success"))
            }
        }

        val start = JSONObject(HermesSystemControlBridge.performActionJson("start_background_runtime"))
        assertTrue(start.toString(), start.getBoolean("success"))
        val startedStatus = JSONObject(HermesSystemControlBridge.statusJson())
        assertTrue(startedStatus.toString(), startedStatus.getBoolean("background_persistence_enabled"))

        val stop = JSONObject(HermesSystemControlBridge.performActionJson("stop_background_runtime"))
        assertTrue(stop.toString(), stop.getBoolean("success"))
        val stoppedStatus = JSONObject(HermesSystemControlBridge.statusJson())
        assertFalse(stoppedStatus.toString(), stoppedStatus.getBoolean("background_persistence_enabled"))
        assertTrue(stoppedStatus.toString(), stoppedStatus.getJSONArray("available_system_actions").length() > 0)
    }

    private fun cron(module: PyObject, action: String, vararg args: Any?): JSONObject {
        return JSONObject(module.callAttr("cronjob", action, *args).toString())
    }

    private fun freePort(): Int {
        return ServerSocket(0).use { it.localPort }
    }

    private class ScriptedChatServer(port: Int) : NanoHTTPD("127.0.0.1", port) {
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
            val createCommand = "printf native-create-ok > \"\$HOME/hermes-tool-access-create.txt\" && " +
                "cat \"\$HOME/hermes-tool-access-create.txt\""
            val deleteCommand = "printf delete-me > \"\$HOME/hermes-tool-access-delete.txt\" && " +
                "rm \"\$HOME/hermes-tool-access-delete.txt\" && " +
                "test ! -e \"\$HOME/hermes-tool-access-delete.txt\" && printf deleted-ok"
            val message = JSONObject()
                .put("role", "assistant")
                .put("content", JSONObject.NULL)
                .put(
                    "tool_calls",
                    JSONArray()
                        .put(toolCall("call_create", "terminal_tool", JSONObject().put("command", createCommand)))
                        .put(toolCall("call_delete", "terminal_tool", JSONObject().put("command", deleteCommand)))
                        .put(toolCall("call_status", "android_system_tool", JSONObject().put("action", "status"))),
                )
            return completionPayload(message, "tool_calls")
        }

        private fun finalPayload(): JSONObject {
            return completionPayload(
                JSONObject()
                    .put("role", "assistant")
                    .put("content", "native tool access ok"),
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
                .put("id", "chatcmpl-native-tool-access")
                .put("object", "chat.completion")
                .put("created", System.currentTimeMillis() / 1000)
                .put("model", "scripted-tool-model")
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
