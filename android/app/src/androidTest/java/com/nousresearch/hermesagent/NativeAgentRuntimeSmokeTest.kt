package com.nousresearch.hermesagent

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.chaquo.python.Python
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.backend.PythonBootProbe
import com.nousresearch.hermesagent.device.HermesLinuxSubsystemBridge
import com.nousresearch.hermesagent.device.HermesSystemControlBridge
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class NativeAgentRuntimeSmokeTest {
    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @After
    fun tearDown() {
        HermesRuntimeManager.stop()
    }

    @Test
    fun linuxCommandSuiteSupportsProcessExecutionAndFilesystemWrites() {
        val state = HermesLinuxSubsystemBridge.ensureInstalled(context)
        val shellPath = state.optString("shell_path", state.getString("bash_path"))
        val prefixPath = state.getString("prefix_path")
        val binPath = state.getString("bin_path")
        val homePath = state.getString("home_path")
        val tmpPath = state.getString("tmp_path")
        val executionMode = state.getString("execution_mode")

        assertTrue(
            "execution_mode=$executionMode",
            executionMode == "embedded_termux" || executionMode == "android_system_shell",
        )
        if (executionMode == "embedded_termux") {
            assertTrue(
                "shell must be APK-packaged embedded bash",
                shellPath.endsWith("/libhermes_android_bash.so") || shellPath.endsWith("/bin/bash"),
            )
        } else {
            assertEquals("/system/bin/sh", shellPath)
        }
        assertTrue("shell must exist", File(shellPath).isFile)
        assertTrue("shell must be executable", File(shellPath).canExecute())
        assertTrue("prefix must exist", File(prefixPath).isDirectory)
        assertTrue("home must exist", File(homePath).isDirectory)

        val probeFile = File(homePath, "hermes-native-fs-smoke.txt")
        val command = "printf hermes-native-ok > ${shellQuote(probeFile.absolutePath)} && " +
            "cat ${shellQuote(probeFile.absolutePath)} && " +
            "printf '\\n' && command -v ls"
        val process = ProcessBuilder(shellPath, "-c", command)
            .directory(File(homePath))
            .redirectErrorStream(true)
            .apply {
                environment().putAll(HermesLinuxSubsystemBridge.buildRunEnvironment(state))
            }
            .start()

        assertTrue("command suite process timed out", process.waitFor(20, TimeUnit.SECONDS))
        val output = process.inputStream.bufferedReader().readText()
        assertEquals("command suite process exit: $output", 0, process.exitValue())
        assertTrue(output, output.contains("hermes-native-ok"))
        assertTrue(output, output.contains("/ls"))
        assertEquals("hermes-native-ok", probeFile.readText())
    }

    @Test
    fun androidDeviceBridgeExposesStatusAndSafeActions() {
        val status = JSONObject(HermesSystemControlBridge.statusJson())
        val actions = status.optJSONArray("available_system_actions") ?: JSONArray()
        assertTrue(status.has("active_network_label"))
        assertTrue(status.has("usb_host_supported"))
        assertTrue(actions.toString(), actions.toString().contains("start_background_runtime"))

        val result = JSONObject(HermesSystemControlBridge.performActionJson("unsupported_smoke_action"))
        assertFalse(result.toString(), result.getBoolean("success"))
        assertEquals("unsupported_smoke_action", result.getString("action"))
    }

    @Test
    fun embeddedHermesPythonRuntimeStartsLocalServer() {
        val bootProbe = JSONObject(PythonBootProbe.readProbe(context))
        assertEquals(bootProbe.toString(), "ok", bootProbe.optString("status"))

        val linuxState = HermesLinuxSubsystemBridge.ensureInstalled(context)
        val homePath = linuxState.getString("home_path")
        val state = HermesRuntimeManager.ensureStarted(context)
        assertTrue(state.error.orEmpty(), state.started)
        assertTrue("baseUrl=${state.baseUrl}", state.baseUrl.orEmpty().startsWith("http://127.0.0.1:"))
        assertTrue("hermesHome=${state.hermesHome}", state.hermesHome.orEmpty().isNotBlank())

        val terminalResult = JSONObject(
            Python.getInstance()
                .getModule("tools.terminal_tool")
                .callAttr(
                    "terminal_tool",
                    "printf tool-ok > \"\$HOME/hermes-tool-smoke.txt\" && cat \"\$HOME/hermes-tool-smoke.txt\" && printf '\\n' && pwd",
                    false,
                    20,
                    "android-native-smoke",
                    true,
                )
                .toString()
        )
        assertEquals(terminalResult.toString(), 0, terminalResult.optInt("exit_code", -1))
        assertTrue(terminalResult.toString(), terminalResult.optString("output").contains("tool-ok"))
        assertEquals("tool-ok", File(homePath, "hermes-tool-smoke.txt").readText())
    }

    private fun shellQuote(value: String): String {
        return "'" + value.replace("'", "'\\''") + "'"
    }
}
