package com.nousresearch.hermesagent

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.nousresearch.hermesagent.device.HermesPrivilegedAccessBridge
import com.nousresearch.hermesagent.device.HermesSystemControlBridge
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class PrivilegedAccessStatusInstrumentedTest {
    private val app: Application
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun systemStatusExposesShizukuPrivilegedAccessSetupState() {
        val status = JSONObject(HermesSystemControlBridge.statusJson())
        assertTrue(status.toString(), status.has("privileged_access"))

        val privileged = status.getJSONObject("privileged_access")
        assertTrue(privileged.toString(), privileged.has("shizuku_installed"))
        assertTrue(privileged.toString(), privileged.has("sui_installed"))
        assertTrue(privileged.toString(), privileged.has("shizuku_binder_alive"))
        assertTrue(privileged.toString(), privileged.has("shizuku_permission_granted"))
        assertTrue(privileged.toString(), privileged.has("shizuku_privilege_label"))
        assertTrue(privileged.toString(), privileged.has("adb_start_command"))
        assertTrue(privileged.toString(), privileged.getJSONArray("available_privileged_actions").length() > 0)
        assertTrue(
            privileged.toString(),
            privileged.getJSONArray("available_privileged_actions").toString().contains("run_privileged_shell"),
        )
    }

    @Test
    fun unsupportedPrivilegedActionReportsFailureWithoutCrash() {
        val result = HermesPrivilegedAccessBridge.performAction(app, "unsupported_privileged_probe")
        assertTrue(result.message, !result.success)
        assertTrue(result.message, result.message.contains("Unsupported privileged Android action"))
    }

    @Test
    fun privilegedShellReportsPermissionStateWithoutCrash() {
        val result = JSONObject(HermesPrivilegedAccessBridge.runShellCommandJson(app, "id", 1))
        assertTrue(result.toString(), result.has("success"))
        assertTrue(result.toString(), result.has("exit_code"))
        if (!result.getBoolean("success")) {
            assertTrue(result.toString(), result.has("error"))
            assertTrue(result.toString(), result.has("shizuku_privilege_label"))
        }
    }
}
