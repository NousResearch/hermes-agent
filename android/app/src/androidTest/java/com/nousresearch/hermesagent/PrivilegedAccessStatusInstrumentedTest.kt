package com.nousresearch.hermesagent

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.nousresearch.hermesagent.device.HermesPrivilegedAccessBridge
import com.nousresearch.hermesagent.device.HermesSystemControlBridge
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
        assertTrue(
            privileged.toString(),
            privileged.getJSONArray("available_privileged_actions").toString().contains("grant_runtime_permission"),
        )
        assertTrue(
            privileged.toString(),
            privileged.getJSONArray("available_privileged_actions").toString().contains("force_stop_app"),
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

    @Test
    fun structuredPrivilegedActionsValidateArgumentsBeforeShellExecution() {
        assertTrue(HermesPrivilegedAccessBridge.handlesStructuredAction("grant_runtime_permission"))
        assertTrue(HermesPrivilegedAccessBridge.handlesStructuredAction("pm grant"))
        assertFalse(HermesPrivilegedAccessBridge.handlesStructuredAction("unsupported_structured_action"))

        val missingPackage = JSONObject(
            HermesPrivilegedAccessBridge.performStructuredActionJson(app, "grant_runtime_permission", JSONObject())
        )
        assertFalse(missingPackage.toString(), missingPackage.getBoolean("success"))
        assertEquals("grant_runtime_permission", missingPackage.getString("action"))
        assertTrue(missingPackage.toString(), missingPackage.getString("error").contains("requires package_name"))

        val invalidPackage = JSONObject(
            HermesPrivilegedAccessBridge.performStructuredActionJson(
                app,
                "force_stop_app",
                JSONObject().put("package_name", "com.example;bad"),
            )
        )
        assertFalse(invalidPackage.toString(), invalidPackage.getBoolean("success"))
        assertTrue(invalidPackage.toString(), invalidPackage.getString("error").contains("valid Android package name"))

        val invalidPermission = JSONObject(
            HermesPrivilegedAccessBridge.performStructuredActionJson(
                app,
                "grant_runtime_permission",
                JSONObject()
                    .put("package_name", "com.example.valid")
                    .put("permission", "android.permission.POST_NOTIFICATIONS;bad"),
            )
        )
        assertFalse(invalidPermission.toString(), invalidPermission.getBoolean("success"))
        assertTrue(invalidPermission.toString(), invalidPermission.getString("error").contains("valid Android permission name"))

        val selfStop = JSONObject(
            HermesPrivilegedAccessBridge.performStructuredActionJson(
                app,
                "force_stop_app",
                JSONObject().put("package_name", app.packageName),
            )
        )
        assertFalse(selfStop.toString(), selfStop.getBoolean("success"))
        assertTrue(selfStop.toString(), selfStop.getString("error").contains("will not disable"))
        assertTrue(selfStop.toString(), selfStop.getString("error").contains("force-stop"))
        assertTrue(selfStop.toString(), selfStop.getString("error").contains("itself"))
    }

    @Test
    fun structuredPrivilegedPermissionActionReportsShizukuStateWithoutCrash() {
        val result = JSONObject(
            HermesPrivilegedAccessBridge.performStructuredActionJson(
                app,
                "grant_runtime_permission",
                JSONObject()
                    .put("package_name", app.packageName)
                    .put("permission", "android.permission.POST_NOTIFICATIONS")
                    .put("timeout_seconds", 1),
            )
        )
        assertTrue(result.toString(), result.has("success"))
        assertTrue(result.toString(), result.has("exit_code"))
        assertEquals("grant_runtime_permission", result.getString("action"))
        assertEquals(app.packageName, result.getString("package_name"))
        assertEquals("android.permission.POST_NOTIFICATIONS", result.getString("permission"))
        assertTrue(result.toString(), result.getString("adb_shell_command").contains("pm grant"))
        if (!result.getBoolean("success")) {
            assertTrue(result.toString(), result.has("error"))
        }
    }
}
