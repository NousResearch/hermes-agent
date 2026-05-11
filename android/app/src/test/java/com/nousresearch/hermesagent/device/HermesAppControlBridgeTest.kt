package com.nousresearch.hermesagent.device

import android.content.Context
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class HermesAppControlBridgeTest {
    private val context: Context = RuntimeEnvironment.getApplication()

    @Test
    fun launchAppRequiresPackageNameOrAppName() {
        val result = HermesAppControlBridge.launchApp(context, packageName = "", appName = "")

        assertFalse(result.getBoolean("success"))
        assertEquals(64, result.getInt("exit_code"))
        assertEquals("launch_app", result.getString("action"))
        assertTrue(result.getString("error").contains("package_name or app_name"))
    }

    @Test
    fun launchAppRejectsNulInAppName() {
        val result = HermesAppControlBridge.launchApp(context, packageName = "", appName = "Chrome\u0000")

        assertFalse(result.getBoolean("success"))
        assertEquals(64, result.getInt("exit_code"))
        assertTrue(result.getString("error").contains("app_name must not contain NUL"))
    }

    @Test
    fun launchAppRejectsNulInPackageName() {
        val result = HermesAppControlBridge.launchApp(context, packageName = "com.example\u0000bad", appName = "")

        assertFalse(result.getBoolean("success"))
        assertEquals(64, result.getInt("exit_code"))
        assertTrue(result.getString("error").contains("package_name must not contain NUL"))
    }
}
