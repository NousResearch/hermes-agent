package com.mobilefork.hermesagent

import android.app.Application
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.mobilefork.hermesagent.device.HermesFloatingButtonService
import com.mobilefork.hermesagent.device.HermesSystemControlBridge
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.FileInputStream

@RunWith(AndroidJUnit4::class)
class HermesFloatingButtonInstrumentedTest {
    private val app: Application
        get() = ApplicationProvider.getApplicationContext()

    @After
    fun tearDown() {
        HermesSystemControlBridge.performAction(app, "stop_floating_button")
        shell("appops set ${app.packageName} SYSTEM_ALERT_WINDOW default")
    }

    @Test
    fun floatingButtonStartsFromAppBridgeAndSurvivesHome() {
        shell("appops set ${app.packageName} SYSTEM_ALERT_WINDOW allow")

        ActivityScenario.launch(MainActivity::class.java).use {
            val result = HermesSystemControlBridge.performAction(app, "start_floating_button")
            assertTrue(result.message, result.success)
            val becameVisible = waitForFloatingButton()
            assertTrue(floatingButtonDebugStatus(), becameVisible)

            shell("input keyevent KEYCODE_HOME")
            Thread.sleep(750)

            assertTrue("Floating button service stopped after Home", HermesFloatingButtonService.isRunning())
            assertTrue(floatingButtonDebugStatus(), HermesFloatingButtonService.isButtonVisible())
        }
    }

    private fun waitForFloatingButton(): Boolean {
        repeat(20) {
            if (HermesFloatingButtonService.isRunning() && HermesFloatingButtonService.isButtonVisible()) {
                return true
            }
            Thread.sleep(250)
        }
        return false
    }

    private fun floatingButtonDebugStatus(): String {
        val status = HermesSystemControlBridge.readStatus(app)
        return "running=${status.floatingButtonRunning}, visible=${status.floatingButtonVisible}, error=${status.floatingButtonError}"
    }

    private fun shell(command: String): String {
        val descriptor = InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand(command)
        return try {
            FileInputStream(descriptor.fileDescriptor).bufferedReader().use { it.readText() }
        } finally {
            descriptor.close()
        }
    }
}
