package com.mobilefork.hermesagent

import android.app.Application
import com.mobilefork.hermesagent.backend.HermesRuntimeService
import com.mobilefork.hermesagent.data.DeviceCapabilityStore
import com.mobilefork.hermesagent.device.DeviceStateWriter
import com.mobilefork.hermesagent.device.HermesCalendarWatcherService
import com.mobilefork.hermesagent.device.HermesLocationWatcherService
import com.mobilefork.hermesagent.device.HermesLogcatWatcherService
import com.mobilefork.hermesagent.device.HermesSensorWatcherService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.MainScope

class HermesApplication : Application() {
    private val appScope = MainScope()

    override fun onCreate() {
        super.onCreate()
        instance = this

        appScope.launch(Dispatchers.IO) {
            delay(STARTUP_BACKGROUND_WORK_DELAY_MS)
            DeviceStateWriter.write(this@HermesApplication)
            if (DeviceCapabilityStore(this@HermesApplication).load().backgroundPersistenceEnabled) {
                delay(BACKGROUND_RUNTIME_STARTUP_DELAY_MS)
                HermesRuntimeService.start(this@HermesApplication)
            }
            HermesLogcatWatcherService.startIfDesired(this@HermesApplication)
            HermesSensorWatcherService.startIfDesired(this@HermesApplication)
            HermesCalendarWatcherService.startIfDesired(this@HermesApplication)
            HermesLocationWatcherService.startIfDesired(this@HermesApplication)
        }
    }

    companion object {
        private const val STARTUP_BACKGROUND_WORK_DELAY_MS = 5000L
        private const val BACKGROUND_RUNTIME_STARTUP_DELAY_MS = 1500L

        lateinit var instance: HermesApplication
            private set
    }
}
