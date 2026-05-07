package com.nousresearch.hermesagent

import android.app.Application
import com.nousresearch.hermesagent.backend.HermesRuntimeService
import com.nousresearch.hermesagent.data.DeviceCapabilityStore
import com.nousresearch.hermesagent.device.DeviceStateWriter
import com.nousresearch.hermesagent.device.HermesLogcatWatcherService
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
            DeviceStateWriter.write(this@HermesApplication)
            if (DeviceCapabilityStore(this@HermesApplication).load().backgroundPersistenceEnabled) {
                delay(BACKGROUND_RUNTIME_STARTUP_DELAY_MS)
                HermesRuntimeService.start(this@HermesApplication)
            }
            HermesLogcatWatcherService.startIfDesired(this@HermesApplication)
        }
    }

    companion object {
        private const val BACKGROUND_RUNTIME_STARTUP_DELAY_MS = 1500L

        lateinit var instance: HermesApplication
            private set
    }
}
