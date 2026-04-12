package com.nousresearch.hermesagent

import android.app.Application
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.backend.HermesRuntimeService
import com.nousresearch.hermesagent.data.DeviceCapabilityStore
import com.nousresearch.hermesagent.device.DeviceStateWriter

class HermesApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        instance = this
        if (DeviceCapabilityStore(this).load().backgroundPersistenceEnabled) {
            HermesRuntimeService.start(this)
        } else {
            HermesRuntimeManager.ensureStarted(this)
        }
        DeviceStateWriter.write(this)
    }

    companion object {
        lateinit var instance: HermesApplication
            private set
    }
}
