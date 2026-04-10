package com.nousresearch.hermesagent

import android.app.Application
import com.nousresearch.hermesagent.backend.HermesRuntimeManager

class HermesApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        HermesRuntimeManager.ensureStarted(this)
    }
}
