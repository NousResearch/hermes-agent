package com.abdulraheemnohri.hermestensor.runtime

import android.app.Service
import android.content.Intent
import android.os.IBinder

class LiteRtLmService : Service() {
    private var state = RuntimeState()

    override fun onCreate() {
        super.onCreate()
        // TODO: initialize LiteRT-LM Android Engine here.
        // The selected delegate must be reported from the real runtime.
    }

    fun selectBackend(requested: Backend): Backend {
        val probe = object : AcceleratorProbe {
            override fun isSupported(backend: Backend): Boolean {
                // TODO: query LiteRT-LM/Android delegate availability.
                return backend == Backend.CPU
            }
        }
        val selected = BackendSelector(probe).select(requested)
        state = state.copy(backend = requested, actualBackend = selected)
        return selected
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
