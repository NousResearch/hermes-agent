package com.abdulraheemnohri.hermestensor.runtime

enum class Backend { AUTO, CPU, GPU, NPU }

data class RuntimeState(
    val backend: Backend = Backend.AUTO,
    val actualBackend: Backend? = null,
    val model: String? = null,
    val loaded: Boolean = false,
    val generating: Boolean = false,
    val thermalStatus: Int = 0
)

interface AcceleratorProbe {
    fun isSupported(backend: Backend): Boolean
}

class BackendSelector(private val probe: AcceleratorProbe) {
    fun select(requested: Backend): Backend {
        if (requested != Backend.AUTO) {
            require(probe.isSupported(requested)) { "Requested backend is unavailable: $requested" }
            return requested
        }
        return listOf(Backend.NPU, Backend.GPU, Backend.CPU)
            .first { probe.isSupported(it) }
    }
}
