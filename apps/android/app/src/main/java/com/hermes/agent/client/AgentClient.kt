package com.hermes.agent.client

enum class ConnectionState {
    LOCAL,
    REMOTE,
    OFFLINE,
    CONNECTING,
    ERROR
}

class AgentClient {
    var state: ConnectionState = ConnectionState.LOCAL

    fun connect(targetUrl: String) {
        state = ConnectionState.CONNECTING
    }

    fun disconnect() {
        state = ConnectionState.OFFLINE
    }
}
