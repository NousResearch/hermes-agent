package com.mobilefork.hermesagent.ui.chat

import android.content.Context
import com.mobilefork.hermesagent.device.HermesDeviceDiagnosticsBridge
import com.mobilefork.hermesagent.device.HermesHindsightMemoryBridge
import org.json.JSONObject

internal object NativeBridgeInvoker {
    fun performDiagnosticsAction(context: Context, action: String, arguments: JSONObject): String {
        return HermesDeviceDiagnosticsBridge.performActionJson(
            context = context.applicationContext,
            action = action,
            arguments = arguments,
        )
    }

    fun performMemoryAction(context: Context, action: String, arguments: JSONObject): String {
        return HermesHindsightMemoryBridge.performActionJson(
            context = context.applicationContext,
            rawAction = action,
            arguments = arguments,
        )
    }
}
