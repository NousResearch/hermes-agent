package com.nousresearch.hermesagent.backend

import android.content.Context
import com.chaquo.python.PyException
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

object PythonBootProbe {
    fun readProbe(context: Context): String {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context.applicationContext))
        }

        return try {
            Python.getInstance()
                .getModule("hermes_android.boot_probe")
                .callAttr("boot_probe")
                .toString()
        } catch (exc: PyException) {
            "{\"status\":\"error\",\"message\":${jsonString(exc.message ?: exc.toString())}}"
        } catch (exc: Throwable) {
            "{\"status\":\"error\",\"message\":${jsonString(exc.message ?: exc.toString())}}"
        }
    }

    private fun jsonString(value: String): String {
        return buildString {
            append('"')
            value.forEach { ch ->
                when (ch) {
                    '\\' -> append("\\\\")
                    '"' -> append("\\\"")
                    '\n' -> append("\\n")
                    '\r' -> append("\\r")
                    '\t' -> append("\\t")
                    else -> append(ch)
                }
            }
            append('"')
        }
    }
}
