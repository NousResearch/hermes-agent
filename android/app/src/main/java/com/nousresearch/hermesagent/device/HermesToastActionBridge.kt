package com.nousresearch.hermesagent.device

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import org.json.JSONObject
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

object HermesToastActionBridge {
    fun showToastJson(context: Context, payload: JSONObject): JSONObject {
        val text = payload.optString("text")
        if (text.indexOf('\u0000') >= 0) {
            return errorJson("Toast text must not contain NUL bytes")
        }
        if (text.isBlank()) {
            return errorJson("Toast text must not be blank")
        }
        val long = payload.optBoolean("long", false)
        return runOnMainThread {
            Toast.makeText(
                context.applicationContext,
                text,
                if (long) Toast.LENGTH_LONG else Toast.LENGTH_SHORT,
            ).show()
            JSONObject()
                .put("success", true)
                .put("exit_code", 0)
                .put("action", "show_toast")
                .put("message", "Toast shown")
                .put("text_length", text.length)
                .put("long", long)
        }
    }

    private fun runOnMainThread(block: () -> JSONObject): JSONObject {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            return runCatching(block).getOrElse { error -> errorJson(error.message ?: error.javaClass.simpleName) }
        }
        val result = AtomicReference<JSONObject>()
        val latch = CountDownLatch(1)
        mainHandler.post {
            result.set(runCatching(block).getOrElse { error -> errorJson(error.message ?: error.javaClass.simpleName) })
            latch.countDown()
        }
        if (!latch.await(2, TimeUnit.SECONDS)) {
            return errorJson("Timed out while showing toast")
        }
        return result.get() ?: errorJson("Toast action did not return a result")
    }

    private fun errorJson(message: String): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 1)
            .put("action", "show_toast")
            .put("error", message)
    }

    private val mainHandler = Handler(Looper.getMainLooper())
}
