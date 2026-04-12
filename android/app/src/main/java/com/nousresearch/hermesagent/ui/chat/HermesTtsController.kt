package com.nousresearch.hermesagent.ui.chat

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

class HermesTtsController(context: Context) : TextToSpeech.OnInitListener {
    private val textToSpeech = TextToSpeech(context.applicationContext, this)
    private var ready = false

    override fun onInit(status: Int) {
        ready = status == TextToSpeech.SUCCESS
        if (ready) {
            textToSpeech.language = Locale.getDefault()
        }
    }

    fun speak(text: String): Boolean {
        if (!ready || text.isBlank()) {
            return false
        }
        textToSpeech.stop()
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, "hermes-${System.currentTimeMillis()}")
        return true
    }

    fun shutdown() {
        textToSpeech.stop()
        textToSpeech.shutdown()
    }
}
