package com.nousresearch.hermesagent

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.nousresearch.hermesagent.auth.AuthRuntimeApplier
import com.nousresearch.hermesagent.data.AuthSessionStore
import com.nousresearch.hermesagent.ui.boot.BootScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        handleAuthCallback(intent)
        setContent {
            BootScreen()
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleAuthCallback(intent)
    }

    private fun handleAuthCallback(intent: Intent?) {
        val session = AuthSessionStore(applicationContext).consumeAuthCallback(intent?.data ?: return) ?: return
        AuthRuntimeApplier.apply(applicationContext, session)
    }
}
