package com.mobilefork.hermesagent

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import com.mobilefork.hermesagent.auth.AuthRuntimeApplier
import com.mobilefork.hermesagent.auth.OpenRouterOAuthClient
import com.mobilefork.hermesagent.data.AuthSessionStore
import com.mobilefork.hermesagent.device.HermesCrashLogStore
import com.mobilefork.hermesagent.device.HermesLauncherShortcutBridge
import com.mobilefork.hermesagent.ui.boot.BootScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
        )
        super.onCreate(savedInstanceState)
        HermesCrashLogStore.install(applicationContext)
        handleAuthCallback(intent)
        handleShortcutIntent(intent)
        setContent {
            BootScreen()
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleAuthCallback(intent)
        handleShortcutIntent(intent)
    }

    private fun handleAuthCallback(intent: Intent?) {
        val callbackUri = intent?.data ?: return
        val store = AuthSessionStore(applicationContext)
        val pending = store.loadPendingRequest()
        if (OpenRouterOAuthClient.isOpenRouterCallback(callbackUri, pending)) {
            lifecycleScope.launch(Dispatchers.IO) {
                val session = OpenRouterOAuthClient.exchangeCallbackForSession(callbackUri, requireNotNull(pending))
                store.clearPendingRequest()
                store.saveSession(session)
                if (session.signedIn) {
                    AuthRuntimeApplier.apply(applicationContext, session)
                }
            }
            return
        }
        val session = store.consumeAuthCallback(callbackUri) ?: return
        lifecycleScope.launch(Dispatchers.IO) {
            AuthRuntimeApplier.apply(applicationContext, session)
        }
    }

    private fun handleShortcutIntent(intent: Intent?) {
        if (!HermesLauncherShortcutBridge.isShortcutIntent(intent)) {
            return
        }
        lifecycleScope.launch(Dispatchers.IO) {
            HermesLauncherShortcutBridge.handleShortcutIntentJson(applicationContext, intent)
        }
    }
}
