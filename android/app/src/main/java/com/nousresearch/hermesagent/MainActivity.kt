package com.nousresearch.hermesagent

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import com.nousresearch.hermesagent.auth.AuthRuntimeApplier
import com.nousresearch.hermesagent.data.AuthSessionStore
import com.nousresearch.hermesagent.device.DeviceStateWriter
import com.nousresearch.hermesagent.device.HermesLauncherShortcutBridge
import com.nousresearch.hermesagent.ui.boot.BootScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
        )
        super.onCreate(savedInstanceState)
        handleAuthCallback(intent)
        handleShortcutIntent(intent)
        DeviceStateWriter.write(applicationContext)
        setContent {
            BootScreen()
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleAuthCallback(intent)
        handleShortcutIntent(intent)
        DeviceStateWriter.write(applicationContext)
    }

    private fun handleAuthCallback(intent: Intent?) {
        val session = AuthSessionStore(applicationContext).consumeAuthCallback(intent?.data ?: return) ?: return
        lifecycleScope.launch(Dispatchers.IO) {
            AuthRuntimeApplier.apply(applicationContext, session)
            DeviceStateWriter.write(applicationContext)
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
