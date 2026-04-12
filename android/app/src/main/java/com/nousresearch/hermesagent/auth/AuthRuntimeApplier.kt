package com.nousresearch.hermesagent.auth

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.data.AppSettings
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.data.AuthScope
import com.nousresearch.hermesagent.data.AuthSession
import com.nousresearch.hermesagent.data.ProviderPresets

object AuthRuntimeApplier {
    fun apply(context: Context, session: AuthSession) {
        if (!session.signedIn || session.scope != AuthScope.RuntimeProvider || session.runtimeProvider.isBlank()) {
            return
        }

        val appContext = context.applicationContext
        val settingsStore = AppSettingsStore(appContext)
        val existingSettings = settingsStore.load()
        val preset = ProviderPresets.find(session.runtimeProvider)
        val resolvedBaseUrl = session.baseUrl.ifBlank { preset?.baseUrl.orEmpty() }
        val resolvedModel = session.model.ifBlank { preset?.modelHint.orEmpty() }

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(appContext))
        }
        val python = Python.getInstance()
        python.getModule("hermes_android.auth_bridge").callAttr(
            "write_provider_auth_bundle",
            session.runtimeProvider,
            session.apiKey,
            session.accessToken,
            session.sessionToken,
        )
        python.getModule("hermes_android.config_bridge").callAttr(
            "write_runtime_config",
            session.runtimeProvider,
            resolvedModel,
            resolvedBaseUrl,
        )

        settingsStore.save(
            AppSettings(
                provider = session.runtimeProvider,
                baseUrl = resolvedBaseUrl,
                model = resolvedModel,
                corr3xtBaseUrl = existingSettings.corr3xtBaseUrl,
                dataSaverMode = existingSettings.dataSaverMode,
            )
        )
        HermesRuntimeManager.stop()
        HermesRuntimeManager.ensureStarted(appContext)
    }
}
