package com.nousresearch.hermesagent.device

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Browser

data class BrowserLaunchResult(
    val success: Boolean,
    val errorName: String = "",
)

object HermesExternalBrowserLauncher {
    fun createBrowserIntent(context: Context, uri: Uri): Intent {
        return Intent(Intent.ACTION_VIEW, uri).apply {
            addCategory(Intent.CATEGORY_BROWSABLE)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            putExtra(Browser.EXTRA_APPLICATION_ID, context.packageName)
        }
    }

    fun createChooserIntent(context: Context, uri: Uri, title: String): Intent {
        return Intent.createChooser(
            createBrowserIntent(context, uri),
            title.ifBlank { "Open link" },
        ).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
    }

    fun open(context: Context, uri: Uri, title: String): BrowserLaunchResult {
        val scheme = uri.scheme?.lowercase().orEmpty()
        if (scheme !in BROWSABLE_URI_SCHEMES) {
            return BrowserLaunchResult(
                success = false,
                errorName = "UnsupportedScheme",
            )
        }
        val appContext = context.applicationContext
        return runCatching {
            appContext.startActivity(createChooserIntent(appContext, uri, title))
            BrowserLaunchResult(success = true)
        }.getOrElse { error ->
            BrowserLaunchResult(
                success = false,
                errorName = error::class.java.simpleName,
            )
        }
    }

    private val BROWSABLE_URI_SCHEMES = setOf("http", "https")
}
