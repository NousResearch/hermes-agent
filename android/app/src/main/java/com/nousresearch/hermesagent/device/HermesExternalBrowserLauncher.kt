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
    fun createBrowserIntent(context: Context, uri: Uri, preferInstalledBrowser: Boolean = true): Intent {
        val appContext = context.applicationContext
        return Intent(Intent.ACTION_VIEW, uri).apply {
            addCategory(Intent.CATEGORY_BROWSABLE)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            putExtra(Browser.EXTRA_APPLICATION_ID, appContext.packageName)
            if (preferInstalledBrowser) {
                HermesIntentBridge.preferredBrowserPackage(appContext)?.let { packageName ->
                    setPackage(packageName)
                }
            }
        }
    }

    fun createChooserIntent(context: Context, uri: Uri, title: String): Intent {
        return Intent.createChooser(
            createBrowserIntent(context, uri, preferInstalledBrowser = false),
            title.ifBlank { "Open link" },
        ).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
    }

    fun open(
        context: Context,
        uri: Uri,
        title: String,
        forceChooser: Boolean = false,
    ): BrowserLaunchResult {
        val scheme = uri.scheme?.lowercase().orEmpty()
        if (scheme !in BROWSABLE_URI_SCHEMES) {
            return BrowserLaunchResult(
                success = false,
                errorName = "UnsupportedScheme",
            )
        }
        val appContext = context.applicationContext
        if (forceChooser) {
            return runCatching {
                appContext.startActivity(createChooserIntent(appContext, uri, title))
                BrowserLaunchResult(success = true)
            }.getOrElse { chooserError ->
                BrowserLaunchResult(success = false, errorName = chooserError::class.java.simpleName)
            }
        }
        val directIntent = createBrowserIntent(appContext, uri)
        runCatching {
            appContext.startActivity(directIntent)
            BrowserLaunchResult(success = true)
        }.onSuccess { result ->
            return result
        }

        return runCatching {
            appContext.startActivity(createChooserIntent(appContext, uri, title))
            BrowserLaunchResult(success = true)
        }.getOrElse { chooserError ->
            BrowserLaunchResult(success = false, errorName = chooserError::class.java.simpleName)
        }
    }

    private val BROWSABLE_URI_SCHEMES = setOf("http", "https")
}
