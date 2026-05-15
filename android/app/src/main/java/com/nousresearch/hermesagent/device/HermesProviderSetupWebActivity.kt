package com.nousresearch.hermesagent.device

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.text.TextUtils
import android.view.View
import android.view.ViewGroup
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.data.AuthSessionStore
import com.nousresearch.hermesagent.data.HermesNetworkPolicy

@Suppress("DEPRECATION", "OVERRIDE_DEPRECATION")
class HermesProviderSetupWebActivity : Activity() {
    private var webView: WebView? = null
    private lateinit var setupUri: Uri
    private lateinit var titleText: TextView
    private lateinit var progressBar: ProgressBar
    private var fallbackShown = false
    private var setupPageTitle = "Provider setup"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val requestedUrl = intent.getStringExtra(EXTRA_URL).orEmpty()
        val requestedTitle = intent.getStringExtra(EXTRA_TITLE).orEmpty().ifBlank { "Provider setup" }
        setupPageTitle = requestedTitle
        setupUri = Uri.parse(requestedUrl)
        if (!canOpen(setupUri)) {
            showFallback(requestedTitle, requestedUrl, "Provider setup URL must start with https:// or http://")
            return
        }
        if (HermesNetworkPolicy.isExternalNetworkBlocked(this, requestedUrl)) {
            showFallback(requestedTitle, requestedUrl, HermesNetworkPolicy.offlineBlockedMessage("provider setup page"))
            return
        }

        buildViewer(requestedTitle, setupUri.toString())
    }

    override fun onBackPressed() {
        val currentWebView = webView
        if (currentWebView != null && currentWebView.canGoBack()) {
            currentWebView.goBack()
            return
        }
        super.onBackPressed()
    }

    override fun onDestroy() {
        releaseWebView()
        super.onDestroy()
    }

    private fun buildViewer(pageTitle: String, url: String) {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(getColor(R.color.hermes_background))
            setPadding(0, statusBarInsetPx(), 0, 0)
        }

        titleText = TextView(this).apply {
            text = pageTitle
            textSize = 18f
            setTextColor(getColor(R.color.hermes_on_background))
            setSingleLine(true)
            ellipsize = TextUtils.TruncateAt.END
            setPadding(20, 16, 20, 4)
        }
        root.addView(titleText, fullWidthWrapParams())

        val toolbar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(12, 4, 12, 12)
        }
        toolbar.addView(toolbarButton("Back") { webView?.takeIf { it.canGoBack() }?.goBack() ?: finish() })
        toolbar.addView(toolbarButton("Browser") { openExternal(currentUrl()) })
        toolbar.addView(toolbarButton("Copy") { copyToClipboard(currentUrl()) })
        toolbar.addView(toolbarButton("Close") { finish() })
        root.addView(toolbar, fullWidthWrapParams())

        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            isIndeterminate = true
        }
        root.addView(progressBar, fullWidthWrapParams())

        val currentWebView = runCatching { WebView(this) }.getOrElse { error ->
            showFallback(pageTitle, url, "Android WebView could not start (${error::class.java.simpleName}).")
            return
        }
        webView = currentWebView
        configureWebView(currentWebView)
        root.addView(
            currentWebView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f,
            ),
        )

        setContentView(root)
        currentWebView.loadUrl(url)
    }

    private fun configureWebView(view: WebView) {
        view.setBackgroundColor(Color.WHITE)
        view.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            cacheMode = WebSettings.LOAD_DEFAULT
            loadsImagesAutomatically = true
            mixedContentMode = WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE
            setSupportMultipleWindows(false)
        }
        view.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                progressBar.isIndeterminate = false
                progressBar.progress = newProgress.coerceIn(0, 100)
                progressBar.visibility = if (newProgress >= 100) View.GONE else View.VISIBLE
            }

            override fun onReceivedTitle(view: WebView?, title: String?) {
                val resolvedTitle = title.orEmpty().ifBlank { intent.getStringExtra(EXTRA_TITLE).orEmpty() }
                if (resolvedTitle.isNotBlank()) {
                    titleText.text = resolvedTitle
                }
            }
        }
        view.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                return shouldOpenOutside(request.url)
            }

            @Suppress("DEPRECATION")
            override fun shouldOverrideUrlLoading(view: WebView, url: String): Boolean {
                return shouldOpenOutside(Uri.parse(url))
            }

            override fun onReceivedError(
                view: WebView,
                request: WebResourceRequest,
                error: WebResourceError,
            ) {
                if (request.isForMainFrame) {
                    showLoadFailureFallback(
                        request.url?.toString().orEmpty().ifBlank { currentUrl() },
                        "Setup page failed to load in Android WebView (${error.description}).",
                    )
                }
            }

            override fun onReceivedHttpError(
                view: WebView,
                request: WebResourceRequest,
                errorResponse: WebResourceResponse,
            ) {
                if (request.isForMainFrame && errorResponse.statusCode >= 400) {
                    showLoadFailureFallback(
                        request.url?.toString().orEmpty().ifBlank { currentUrl() },
                        "Setup page returned HTTP ${errorResponse.statusCode} in Android WebView.",
                    )
                }
            }
        }
    }

    private fun shouldOpenOutside(uri: Uri): Boolean {
        if (canOpen(uri)) {
            return false
        }
        if (!openHermesAuthCallback(uri)) {
            openExternal(uri.toString())
        }
        return true
    }

    private fun showFallback(pageTitle: String, url: String, message: String) {
        releaseWebView()
        if (::progressBar.isInitialized) {
            progressBar.visibility = View.GONE
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(28, 28 + statusBarInsetPx(), 28, 28)
            setBackgroundColor(getColor(R.color.hermes_background))
        }
        root.addView(TextView(this).apply {
            text = pageTitle
            textSize = 22f
            setTextColor(getColor(R.color.hermes_on_background))
        }, fullWidthWrapParams())
        root.addView(TextView(this).apply {
            text = "$message\n\n$url"
            textSize = 16f
            setTextColor(getColor(R.color.hermes_on_surface))
            setPadding(0, 20, 0, 20)
        }, fullWidthWrapParams())
        root.addView(toolbarButton("Open in browser") { openExternal(url) }, fullWidthWrapParams())
        root.addView(toolbarButton("Copy URL") { copyToClipboard(url) }, fullWidthWrapParams())
        root.addView(toolbarButton("Close") { finish() }, fullWidthWrapParams())
        setContentView(root)
        if (url.isNotBlank()) {
            copyToClipboard(url, showToast = false)
        }
    }

    private fun showLoadFailureFallback(url: String, message: String) {
        if (fallbackShown) {
            return
        }
        fallbackShown = true
        val targetUrl = url.ifBlank { setupUri.toString() }
        copyToClipboard(targetUrl, showToast = false)
        Toast.makeText(
            this,
            "Setup page failed to load; URL copied.",
            Toast.LENGTH_LONG,
        ).show()
        showFallback(setupPageTitle, targetUrl, message)
    }

    private fun toolbarButton(label: String, onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            setTextColor(getColor(R.color.hermes_on_primary))
            setBackgroundColor(getColor(R.color.hermes_primary))
            setOnClickListener { onClick() }
        }
    }

    private fun fullWidthWrapParams(): LinearLayout.LayoutParams {
        return LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        )
    }

    private fun statusBarInsetPx(): Int {
        val resourceId = resources.getIdentifier("status_bar_height", "dimen", "android")
        return if (resourceId > 0) {
            resources.getDimensionPixelSize(resourceId)
        } else {
            0
        }
    }

    private fun currentUrl(): String {
        return webView?.url.orEmpty().ifBlank { setupUri.toString() }
    }

    private fun releaseWebView() {
        webView?.let { existing ->
            runCatching { existing.stopLoading() }
            (existing.parent as? ViewGroup)?.removeView(existing)
            runCatching { existing.destroy() }
        }
        webView = null
    }

    private fun copyToClipboard(url: String, showToast: Boolean = true) {
        val target = url.trim()
        if (target.isBlank()) {
            return
        }
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
        clipboard?.setPrimaryClip(ClipData.newPlainText("Hermes provider setup URL", target))
        if (showToast) {
            Toast.makeText(this, "Provider setup URL copied.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun openExternal(url: String) {
        val targetUri = Uri.parse(url.trim())
        if (!canOpen(targetUri)) {
            if (!openHermesAuthCallback(targetUri)) {
                copyToClipboard(url)
            }
            return
        }
        val result = HermesExternalBrowserLauncher.open(
            context = this,
            uri = targetUri,
            title = "Open provider setup page",
            forceChooser = true,
        )
        if (!result.success) {
            copyToClipboard(url)
            Toast.makeText(this, "No external browser opened; URL copied.", Toast.LENGTH_LONG).show()
        }
    }

    private fun openHermesAuthCallback(uri: Uri): Boolean {
        if (!AuthSessionStore.isAuthCallback(uri)) {
            return false
        }
        return runCatching {
            startActivity(
                Intent(Intent.ACTION_VIEW, uri).apply {
                    addCategory(Intent.CATEGORY_BROWSABLE)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                },
            )
            finish()
        }.isSuccess
    }

    companion object {
        internal const val EXTRA_URL = "com.nousresearch.hermesagent.PROVIDER_SETUP_URL"
        internal const val EXTRA_TITLE = "com.nousresearch.hermesagent.PROVIDER_SETUP_TITLE"

        fun createIntent(context: Context, uri: Uri, title: String): Intent {
            return Intent(context, HermesProviderSetupWebActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                putExtra(EXTRA_URL, uri.toString())
                putExtra(EXTRA_TITLE, title.ifBlank { "Provider setup" })
            }
        }

        fun open(context: Context, uri: Uri, title: String): BrowserLaunchResult {
            if (!canOpen(uri)) {
                return BrowserLaunchResult(success = false, errorName = "UnsupportedScheme")
            }
            val external = HermesExternalBrowserLauncher.open(
                context = context,
                uri = uri,
                title = title,
                forceChooser = true,
            )
            if (external.success) {
                return external
            }
            val appContext = context.applicationContext
            return runCatching {
                appContext.startActivity(createIntent(appContext, uri, title))
                BrowserLaunchResult(success = true)
            }.getOrElse { error ->
                BrowserLaunchResult(success = false, errorName = error::class.java.simpleName)
            }
        }

        fun openInApp(context: Context, uri: Uri, title: String): BrowserLaunchResult {
            if (!canOpen(uri)) {
                return BrowserLaunchResult(success = false, errorName = "UnsupportedScheme")
            }
            val appContext = context.applicationContext
            return runCatching {
                appContext.startActivity(createIntent(appContext, uri, title))
                BrowserLaunchResult(success = true)
            }.getOrElse { error ->
                BrowserLaunchResult(success = false, errorName = error::class.java.simpleName)
            }
        }

        fun canOpen(uri: Uri): Boolean {
            return uri.scheme?.lowercase() in SUPPORTED_URI_SCHEMES && !uri.host.isNullOrBlank()
        }

        private val SUPPORTED_URI_SCHEMES = setOf("http", "https")
    }
}
