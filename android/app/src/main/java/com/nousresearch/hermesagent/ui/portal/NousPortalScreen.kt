package com.nousresearch.hermesagent.ui.portal

import android.app.Application
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.view.ViewGroup
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.ui.shell.ShellActionItem
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject

private const val DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"
private const val PORTAL_EMBED_USER_AGENT = "Mozilla/5.0 (Linux; Android 15; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"

data class NousPortalUiState(
    val portalUrl: String = DEFAULT_NOUS_PORTAL_URL,
    val loggedIn: Boolean = false,
    val inferenceUrl: String = "",
    val status: String = "Loading Nous Portal…",
)

class NousPortalViewModel(application: Application) : AndroidViewModel(application) {
    private val _uiState = MutableStateFlow(NousPortalUiState())
    val uiState: StateFlow<NousPortalUiState> = _uiState.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = runCatching {
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(getApplication()))
                }
                val payload = Python.getInstance()
                    .getModule("hermes_android.nous_portal_bridge")
                    .callAttr("read_nous_portal_state_json")
                    .toString()
                val json = JSONObject(payload)
                NousPortalUiState(
                    portalUrl = json.optString("portal_url").ifBlank { DEFAULT_NOUS_PORTAL_URL },
                    loggedIn = json.optBoolean("logged_in", false),
                    inferenceUrl = json.optString("inference_url").orEmpty(),
                    status = if (json.optBoolean("logged_in", false)) {
                        "Signed in to Nous Portal"
                    } else {
                        "Loading the embedded portal preview"
                    },
                )
            }.getOrElse { error ->
                NousPortalUiState(
                    portalUrl = DEFAULT_NOUS_PORTAL_URL,
                    status = "Using default Nous Portal URL (${error.message ?: error.javaClass.simpleName})",
                )
            }
        }
    }
}

@Composable
fun NousPortalScreen(
    modifier: Modifier = Modifier,
    viewModel: NousPortalViewModel = viewModel(),
    extraBottomSpacing: Dp = 0.dp,
    onContextActionsChanged: (List<ShellActionItem>) -> Unit = {},
) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    var isLoading by remember { mutableStateOf(true) }
    var pageError by remember { mutableStateOf<String?>(null) }
    var webViewRef by remember { mutableStateOf<WebView?>(null) }
    var isFullscreen by rememberSaveable { mutableStateOf(false) }

    SideEffect {
        onContextActionsChanged(
            listOf(
                ShellActionItem(
                    label = "Refresh portal",
                    description = "Reload the embedded Nous Portal page.",
                    iconRes = R.drawable.ic_action_refresh,
                    onClick = {
                        isLoading = true
                        pageError = null
                        viewModel.refresh()
                        webViewRef?.reload()
                    },
                ),
                ShellActionItem(
                    label = if (isFullscreen) "Minimize portal" else "Full screen portal",
                    description = "Resize the embedded portal preview without leaving the app.",
                    iconRes = if (isFullscreen) R.drawable.ic_action_minimize else R.drawable.ic_action_fullscreen,
                    onClick = { isFullscreen = !isFullscreen },
                ),
                ShellActionItem(
                    label = "Open externally",
                    description = "Open the full portal in your browser if the embed is limited.",
                    iconRes = R.drawable.ic_action_external,
                    onClick = {
                        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(uiState.portalUrl))
                        context.startActivity(intent)
                    },
                ),
            )
        )
    }

    MaterialTheme {
        Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .widthIn(max = if (isFullscreen) 1200.dp else 920.dp)
                        .fillMaxSize()
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    if (!isFullscreen) {
                        PortalGuidanceCard(
                            status = uiState.status,
                            inferenceUrl = uiState.inferenceUrl,
                            pageError = pageError,
                        )
                    }
                    if (isLoading) {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    }
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth()
                            .padding(bottom = if (isFullscreen) 8.dp else extraBottomSpacing),
                    ) {
                        Surface(
                            modifier = Modifier.fillMaxSize(),
                            shape = RoundedCornerShape(if (isFullscreen) 18.dp else 24.dp),
                            tonalElevation = 2.dp,
                        ) {
                            Box(modifier = Modifier.fillMaxSize()) {
                                AndroidView(
                                    modifier = Modifier.fillMaxSize(),
                                    factory = { androidContext ->
                                        WebView(androidContext).apply {
                                            webViewRef = this
                                            layoutParams = ViewGroup.LayoutParams(
                                                ViewGroup.LayoutParams.MATCH_PARENT,
                                                ViewGroup.LayoutParams.MATCH_PARENT,
                                            )
                                            val cookieManager = CookieManager.getInstance()
                                            cookieManager.setAcceptCookie(true)
                                            cookieManager.setAcceptThirdPartyCookies(this, true)
                                            settings.javaScriptEnabled = true
                                            settings.domStorageEnabled = true
                                            settings.loadsImagesAutomatically = true
                                            settings.javaScriptCanOpenWindowsAutomatically = true
                                            settings.setSupportMultipleWindows(true)
                                            settings.loadWithOverviewMode = true
                                            settings.useWideViewPort = true
                                            settings.builtInZoomControls = false
                                            settings.displayZoomControls = false
                                            settings.userAgentString = PORTAL_EMBED_USER_AGENT
                                            webChromeClient = WebChromeClient()
                                            webViewClient = object : WebViewClient() {
                                                override fun shouldOverrideUrlLoading(
                                                    view: WebView?,
                                                    request: WebResourceRequest?,
                                                ): Boolean = false

                                                override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                                                    isLoading = true
                                                    pageError = null
                                                }

                                                override fun onPageFinished(view: WebView?, url: String?) {
                                                    isLoading = false
                                                    pageError = null
                                                }

                                                override fun onReceivedError(
                                                    view: WebView?,
                                                    request: WebResourceRequest?,
                                                    error: WebResourceError?,
                                                ) {
                                                    if (request?.isForMainFrame != false) {
                                                        isLoading = false
                                                        pageError = error?.description?.toString() ?: "Failed to load Nous Portal"
                                                    }
                                                }

                                                override fun onReceivedHttpError(
                                                    view: WebView?,
                                                    request: WebResourceRequest?,
                                                    errorResponse: WebResourceResponse?,
                                                ) {
                                                    if (request?.isForMainFrame != false) {
                                                        isLoading = false
                                                        pageError = "Nous Portal returned HTTP ${errorResponse?.statusCode ?: "error"}"
                                                    }
                                                }
                                            }
                                            loadUrl(uiState.portalUrl)
                                        }
                                    },
                                    update = { webView ->
                                        webViewRef = webView
                                        if (webView.url != uiState.portalUrl) {
                                            isLoading = true
                                            pageError = null
                                            webView.loadUrl(uiState.portalUrl)
                                        }
                                    },
                                )
                                Row(
                                    modifier = Modifier
                                        .align(Alignment.TopEnd)
                                        .padding(12.dp),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                ) {
                                    Surface(
                                        shape = RoundedCornerShape(999.dp),
                                        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.92f),
                                    ) {
                                        IconButton(onClick = { isFullscreen = !isFullscreen }) {
                                            Icon(
                                                painter = painterResource(id = if (isFullscreen) R.drawable.ic_action_minimize else R.drawable.ic_action_fullscreen),
                                                contentDescription = if (isFullscreen) "Minimize portal" else "Full screen portal",
                                                tint = MaterialTheme.colorScheme.primary,
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PortalGuidanceCard(
    status: String,
    inferenceUrl: String,
    pageError: String?,
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.large,
        tonalElevation = 2.dp,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Nous Portal", style = MaterialTheme.typography.titleMedium)
            Text(status, style = MaterialTheme.typography.bodySmall)
            Text(
                "The embedded portal now auto-loads on this page. Use the top-right full screen button to maximize or minimize the preview, or fall back to the browser if verification gets stuck.",
                style = MaterialTheme.typography.bodySmall,
            )
            if (inferenceUrl.isNotBlank()) {
                Text("Inference: $inferenceUrl", style = MaterialTheme.typography.labelMedium)
            }
            if (!pageError.isNullOrBlank()) {
                Text(pageError, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}
