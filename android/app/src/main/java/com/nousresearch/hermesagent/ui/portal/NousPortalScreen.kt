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
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import com.nousresearch.hermesagent.R
import com.nousresearch.hermesagent.backend.HermesRuntimeManager
import com.nousresearch.hermesagent.data.AppSettingsStore
import com.nousresearch.hermesagent.ui.i18n.AppLanguage
import com.nousresearch.hermesagent.ui.i18n.LocalHermesStrings
import com.nousresearch.hermesagent.ui.i18n.hermesStringsFor
import com.nousresearch.hermesagent.ui.shell.ShellActionItem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

private const val DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"
private const val PORTAL_EMBED_USER_AGENT = "Mozilla/5.0 (Linux; Android 15; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"

data class NousPortalUiState(
    val portalUrl: String = DEFAULT_NOUS_PORTAL_URL,
    val loggedIn: Boolean = false,
    val inferenceUrl: String = "",
    val portalEnabled: Boolean = true,
    val offlineAirplaneMode: Boolean = false,
    val status: String = "Loading Nous Portal…",
)

class NousPortalViewModel(application: Application) : AndroidViewModel(application) {
    private val settingsStore = AppSettingsStore(getApplication())
    private fun currentStrings() = hermesStringsFor(AppLanguage.fromTag(settingsStore.load().languageTag))

    private val _uiState = MutableStateFlow(NousPortalUiState())
    val uiState: StateFlow<NousPortalUiState> = _uiState.asStateFlow()

    fun refresh() {
        viewModelScope.launch {
            val settings = settingsStore.load()
            val strings = currentStrings()
            if (!settings.portalEnabled || settings.offlineAirplaneMode) {
                _uiState.value = NousPortalUiState(
                    portalEnabled = settings.portalEnabled,
                    offlineAirplaneMode = settings.offlineAirplaneMode,
                    status = if (settings.offlineAirplaneMode) {
                        "Offline airplane mode is on, so Nous Portal is blocked."
                    } else {
                        "Nous Portal is disabled on this device."
                    },
                )
                return@launch
            }
            _uiState.value = runCatching {
                val payload = withContext(Dispatchers.IO) {
                    HermesRuntimeManager.ensurePythonStarted(getApplication())
                    Python.getInstance()
                        .getModule("hermes_android.nous_portal_bridge")
                        .callAttr("read_nous_portal_state_json")
                        .toString()
                }
                val json = JSONObject(payload)
                val loggedIn = json.optBoolean("logged_in", false)
                NousPortalUiState(
                    portalUrl = json.optString("portal_url").ifBlank { DEFAULT_NOUS_PORTAL_URL },
                    loggedIn = loggedIn,
                    inferenceUrl = json.optString("inference_url").orEmpty(),
                    portalEnabled = settings.portalEnabled,
                    offlineAirplaneMode = settings.offlineAirplaneMode,
                    status = strings.portalLoadingStatus(loggedIn),
                )
            }.getOrElse { error ->
                NousPortalUiState(
                    portalUrl = DEFAULT_NOUS_PORTAL_URL,
                    portalEnabled = settings.portalEnabled,
                    offlineAirplaneMode = settings.offlineAirplaneMode,
                    status = strings.portalFallbackStatus(error.message ?: error.javaClass.simpleName),
                )
            }
        }
    }

    fun setPortalEnabled(enabled: Boolean) {
        val updated = settingsStore.load().copy(portalEnabled = enabled)
        settingsStore.save(updated)
        _uiState.value = _uiState.value.copy(
            portalEnabled = enabled,
            offlineAirplaneMode = updated.offlineAirplaneMode,
            status = if (enabled) "Nous Portal is enabled." else "Nous Portal is disabled on this device.",
        )
        if (enabled) {
            refresh()
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
    val strings = LocalHermesStrings.current
    val context = LocalContext.current
    var isLoading by remember { mutableStateOf(true) }

    LaunchedEffect(strings.language) {
        viewModel.refresh()
    }
    var pageError by remember { mutableStateOf<String?>(null) }
    var webViewRef by remember { mutableStateOf<WebView?>(null) }
    var isFullscreen by rememberSaveable { mutableStateOf(false) }
    val portalAvailable = uiState.portalEnabled && !uiState.offlineAirplaneMode

    LaunchedEffect(portalAvailable) {
        if (!portalAvailable) {
            isLoading = false
            pageError = null
            webViewRef?.loadUrl("about:blank")
        }
    }

    SideEffect {
        onContextActionsChanged(
            listOf(
                // label = "Refresh portal"
                ShellActionItem(
                    label = strings.refreshPortal.ifBlank { "Refresh portal" },
                    description = "Reload the embedded Nous Portal page.",
                    iconRes = R.drawable.ic_action_refresh,
                    onClick = {
                        if (portalAvailable) {
                            isLoading = true
                            pageError = null
                            viewModel.refresh()
                            webViewRef?.reload()
                        }
                    },
                ),
                ShellActionItem(
                    label = if (isFullscreen) strings.minimizePortal.ifBlank { "Minimize portal" } else strings.fullScreenPortal.ifBlank { "Full screen portal" },
                    description = "Resize the embedded portal preview without leaving the app.",
                    iconRes = if (isFullscreen) R.drawable.ic_action_minimize else R.drawable.ic_action_fullscreen,
                    onClick = { isFullscreen = !isFullscreen },
                ),
                // label = "Open externally"
                ShellActionItem(
                    label = strings.openExternally.ifBlank { "Open externally" },
                    description = "Open the full portal in your browser if the embed is limited.",
                    iconRes = R.drawable.ic_action_external,
                    onClick = {
                        if (portalAvailable) {
                            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(uiState.portalUrl))
                            context.startActivity(intent)
                        }
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
                            portalEnabled = uiState.portalEnabled,
                            offlineAirplaneMode = uiState.offlineAirplaneMode,
                            onPortalEnabledChange = viewModel::setPortalEnabled,
                        )
                    }
                    if (isLoading && portalAvailable) {
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
                                if (portalAvailable) {
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
                                } else {
                                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                        Text(
                                            if (uiState.offlineAirplaneMode) {
                                                "Portal network access is blocked by offline airplane mode."
                                            } else {
                                                "Portal is disabled."
                                            },
                                            style = MaterialTheme.typography.bodyMedium,
                                        )
                                    }
                                }
                                if (portalAvailable) Row(
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
                                                contentDescription = if (isFullscreen) strings.minimizePortal.ifBlank { "Minimize portal" } else strings.fullScreenPortal.ifBlank { "Full screen portal" },
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
    portalEnabled: Boolean,
    offlineAirplaneMode: Boolean,
    onPortalEnabledChange: (Boolean) -> Unit,
) {
    val strings = LocalHermesStrings.current
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
            Text(strings.portalTitle.ifBlank { "Nous Portal" }, style = MaterialTheme.typography.titleMedium)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Portal enabled", style = MaterialTheme.typography.titleSmall)
                Switch(
                    checked = portalEnabled,
                    onCheckedChange = onPortalEnabledChange,
                    enabled = !offlineAirplaneMode,
                )
            }
            Text(status, style = MaterialTheme.typography.bodySmall)
            Text(
                strings.portalEmbeddedDescription.ifBlank {
                    "The embedded portal now auto-loads on this page. Use the top-right full screen button to maximize or minimize the preview, or fall back to the browser if verification gets stuck."
                },
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
