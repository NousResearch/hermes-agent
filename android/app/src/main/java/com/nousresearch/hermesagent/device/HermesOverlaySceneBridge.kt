package com.nousresearch.hermesagent.device

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.TextUtils
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

object HermesOverlaySceneBridge {
    fun performSceneJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        return when (action.lowercase(Locale.US).ifBlank { "overlay_scene_status" }) {
            "overlay_scene_status", "scene_status", "overlay_status" -> statusJson(context)
            "show_overlay_scene", "show_scene", "overlay_scene" -> showSceneJson(context, arguments)
            "hide_overlay_scene", "dismiss_overlay_scene", "clear_overlay_scene", "hide_scene" -> hideSceneJson(context)
            else -> JSONObject()
                .put("success", false)
                .put("error", "Unsupported overlay scene action: $action")
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
    }

    fun statusJson(context: Context): String {
        val overlayPermissionGranted = Settings.canDrawOverlays(context.applicationContext)
        return JSONObject()
            .put("success", true)
            .put("shown", currentView != null)
            .put("current_scene_id", currentSceneId ?: JSONObject.NULL)
            .put("overlay_permission_granted", overlayPermissionGranted)
            .put("requires_overlay_permission", !overlayPermissionGranted)
            .put("available_actions", JSONArray(ACTIONS))
            .toString()
    }

    fun showSceneJson(context: Context, arguments: JSONObject): String {
        val payload = runCatching { payloadFromArguments(arguments) }.getOrElse { error ->
            return errorJson(error.message ?: "show_overlay_scene arguments are invalid")
        }
        val appContext = context.applicationContext
        if (!Settings.canDrawOverlays(appContext)) {
            return JSONObject()
                .put("success", false)
                .put("exit_code", 3)
                .put("action", "show_overlay_scene")
                .put("error", "Android overlay permission is not granted. Run open_overlay_settings first, then enable draw-over-other-apps for Hermes.")
                .put("requires_overlay_permission", true)
                .put("overlay_permission_granted", false)
                .put("payload", payload)
                .put("available_actions", JSONArray(ACTIONS))
                .toString()
        }
        return runOnMainThread {
            val manager = appContext.getSystemService(Context.WINDOW_SERVICE) as WindowManager
            removeCurrentView(manager)
            val layoutMetrics = resolvedLayoutMetrics(appContext, payload)
            val view = buildSceneView(appContext, payload, layoutMetrics)
            manager.addView(view, layoutParams(appContext, payload, layoutMetrics))
            currentView = view
            currentSceneId = payload.optString("scene_id").ifBlank { DEFAULT_SCENE_ID }
            val sceneToken = currentSceneToken + 1L
            currentSceneToken = sceneToken
            val hideAfterMs = payload.optLong("hide_after_ms", 0L)
            if (hideAfterMs > 0L) {
                mainHandler.postDelayed({
                    if (currentSceneToken == sceneToken) {
                        hideSceneJson(appContext)
                    }
                }, hideAfterMs)
            }
            JSONObject()
                .put("success", true)
                .put("exit_code", 0)
                .put("action", "show_overlay_scene")
                .put("scene_id", currentSceneId)
                .put("hide_after_ms", hideAfterMs)
                .put("overlay_permission_granted", true)
                .put("layout", layoutMetrics.toJson())
                .put("message", "Overlay scene shown")
                .toString()
        }
    }

    fun hideSceneJson(context: Context): String {
        return runOnMainThread {
            val manager = context.applicationContext.getSystemService(Context.WINDOW_SERVICE) as WindowManager
            val hadView = removeCurrentView(manager)
            JSONObject()
                .put("success", true)
                .put("exit_code", 0)
                .put("action", "hide_overlay_scene")
                .put("dismissed", hadView)
                .put("message", if (hadView) "Overlay scene hidden" else "No overlay scene was shown")
                .toString()
        }
    }

    internal fun payloadFromArguments(arguments: JSONObject): JSONObject {
        val sceneAction = normalizeSceneAction(
            stringArgument(arguments, "scene_action", "overlay_action", "mode", "action_type")
                .orEmpty()
                .ifBlank { "show" },
        )
        val sceneId = stringArgument(arguments, "scene_id", "id", "name")
            ?.trim()
            ?.take(MAX_SCENE_ID_CHARS)
            ?.ifBlank { DEFAULT_SCENE_ID }
            ?: DEFAULT_SCENE_ID
        rejectNul(sceneId, "scene_id")
        if (sceneAction == "hide") {
            return JSONObject()
                .put("scene_action", sceneAction)
                .put("scene_id", sceneId)
        }

        val title = stringArgument(arguments, "scene_title", "title", "heading")
            ?.take(MAX_TITLE_CHARS)
            ?: "Hermes"
        val text = stringArgument(arguments, "scene_text", "message", "text", "content", allowEmpty = true)
            ?: throw IllegalArgumentException("show_overlay_scene requires scene_text, message, text, or content")
        val buttonText = stringArgument(arguments, "scene_button_text", "button_text", "dismiss_text")
            ?.take(MAX_BUTTON_CHARS)
            ?: "Dismiss"
        rejectNul(title, "scene_title")
        rejectNul(text, "scene_text")
        rejectNul(buttonText, "scene_button_text")
        val position = normalizePosition(
            stringArgument(arguments, "scene_position", "position", "gravity")
                .orEmpty()
                .ifBlank { "center" },
        )
        val widthRequest = widthRequestFromArguments(arguments)
        val hideAfterMs = longArgument(arguments, "scene_hide_after_ms", "hide_after_ms", "timeout_ms", "duration_ms")
            ?.coerceIn(MIN_HIDE_AFTER_MS, MAX_HIDE_AFTER_MS)
            ?: 0L
        return JSONObject()
            .put("scene_action", sceneAction)
            .put("scene_id", sceneId)
            .put("title", title)
            .put("text", text.take(MAX_TEXT_CHARS))
            .put("button_text", buttonText)
            .put("position", position)
            .put("width_dp", widthRequest.widthDp)
            .put("width_mode", widthRequest.mode)
            .put("width_fraction", widthRequest.widthFraction ?: JSONObject.NULL)
            .put("width_px", widthRequest.widthPx ?: JSONObject.NULL)
            .put("hide_after_ms", hideAfterMs)
    }

    private fun buildSceneView(context: Context, payload: JSONObject, layoutMetrics: OverlayLayoutMetrics): View {
        val density = context.resources.displayMetrics.density
        val padding = (18 * density).toInt()
        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(padding, padding, padding, padding)
            background = GradientDrawable().apply {
                setColor(Color.rgb(27, 29, 43))
                setStroke((1 * density).toInt().coerceAtLeast(1), Color.rgb(128, 112, 255))
                cornerRadius = 18 * density
            }
            elevation = 12 * density
            isClickable = true
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
        }
        container.addView(TextView(context).apply {
            text = payload.optString("title")
            setTextColor(Color.WHITE)
            textSize = 18f
            typeface = Typeface.DEFAULT_BOLD
        })
        container.addView(TextView(context).apply {
            text = payload.optString("text")
            setTextColor(Color.rgb(224, 226, 238))
            textSize = 15f
            setSingleLine(false)
            maxLines = layoutMetrics.textMaxLines
            maxHeight = layoutMetrics.textMaxHeightPx
            ellipsize = TextUtils.TruncateAt.END
            setPadding(0, (10 * density).toInt(), 0, (14 * density).toInt())
        })
        container.addView(TextView(context).apply {
            text = payload.optString("button_text")
            setTextColor(Color.WHITE)
            textSize = 14f
            gravity = Gravity.CENTER
            typeface = Typeface.DEFAULT_BOLD
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
            setPadding((18 * density).toInt(), (10 * density).toInt(), (18 * density).toInt(), (10 * density).toInt())
            background = GradientDrawable().apply {
                setColor(Color.rgb(132, 113, 246))
                cornerRadius = 24 * density
            }
            setOnClickListener { hideSceneJson(context) }
        })
        return container
    }

    private fun layoutParams(context: Context, payload: JSONObject, layoutMetrics: OverlayLayoutMetrics): WindowManager.LayoutParams {
        return WindowManager.LayoutParams(
            layoutMetrics.resolvedWidthPx,
            WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            } else {
                @Suppress("DEPRECATION")
                WindowManager.LayoutParams.TYPE_PHONE
            },
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = gravityForPosition(payload.optString("position"))
            y = when (payload.optString("position")) {
                "top" -> layoutMetrics.verticalInsetPx
                "bottom" -> -layoutMetrics.verticalInsetPx
                else -> 0
            }
        }
    }

    internal fun resolvedLayoutMetrics(context: Context, payload: JSONObject): OverlayLayoutMetrics {
        val density = context.resources.displayMetrics.density.takeIf { it > 0f } ?: 1f
        val resourcesMetrics = context.resources.displayMetrics
        var screenWidthPx = resourcesMetrics.widthPixels
        var screenHeightPx = resourcesMetrics.heightPixels
        var safeInsetLeftPx = 0
        var safeInsetTopPx = 0
        var safeInsetRightPx = 0
        var safeInsetBottomPx = 0

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            runCatching {
                val manager = context.applicationContext.getSystemService(Context.WINDOW_SERVICE) as WindowManager
                val windowMetrics = manager.currentWindowMetrics
                val bounds = windowMetrics.bounds
                if (bounds.width() > 0 && bounds.height() > 0) {
                    screenWidthPx = bounds.width()
                    screenHeightPx = bounds.height()
                }
                val insets = windowMetrics.windowInsets.getInsetsIgnoringVisibility(
                    WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout(),
                )
                safeInsetLeftPx = insets.left.coerceAtLeast(0)
                safeInsetTopPx = insets.top.coerceAtLeast(0)
                safeInsetRightPx = insets.right.coerceAtLeast(0)
                safeInsetBottomPx = insets.bottom.coerceAtLeast(0)
            }
        }

        if (screenWidthPx <= 0) {
            screenWidthPx = (DEFAULT_WIDTH_DP * density).roundToInt()
        }
        if (screenHeightPx <= 0) {
            screenHeightPx = (640 * density).roundToInt()
        }

        val usableWidthPx = (screenWidthPx - safeInsetLeftPx - safeInsetRightPx)
            .coerceAtLeast((160 * density).roundToInt())
        val usableHeightPx = (screenHeightPx - safeInsetTopPx - safeInsetBottomPx)
            .coerceAtLeast((240 * density).roundToInt())
        val shortEdgePx = min(usableWidthPx, usableHeightPx)
        val longEdgePx = max(usableWidthPx, usableHeightPx)
        val shortEdgeDp = shortEdgePx / density
        val usableHeightDp = usableHeightPx / density
        val orientation = if (usableWidthPx >= usableHeightPx) "landscape" else "portrait"

        val requestedWidthDp = payload.optInt("width_dp", DEFAULT_WIDTH_DP)
            .coerceIn(MIN_WIDTH_DP, MAX_WIDTH_DP)
        val widthMode = payload.optString("width_mode").ifBlank { WIDTH_MODE_DP }
        val requestedWidthPx = when (widthMode) {
            WIDTH_MODE_FRACTION -> {
                val fraction = payload.optString("width_fraction")
                    .toDoubleOrNull()
                    ?.coerceIn(MIN_WIDTH_FRACTION, MAX_WIDTH_FRACTION)
                    ?: DEFAULT_WIDTH_FRACTION
                (usableWidthPx * fraction).roundToInt()
            }
            WIDTH_MODE_PX -> payload.optInt("width_px", 0)
                .takeIf { it > 0 }
                ?: (requestedWidthDp * density).roundToInt()
            else -> (requestedWidthDp * density).roundToInt()
        }
        val edgeMarginDp = when {
            shortEdgeDp < 360f -> 8
            shortEdgeDp < 600f -> 12
            else -> OVERLAY_EDGE_MARGIN_DP
        }
        val edgeMarginPx = (edgeMarginDp * density).roundToInt().coerceAtLeast(1)
        val maxWidthFraction = when {
            shortEdgeDp < 360f -> 0.96f
            orientation == "landscape" -> 0.72f
            else -> 0.92f
        }
        val availableWidthPx = min(
            usableWidthPx - (edgeMarginPx * 2),
            (usableWidthPx * maxWidthFraction).roundToInt(),
        ).coerceAtLeast((160 * density).roundToInt())
        val minWidthPx = (MIN_WIDTH_DP * density).roundToInt().coerceAtMost(availableWidthPx)
        val resolvedWidthPx = requestedWidthPx.coerceIn(minWidthPx, availableWidthPx)
        val verticalInsetPx = ((usableHeightPx * 0.08f).roundToInt())
            .coerceIn((20 * density).roundToInt(), (72 * density).roundToInt())
        val maxHeightPx = (usableHeightPx - (verticalInsetPx * 2))
            .coerceAtLeast((160 * density).roundToInt())
        val textMaxLines = when {
            shortEdgeDp < 360f || usableHeightDp < 520f -> 6
            orientation == "landscape" -> 8
            else -> 12
        }
        val textMaxHeightPx = (maxHeightPx - (112 * density).roundToInt())
            .coerceAtLeast((96 * density).roundToInt())
        return OverlayLayoutMetrics(
            screenWidthPx = screenWidthPx,
            screenHeightPx = screenHeightPx,
            usableWidthPx = usableWidthPx,
            usableHeightPx = usableHeightPx,
            safeInsetLeftPx = safeInsetLeftPx,
            safeInsetTopPx = safeInsetTopPx,
            safeInsetRightPx = safeInsetRightPx,
            safeInsetBottomPx = safeInsetBottomPx,
            density = density,
            orientation = orientation,
            shortEdgePx = shortEdgePx,
            longEdgePx = longEdgePx,
            widthMode = widthMode,
            requestedWidthDp = requestedWidthDp,
            requestedWidthPx = requestedWidthPx,
            availableWidthPx = availableWidthPx,
            resolvedWidthPx = resolvedWidthPx,
            edgeMarginPx = edgeMarginPx,
            verticalInsetPx = verticalInsetPx,
            maxHeightPx = maxHeightPx,
            textMaxHeightPx = textMaxHeightPx,
            textMaxLines = textMaxLines,
        )
    }

    private fun gravityForPosition(position: String): Int {
        return when (position) {
            "top" -> Gravity.TOP or Gravity.CENTER_HORIZONTAL
            "bottom" -> Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            else -> Gravity.CENTER
        }
    }

    private fun runOnMainThread(block: () -> String): String {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            return runCatching(block).getOrElse { error -> errorJson(error.message ?: error.javaClass.simpleName) }
        }
        val result = AtomicReference<String>()
        val latch = CountDownLatch(1)
        mainHandler.post {
            result.set(runCatching(block).getOrElse { error -> errorJson(error.message ?: error.javaClass.simpleName) })
            latch.countDown()
        }
        if (!latch.await(2, TimeUnit.SECONDS)) {
            return errorJson("Timed out while updating overlay scene")
        }
        return result.get() ?: errorJson("Overlay scene update did not return a result")
    }

    private fun removeCurrentView(manager: WindowManager): Boolean {
        val view = currentView ?: return false
        runCatching { manager.removeView(view) }
        currentView = null
        currentSceneId = null
        currentSceneToken += 1L
        return true
    }

    private fun normalizeSceneAction(value: String): String {
        return when (value.trim().lowercase(Locale.US)) {
            "", "show", "display", "update" -> "show"
            "hide", "dismiss", "clear", "remove" -> "hide"
            else -> throw IllegalArgumentException("Unsupported scene_action: $value")
        }
    }

    private fun normalizePosition(value: String): String {
        return when (value.trim().lowercase(Locale.US)) {
            "top", "upper" -> "top"
            "bottom", "lower" -> "bottom"
            "", "center", "middle" -> "center"
            else -> throw IllegalArgumentException("scene_position must be top, center, or bottom")
        }
    }

    private fun widthRequestFromArguments(arguments: JSONObject): OverlayWidthRequest {
        explicitFractionArgument(arguments, "scene_width_fraction", "width_fraction", "scene_width_percent", "width_percent")?.let { fraction ->
            return OverlayWidthRequest(
                mode = WIDTH_MODE_FRACTION,
                widthDp = DEFAULT_WIDTH_DP,
                widthFraction = fraction.coerceIn(MIN_WIDTH_FRACTION, MAX_WIDTH_FRACTION),
            )
        }
        intArgument(arguments, "scene_width_px", "width_px")?.let { widthPx ->
            return OverlayWidthRequest(
                mode = WIDTH_MODE_PX,
                widthDp = DEFAULT_WIDTH_DP,
                widthPx = widthPx.coerceAtLeast(1),
            )
        }
        rawArgument(arguments, "scene_width", "width")?.let { raw ->
            parseWidthToken(raw)?.let { return it }
        }
        val widthDp = intArgument(arguments, "scene_width_dp", "width_dp", "width")
            ?.coerceIn(MIN_WIDTH_DP, MAX_WIDTH_DP)
            ?: DEFAULT_WIDTH_DP
        return OverlayWidthRequest(mode = WIDTH_MODE_DP, widthDp = widthDp)
    }

    private fun parseWidthToken(raw: Any): OverlayWidthRequest? {
        return when (raw) {
            is Number -> {
                val value = raw.toDouble()
                if (value > 0.0 && value <= 1.0) {
                    OverlayWidthRequest(
                        mode = WIDTH_MODE_FRACTION,
                        widthDp = DEFAULT_WIDTH_DP,
                        widthFraction = value.coerceIn(MIN_WIDTH_FRACTION, MAX_WIDTH_FRACTION),
                    )
                } else {
                    val intValue = value.roundToInt()
                    if (intValue > 0) {
                        OverlayWidthRequest(
                            mode = WIDTH_MODE_DP,
                            widthDp = intValue.coerceIn(MIN_WIDTH_DP, MAX_WIDTH_DP),
                        )
                    } else {
                        null
                    }
                }
            }
            is String -> {
                val trimmed = raw.trim().lowercase(Locale.US)
                when {
                    trimmed.endsWith("%") -> trimmed.dropLast(1).trim().toDoubleOrNull()?.let { percent ->
                        OverlayWidthRequest(
                            mode = WIDTH_MODE_FRACTION,
                            widthDp = DEFAULT_WIDTH_DP,
                            widthFraction = (percent / 100.0).coerceIn(MIN_WIDTH_FRACTION, MAX_WIDTH_FRACTION),
                        )
                    }
                    trimmed.endsWith("px") -> trimmed.dropLast(2).trim().toIntOrNull()?.takeIf { it > 0 }?.let { widthPx ->
                        OverlayWidthRequest(
                            mode = WIDTH_MODE_PX,
                            widthDp = DEFAULT_WIDTH_DP,
                            widthPx = widthPx,
                        )
                    }
                    trimmed.endsWith("dp") -> trimmed.dropLast(2).trim().toIntOrNull()?.takeIf { it > 0 }?.let { widthDp ->
                        OverlayWidthRequest(
                            mode = WIDTH_MODE_DP,
                            widthDp = widthDp.coerceIn(MIN_WIDTH_DP, MAX_WIDTH_DP),
                        )
                    }
                    else -> trimmed.toDoubleOrNull()?.let { value ->
                        parseWidthToken(value)
                    }
                }
            }
            else -> null
        }
    }

    private fun stringArgument(arguments: JSONObject, vararg names: String, allowEmpty: Boolean = false): String? {
        names.forEach { name ->
            if (arguments.has(name) && !arguments.isNull(name)) {
                val value = arguments.optString(name)
                if (allowEmpty || value.isNotBlank()) {
                    return value
                }
            }
        }
        return null
    }

    private fun rawArgument(arguments: JSONObject, vararg names: String): Any? {
        names.forEach { name ->
            if (arguments.has(name) && !arguments.isNull(name)) {
                return arguments.opt(name)
            }
        }
        return null
    }

    private fun intArgument(arguments: JSONObject, vararg names: String): Int? {
        names.forEach { name ->
            if (arguments.has(name) && !arguments.isNull(name)) {
                val value = when (val raw = arguments.opt(name)) {
                    is Number -> raw.toInt()
                    else -> raw?.toString()?.trim()?.toIntOrNull()
                }
                if (value != null) {
                    return value
                }
            }
        }
        return null
    }

    private fun explicitFractionArgument(arguments: JSONObject, vararg names: String): Double? {
        names.forEach { name ->
            if (arguments.has(name) && !arguments.isNull(name)) {
                val raw = arguments.opt(name)
                val value = when (raw) {
                    is Number -> raw.toDouble()
                    is String -> raw.trim().removeSuffix("%").toDoubleOrNull()?.let {
                        if (raw.trim().endsWith("%")) it / 100.0 else it
                    }
                    else -> null
                }
                if (value != null && value > 0.0) {
                    return if (value > 1.0) value / 100.0 else value
                }
            }
        }
        return null
    }

    private fun longArgument(arguments: JSONObject, vararg names: String): Long? {
        names.forEach { name ->
            if (arguments.has(name) && !arguments.isNull(name)) {
                val value = when (val raw = arguments.opt(name)) {
                    is Number -> raw.toLong()
                    else -> raw?.toString()?.trim()?.toLongOrNull()
                }
                if (value != null) {
                    return value
                }
            }
        }
        return null
    }

    private fun rejectNul(value: String, label: String) {
        if (value.indexOf('\u0000') >= 0) {
            throw IllegalArgumentException("$label must not contain NUL bytes")
        }
    }

    private fun errorJson(message: String): String {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 1)
            .put("error", message)
            .put("available_actions", JSONArray(ACTIONS))
            .toString()
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    private var currentView: View? = null

    @Volatile
    private var currentSceneId: String? = null

    @Volatile
    private var currentSceneToken: Long = 0L

    private val ACTIONS = listOf(
        "overlay_scene_status",
        "show_overlay_scene",
        "hide_overlay_scene",
    )

    private const val DEFAULT_SCENE_ID = "hermes_scene"
    private const val MAX_SCENE_ID_CHARS = 64
    private const val MAX_TITLE_CHARS = 80
    private const val MAX_TEXT_CHARS = 1000
    private const val MAX_BUTTON_CHARS = 32
    private const val DEFAULT_WIDTH_DP = 360
    private const val MIN_WIDTH_DP = 220
    private const val MAX_WIDTH_DP = 560
    private const val DEFAULT_WIDTH_FRACTION = 0.92
    private const val MIN_WIDTH_FRACTION = 0.35
    private const val MAX_WIDTH_FRACTION = 0.98
    private const val WIDTH_MODE_DP = "dp"
    private const val WIDTH_MODE_PX = "px"
    private const val WIDTH_MODE_FRACTION = "fraction"
    private const val OVERLAY_EDGE_MARGIN_DP = 16
    private const val MIN_HIDE_AFTER_MS = 1000L
    private const val MAX_HIDE_AFTER_MS = 600_000L
}

private data class OverlayWidthRequest(
    val mode: String,
    val widthDp: Int,
    val widthPx: Int? = null,
    val widthFraction: Double? = null,
)

internal data class OverlayLayoutMetrics(
    val screenWidthPx: Int,
    val screenHeightPx: Int,
    val usableWidthPx: Int,
    val usableHeightPx: Int,
    val safeInsetLeftPx: Int,
    val safeInsetTopPx: Int,
    val safeInsetRightPx: Int,
    val safeInsetBottomPx: Int,
    val density: Float,
    val orientation: String,
    val shortEdgePx: Int,
    val longEdgePx: Int,
    val widthMode: String,
    val requestedWidthDp: Int,
    val requestedWidthPx: Int,
    val availableWidthPx: Int,
    val resolvedWidthPx: Int,
    val edgeMarginPx: Int,
    val verticalInsetPx: Int,
    val maxHeightPx: Int,
    val textMaxHeightPx: Int,
    val textMaxLines: Int,
) {
    fun toJson(): JSONObject {
        return JSONObject()
            .put("screen_width_px", screenWidthPx)
            .put("screen_height_px", screenHeightPx)
            .put("usable_width_px", usableWidthPx)
            .put("usable_height_px", usableHeightPx)
            .put("safe_inset_left_px", safeInsetLeftPx)
            .put("safe_inset_top_px", safeInsetTopPx)
            .put("safe_inset_right_px", safeInsetRightPx)
            .put("safe_inset_bottom_px", safeInsetBottomPx)
            .put("density", density.toDouble())
            .put("orientation", orientation)
            .put("screen_aspect_ratio", if (shortEdgePx > 0) longEdgePx.toDouble() / shortEdgePx.toDouble() else 1.0)
            .put("short_edge_px", shortEdgePx)
            .put("long_edge_px", longEdgePx)
            .put("width_mode", widthMode)
            .put("requested_width_dp", requestedWidthDp)
            .put("requested_width_px", requestedWidthPx)
            .put("available_width_px", availableWidthPx)
            .put("resolved_width_px", resolvedWidthPx)
            .put("edge_margin_px", edgeMarginPx)
            .put("vertical_inset_px", verticalInsetPx)
            .put("max_height_px", maxHeightPx)
            .put("text_max_height_px", textMaxHeightPx)
            .put("text_max_lines", textMaxLines)
    }
}
