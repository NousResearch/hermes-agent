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
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
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
            val view = buildSceneView(appContext, payload)
            manager.addView(view, layoutParams(appContext, payload))
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
        val widthDp = intArgument(arguments, "scene_width_dp", "width_dp", "width")
            ?.coerceIn(MIN_WIDTH_DP, MAX_WIDTH_DP)
            ?: DEFAULT_WIDTH_DP
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
            .put("width_dp", widthDp)
            .put("hide_after_ms", hideAfterMs)
    }

    private fun buildSceneView(context: Context, payload: JSONObject): View {
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
            setPadding(0, (10 * density).toInt(), 0, (14 * density).toInt())
        })
        container.addView(TextView(context).apply {
            text = payload.optString("button_text")
            setTextColor(Color.WHITE)
            textSize = 14f
            gravity = Gravity.CENTER
            typeface = Typeface.DEFAULT_BOLD
            setPadding((18 * density).toInt(), (10 * density).toInt(), (18 * density).toInt(), (10 * density).toInt())
            background = GradientDrawable().apply {
                setColor(Color.rgb(132, 113, 246))
                cornerRadius = 24 * density
            }
            setOnClickListener { hideSceneJson(context) }
        })
        return container
    }

    private fun layoutParams(context: Context, payload: JSONObject): WindowManager.LayoutParams {
        val density = context.resources.displayMetrics.density
        val widthPx = (payload.optInt("width_dp", DEFAULT_WIDTH_DP) * density).toInt()
        return WindowManager.LayoutParams(
            widthPx,
            WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            } else {
                @Suppress("DEPRECATION")
                WindowManager.LayoutParams.TYPE_PHONE
            },
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = gravityForPosition(payload.optString("position"))
            y = when (payload.optString("position")) {
                "top" -> (72 * density).toInt()
                "bottom" -> -(72 * density).toInt()
                else -> 0
            }
        }
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
    private const val MIN_HIDE_AFTER_MS = 1000L
    private const val MAX_HIDE_AFTER_MS = 600_000L
}
