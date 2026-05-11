package com.nousresearch.hermesagent.device

import android.graphics.Rect
import android.os.Bundle
import android.view.accessibility.AccessibilityNodeInfo
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.security.MessageDigest

object HermesAccessibilityUiBridge {
    private const val DEFAULT_LIMIT = 80
    private const val MAX_LIMIT = 200
    private const val DEFAULT_TAP_DURATION_MS = 80L
    private const val DEFAULT_LONG_PRESS_DURATION_MS = 650L
    private const val DEFAULT_SWIPE_DURATION_MS = 450L
    private const val DEFAULT_SCROLL_DURATION_MS = 500L
    private const val MIN_GESTURE_DURATION_MS = 1L
    private const val MAX_GESTURE_DURATION_MS = 5_000L
    private const val DEFAULT_SCROLL_DISTANCE_FRACTION = 0.5f
    private val NORMALIZED_COORDINATE_SPACES = setOf("normalized", "normalised", "relative", "fraction", "unit", "unit_interval")
    private val PERCENT_COORDINATE_SPACES = setOf("percent", "percentage", "normalized_percent", "normalised_percent")

    @JvmStatic
    fun snapshotJson(limit: Int): String {
        return runCatching {
            val service = HermesAccessibilityController.currentService()
                ?: return errorJson("Hermes accessibility service is not connected")
            val root = service.rootInActiveWindow
                ?: return errorJson("No active accessibility window is available")
            val cappedLimit = limit.coerceIn(1, MAX_LIMIT)
            val nodes = flattenNodes(root, cappedLimit)
            val stateHash = uiStateHash(root, nodes)
            JSONObject().apply {
                put("accessibility_connected", true)
                put("active_package", root.packageName?.toString().orEmpty())
                put("current_app_name", root.packageName?.toString().orEmpty())
                put("ui_state_hash", stateHash)
                put("screen_hash", stateHash)
                put("screen_hash_kind", "accessibility_semantic_sha256_64")
                put("coordinate_space", "absolute_px")
                put("scale_factor", 1.0)
                put("normalized_coordinate_support", true)
                HermesAccessibilityController.screenMetrics()?.let { metrics ->
                    putScreenMetrics(metrics)
                }
                put("node_count", nodes.size)
                put("nodes", JSONArray(nodes.mapIndexed { index, node -> nodeJson(node, index) }))
            }.toString()
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    @JvmStatic
    fun performActionJson(
        action: String,
        textContains: String,
        contentDescriptionContains: String,
        viewId: String,
        packageName: String,
        className: String,
        value: String,
        index: Int,
    ): String {
        return runCatching {
            val service = HermesAccessibilityController.currentService()
                ?: return errorJson("Hermes accessibility service is not connected")
            val root = service.rootInActiveWindow
                ?: return errorJson("No active accessibility window is available")
            val nodes = flattenNodes(root, MAX_LIMIT)
            val matches = nodes.filter { node ->
                matchesSelector(node, textContains, contentDescriptionContains, viewId, packageName, className)
            }
            if (matches.isEmpty()) {
                return errorJson("No accessibility node matched the requested selector")
            }

            val resolvedIndex = index.coerceAtLeast(0)
            if (resolvedIndex >= matches.size) {
                return errorJson("Requested match index $resolvedIndex but only ${matches.size} node(s) matched")
            }
            val selected = matches[resolvedIndex]
            val performed = performResolvedAction(action, selected, value)
            JSONObject().apply {
                put("success", performed)
                put("action", action)
                put("matched_count", matches.size)
                put("matched_node", nodeJson(selected, resolvedIndex))
            }.toString()
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    @JvmStatic
    fun performCoordinateGestureJson(
        action: String,
        x: Double?,
        y: Double?,
        x1: Double?,
        y1: Double?,
        x2: Double?,
        y2: Double?,
        durationMs: Long,
        coordinateSpace: String,
    ): String {
        return runCatching {
            if (!HermesAccessibilityController.isServiceConnected()) {
                return errorJson("Hermes accessibility service is not connected")
            }
            val metrics = HermesAccessibilityController.screenMetrics()
                ?: return errorJson("Screen metrics are not available")
            if (metrics.width <= 0 || metrics.height <= 0) {
                return errorJson("Screen metrics are not valid")
            }

            val normalizedAction = normalizeCoordinateAction(action)
            val duration = resolvedGestureDuration(normalizedAction, durationMs)
            val points = mutableListOf<CoordinatePoint>()
            val performed = when (normalizedAction) {
                "tap", "long_press" -> {
                    val point = resolveCoordinatePoint(
                        x = x ?: x1,
                        y = y ?: y1,
                        metrics = metrics,
                        coordinateSpace = coordinateSpace,
                        xLabel = "x",
                        yLabel = "y",
                    )
                    points += point
                    HermesAccessibilityController.performTap(point.x, point.y, duration)
                }
                "swipe" -> {
                    val start = resolveCoordinatePoint(
                        x = x1 ?: x,
                        y = y1 ?: y,
                        metrics = metrics,
                        coordinateSpace = coordinateSpace,
                        xLabel = "x1",
                        yLabel = "y1",
                    )
                    val end = resolveCoordinatePoint(
                        x = x2,
                        y = y2,
                        metrics = metrics,
                        coordinateSpace = coordinateSpace,
                        xLabel = "x2",
                        yLabel = "y2",
                    )
                    points += start
                    points += end
                    HermesAccessibilityController.performSwipe(start.x, start.y, end.x, end.y, duration)
                }
                else -> return errorJson("Unsupported coordinate UI action: $action")
            }

            JSONObject().apply {
                put("success", performed)
                put("action", normalizedAction)
                put("accessibility_connected", true)
                put("current_app_name", currentAppName())
                put("coordinate_space", resolvedCoordinateSpaceLabel(coordinateSpace))
                put("requested_coordinate_space", coordinateSpace.ifBlank { "absolute_px" })
                put("duration_ms", duration)
                putScreenMetrics(metrics)
                put("scale_factor", 1.0)
                put("resolved_coordinates", JSONArray(points.map { point -> point.toJson() }))
                put(
                    "message",
                    if (performed) {
                        "Dispatched Android accessibility gesture: $normalizedAction"
                    } else {
                        "Android rejected accessibility gesture: $normalizedAction"
                    },
                )
            }.toString()
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    @JvmStatic
    fun performScrollGestureJson(
        direction: String,
        x: Double?,
        y: Double?,
        distancePx: Double?,
        durationMs: Long,
        coordinateSpace: String,
    ): String {
        return runCatching {
            if (!HermesAccessibilityController.isServiceConnected()) {
                return errorJson("Hermes accessibility service is not connected")
            }
            val metrics = HermesAccessibilityController.screenMetrics()
                ?: return errorJson("Screen metrics are not available")
            if (metrics.width <= 0 || metrics.height <= 0) {
                return errorJson("Screen metrics are not valid")
            }

            val normalizedDirection = normalizeScrollDirection(direction)
            val start = if (x != null || y != null) {
                resolveCoordinatePoint(
                    x = x,
                    y = y,
                    metrics = metrics,
                    coordinateSpace = coordinateSpace,
                    xLabel = "x",
                    yLabel = "y",
                )
            } else {
                defaultScrollStartPoint(normalizedDirection, metrics)
            }
            val distance = resolvedScrollDistance(normalizedDirection, distancePx, metrics)
            val end = scrollEndPoint(start, normalizedDirection, distance, metrics)
            val duration = (durationMs.takeIf { it > 0L } ?: DEFAULT_SCROLL_DURATION_MS)
                .coerceIn(MIN_GESTURE_DURATION_MS, MAX_GESTURE_DURATION_MS)
            val performed = HermesAccessibilityController.performSwipe(start.x, start.y, end.x, end.y, duration)

            JSONObject().apply {
                put("success", performed)
                put("action", "scroll")
                put("direction", normalizedDirection)
                put("accessibility_connected", true)
                put("current_app_name", currentAppName())
                put("coordinate_space", resolvedCoordinateSpaceLabel(coordinateSpace))
                put("requested_coordinate_space", coordinateSpace.ifBlank { "absolute_px" })
                put("duration_ms", duration)
                put("distance_px", distance.toDouble())
                putScreenMetrics(metrics)
                put("scale_factor", 1.0)
                put("resolved_coordinates", JSONArray(listOf(start, end).map { point -> point.toJson() }))
                put(
                    "message",
                    if (performed) {
                        "Dispatched Android accessibility scroll gesture: $normalizedDirection"
                    } else {
                        "Android rejected accessibility scroll gesture: $normalizedDirection"
                    },
                )
            }.toString()
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    @JvmStatic
    fun performTextInputJson(
        value: String,
        textContains: String,
        contentDescriptionContains: String,
        viewId: String,
        packageName: String,
        className: String,
        index: Int,
    ): String {
        return runCatching {
            val service = HermesAccessibilityController.currentService()
                ?: return errorJson("Hermes accessibility service is not connected")
            val root = service.rootInActiveWindow
                ?: return errorJson("No active accessibility window is available")
            val nodes = flattenNodes(root, MAX_LIMIT)
            val hasSelector = textContains.isNotBlank() ||
                contentDescriptionContains.isNotBlank() ||
                viewId.isNotBlank() ||
                packageName.isNotBlank() ||
                className.isNotBlank()
            val target = if (hasSelector) {
                val matches = nodes.filter { node ->
                    matchesSelector(node, textContains, contentDescriptionContains, viewId, packageName, className)
                }
                val resolvedIndex = index.coerceAtLeast(0)
                if (matches.isEmpty()) {
                    return errorJson("No accessibility node matched the requested selector")
                }
                if (resolvedIndex >= matches.size) {
                    return errorJson("Requested match index $resolvedIndex but only ${matches.size} node(s) matched")
                }
                findEditableNode(matches[resolvedIndex])
                    ?: throw IOException("No editable accessibility node matched the selector")
            } else {
                nodes.firstOrNull { node -> node.isFocused && isEditableNode(node) }
                    ?: nodes.firstOrNull { node -> isEditableNode(node) }
                    ?: throw IOException("No focused or editable accessibility node is available")
            }
            val arguments = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, value)
            }
            val performed = target.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)
            JSONObject().apply {
                put("success", performed)
                put("action", "type")
                put("accessibility_connected", true)
                put("current_app_name", currentAppName())
                put("matched_node", nodeJson(target, nodes.indexOf(target).coerceAtLeast(0)))
                put(
                    "message",
                    if (performed) {
                        "Set text on Android editable field"
                    } else {
                        "Android rejected text input for editable field"
                    },
                )
            }.toString()
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    @JvmStatic
    fun performGlobalActionJson(action: String): String {
        val normalizedAction = normalizeGlobalAction(action)
        val globalAction = when (normalizedAction) {
            "back", "global_back" -> HermesGlobalAction.Back
            "home", "global_home" -> HermesGlobalAction.Home
            "recents", "global_recents" -> HermesGlobalAction.Recents
            "notifications", "global_notifications" -> HermesGlobalAction.Notifications
            "quick_settings", "global_quick_settings" -> HermesGlobalAction.QuickSettings
            else -> return errorJson("Unsupported Android global UI action: $action")
        }
        val connected = HermesAccessibilityController.isServiceConnected()
        val success = connected && HermesAccessibilityController.performAction(globalAction)
        return JSONObject()
            .put("success", success)
            .put("action", normalizedAction)
            .put("accessibility_connected", connected)
            .put(
                "message",
                if (success) {
                    "Performed Android global action: $normalizedAction"
                } else {
                    "Hermes accessibility service is not connected or Android rejected global action: $normalizedAction"
                },
            )
            .toString()
    }

    private fun normalizeGlobalAction(action: String): String {
        return action.trim().lowercase().replace("-", "_").replace(" ", "_")
    }

    private fun normalizeCoordinateAction(action: String): String {
        return when (action.trim().lowercase().replace("-", "_").replace(" ", "_")) {
            "tap_at", "click_at", "coordinate_tap", "coordinate_click", "gesture_tap" -> "tap"
            "long_press_at", "coordinate_long_press", "gesture_long_press" -> "long_press"
            "drag", "coordinate_swipe", "gesture_swipe" -> "swipe"
            else -> action.trim().lowercase().replace("-", "_").replace(" ", "_")
        }
    }

    private fun normalizeScrollDirection(direction: String): String {
        return when (direction.trim().lowercase().replace("-", "_").replace(" ", "_")) {
            "down", "scroll_down", "finger_down" -> "down"
            "left", "scroll_left", "finger_left" -> "left"
            "right", "scroll_right", "finger_right" -> "right"
            "up", "scroll_up", "finger_up", "" -> "up"
            else -> throw IOException("Unsupported scroll direction: $direction")
        }
    }

    private fun resolvedGestureDuration(action: String, durationMs: Long): Long {
        val defaultDuration = when (action) {
            "long_press" -> DEFAULT_LONG_PRESS_DURATION_MS
            "swipe" -> DEFAULT_SWIPE_DURATION_MS
            else -> DEFAULT_TAP_DURATION_MS
        }
        return durationMs.takeIf { it > 0L }
            ?.coerceIn(MIN_GESTURE_DURATION_MS, MAX_GESTURE_DURATION_MS)
            ?: defaultDuration
    }

    private fun resolveCoordinatePoint(
        x: Double?,
        y: Double?,
        metrics: HermesScreenMetrics,
        coordinateSpace: String,
        xLabel: String,
        yLabel: String,
    ): CoordinatePoint {
        return CoordinatePoint(
            x = resolveCoordinate(x, metrics.width, coordinateSpace, xLabel),
            y = resolveCoordinate(y, metrics.height, coordinateSpace, yLabel),
        )
    }

    private fun resolveCoordinate(rawValue: Double?, axisSize: Int, coordinateSpace: String, label: String): Float {
        val value = rawValue ?: throw IOException("$label coordinate is required")
        if (value.isNaN() || value.isInfinite()) {
            throw IOException("$label coordinate must be finite")
        }
        val axisMax = (axisSize - 1).coerceAtLeast(0).toDouble()
        val resolved = when (resolvedCoordinateSpaceLabel(coordinateSpace)) {
            "normalized" -> value * axisMax
            "percent" -> (value / 100.0) * axisMax
            else -> value
        }
        return resolved.coerceIn(0.0, axisMax).toFloat()
    }

    private fun defaultScrollStartPoint(direction: String, metrics: HermesScreenMetrics): CoordinatePoint {
        val widthMax = (metrics.width - 1).coerceAtLeast(0).toFloat()
        val heightMax = (metrics.height - 1).coerceAtLeast(0).toFloat()
        return when (direction) {
            "down" -> CoordinatePoint(widthMax * 0.5f, heightMax * 0.25f)
            "left" -> CoordinatePoint(widthMax * 0.75f, heightMax * 0.5f)
            "right" -> CoordinatePoint(widthMax * 0.25f, heightMax * 0.5f)
            else -> CoordinatePoint(widthMax * 0.5f, heightMax * 0.75f)
        }
    }

    private fun resolvedScrollDistance(direction: String, distancePx: Double?, metrics: HermesScreenMetrics): Float {
        val axis = if (direction == "left" || direction == "right") metrics.width else metrics.height
        val defaultDistance = axis * DEFAULT_SCROLL_DISTANCE_FRACTION
        val requested = distancePx?.toFloat()?.takeIf { it.isFinite() && it > 0f }
        return (requested ?: defaultDistance).coerceIn(1f, axis.toFloat().coerceAtLeast(1f))
    }

    private fun scrollEndPoint(
        start: CoordinatePoint,
        direction: String,
        distance: Float,
        metrics: HermesScreenMetrics,
    ): CoordinatePoint {
        val maxX = (metrics.width - 1).coerceAtLeast(0).toFloat()
        val maxY = (metrics.height - 1).coerceAtLeast(0).toFloat()
        return when (direction) {
            "down" -> CoordinatePoint(start.x, (start.y + distance).coerceIn(0f, maxY))
            "left" -> CoordinatePoint((start.x - distance).coerceIn(0f, maxX), start.y)
            "right" -> CoordinatePoint((start.x + distance).coerceIn(0f, maxX), start.y)
            else -> CoordinatePoint(start.x, (start.y - distance).coerceIn(0f, maxY))
        }
    }

    private fun resolvedCoordinateSpaceLabel(coordinateSpace: String): String {
        val normalized = coordinateSpace.trim().lowercase().replace("-", "_").replace(" ", "_")
        return when {
            normalized in NORMALIZED_COORDINATE_SPACES -> "normalized"
            normalized in PERCENT_COORDINATE_SPACES -> "percent"
            else -> "absolute_px"
        }
    }

    private fun currentAppName(): String {
        return HermesAccessibilityController.currentService()
            ?.rootInActiveWindow
            ?.packageName
            ?.toString()
            .orEmpty()
            .ifBlank { HermesAccessibilityController.currentForegroundPackageName() }
    }

    private fun JSONObject.putScreenMetrics(metrics: HermesScreenMetrics): JSONObject {
        put("screen_width", metrics.width)
        put("screen_height", metrics.height)
        put("density", metrics.density.toDouble())
        return this
    }

    private fun flattenNodes(root: AccessibilityNodeInfo, limit: Int): List<AccessibilityNodeInfo> {
        val nodes = mutableListOf<AccessibilityNodeInfo>()

        fun visit(node: AccessibilityNodeInfo?) {
            if (node == null || nodes.size >= limit) {
                return
            }
            nodes.add(node)
            for (childIndex in 0 until node.childCount) {
                visit(node.getChild(childIndex))
                if (nodes.size >= limit) {
                    return
                }
            }
        }

        visit(root)
        return nodes
    }

    private fun matchesSelector(
        node: AccessibilityNodeInfo,
        textContains: String,
        contentDescriptionContains: String,
        viewId: String,
        packageName: String,
        className: String,
    ): Boolean {
        if (textContains.isNotBlank() && !node.text?.toString().orEmpty().contains(textContains, ignoreCase = true)) {
            return false
        }
        if (contentDescriptionContains.isNotBlank() && !node.contentDescription?.toString().orEmpty().contains(contentDescriptionContains, ignoreCase = true)) {
            return false
        }
        if (viewId.isNotBlank() && !node.viewIdResourceName.orEmpty().contains(viewId, ignoreCase = true)) {
            return false
        }
        if (packageName.isNotBlank() && !node.packageName?.toString().orEmpty().contains(packageName, ignoreCase = true)) {
            return false
        }
        if (className.isNotBlank() && !node.className?.toString().orEmpty().contains(className, ignoreCase = true)) {
            return false
        }
        return textContains.isNotBlank() ||
            contentDescriptionContains.isNotBlank() ||
            viewId.isNotBlank() ||
            packageName.isNotBlank() ||
            className.isNotBlank()
    }

    private fun uiStateHash(root: AccessibilityNodeInfo, nodes: List<AccessibilityNodeInfo>): String {
        val digest = MessageDigest.getInstance("SHA-256")
        fun update(value: String) {
            digest.update(value.toByteArray(Charsets.UTF_8))
            digest.update(0.toByte())
        }
        update(root.packageName?.toString().orEmpty())
        nodes.forEachIndexed { index, node ->
            val bounds = Rect()
            node.getBoundsInScreen(bounds)
            update(index.toString())
            update(node.packageName?.toString().orEmpty())
            update(node.className?.toString().orEmpty())
            update(node.text?.toString().orEmpty())
            update(node.contentDescription?.toString().orEmpty())
            update("${bounds.left},${bounds.top},${bounds.right},${bounds.bottom}")
            update("${node.isClickable},${node.isEditable},${node.isFocused},${node.isScrollable}")
        }
        return digest.digest()
            .take(8)
            .joinToString(separator = "") { byte -> "%02x".format(byte.toInt() and 0xff) }
    }

    private fun performResolvedAction(action: String, node: AccessibilityNodeInfo, value: String): Boolean {
        return when (action.lowercase()) {
            "click" -> findSelfOrAncestor(node) { it.isClickable }?.performAction(AccessibilityNodeInfo.ACTION_CLICK) == true
            "long_click" -> findSelfOrAncestor(node) { it.isLongClickable }?.performAction(AccessibilityNodeInfo.ACTION_LONG_CLICK) == true
            "focus" -> findSelfOrAncestor(node) { it.isFocusable }?.performAction(AccessibilityNodeInfo.ACTION_FOCUS) == true
            "scroll_forward" -> findSelfOrAncestor(node) { it.isScrollable }?.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD) == true
            "scroll_backward" -> findSelfOrAncestor(node) { it.isScrollable }?.performAction(AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD) == true
            "set_text" -> {
                val editableTarget = findEditableNode(node)
                    ?: throw IOException("No editable accessibility node matched the selector")
                val arguments = Bundle().apply {
                    putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, value)
                }
                editableTarget.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)
            }
            else -> throw IOException("Unsupported accessibility action: $action")
        }
    }

    private fun findEditableNode(node: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        return findSelfOrAncestor(node) { candidate -> isEditableNode(candidate) }
    }

    private fun isEditableNode(node: AccessibilityNodeInfo): Boolean {
        return node.isEditable || node.actionList.any { actionItem -> actionItem.id == AccessibilityNodeInfo.ACTION_SET_TEXT }
    }

    private fun findSelfOrAncestor(node: AccessibilityNodeInfo, predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo? {
        var current: AccessibilityNodeInfo? = node
        while (current != null) {
            if (predicate(current)) {
                return current
            }
            current = current.parent
        }
        return null
    }

    private fun nodeJson(node: AccessibilityNodeInfo, index: Int): JSONObject {
        val bounds = Rect()
        node.getBoundsInScreen(bounds)
        return JSONObject().apply {
            put("index", index)
            put("text", node.text?.toString().orEmpty())
            put("content_description", node.contentDescription?.toString().orEmpty())
            put("view_id", node.viewIdResourceName.orEmpty())
            put("package_name", node.packageName?.toString().orEmpty())
            put("class_name", node.className?.toString().orEmpty())
            put("clickable", node.isClickable)
            put("editable", node.isEditable)
            put("scrollable", node.isScrollable)
            put("enabled", node.isEnabled)
            put("focused", node.isFocused)
            put(
                "bounds",
                JSONObject().apply {
                    put("left", bounds.left)
                    put("top", bounds.top)
                    put("right", bounds.right)
                    put("bottom", bounds.bottom)
                },
            )
        }
    }

    private data class CoordinatePoint(val x: Float, val y: Float) {
        fun toJson(): JSONObject {
            return JSONObject()
                .put("x", x.toDouble())
                .put("y", y.toDouble())
        }
    }

    private fun errorJson(message: String): String {
        return JSONObject().apply {
            put("success", false)
            put("error", message)
        }.toString()
    }
}
