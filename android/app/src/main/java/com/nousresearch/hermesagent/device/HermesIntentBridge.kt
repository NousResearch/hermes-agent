package com.nousresearch.hermesagent.device

import android.content.ActivityNotFoundException
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Browser
import androidx.core.content.FileProvider
import com.nousresearch.hermesagent.backend.OnDeviceBackendManager
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

object HermesIntentBridge {
    fun performIntentJson(context: Context, payload: JSONObject): JSONObject {
        val intentTaskAction = normalizeIntentTaskAction(payload.optString("intent_task_action"))
            ?: return errorJson(
                exitCode = 64,
                action = payload.optString("intent_task_action"),
                message = "Unsupported Android intent task action. Use start_activity, open_uri, or send_broadcast",
            )
        val validationError = validatePayload(payload)
        if (validationError != null) {
            return errorJson(exitCode = 64, action = intentTaskAction, message = validationError)
        }
        if (payload.optBoolean("__validate_only", false)) {
            return successJson(intentTaskAction, payload, "Android intent payload is valid")
        }

        val appContext = context.applicationContext
        val builtIntent = buildIntent(appContext, intentTaskAction, payload)
        val intent = builtIntent.intent
        return when (intentTaskAction) {
            INTENT_TASK_SEND_BROADCAST -> runCatching {
                appContext.sendBroadcast(intent)
                DeviceStateWriter.write(appContext)
                successJson(intentTaskAction, payload, "Sent Android broadcast intent")
            }.getOrElse { error ->
                errorJson(exitCode = 1, action = intentTaskAction, message = error.message ?: error.javaClass.simpleName)
            }
            else -> runCatching {
                val localBackendRelease = releaseLocalBackendForExternalActivity(appContext, intentTaskAction)
                appContext.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                DeviceStateWriter.write(appContext)
                successJson(intentTaskAction, payload, "Started Android intent").also { result ->
                    if (intentTaskAction == INTENT_TASK_OPEN_URI) {
                        result.put("external_activity_handoff", true)
                        builtIntent.resolvedOpenUri?.let { resolved ->
                            result.put("resolved_uri", resolved.uri.toString())
                            result.put("resolved_mime_type", resolved.mimeType.orEmpty())
                            result.put("resolved_with_file_provider", resolved.grantReadPermission)
                            result.put("preferred_browser_package", intent.getPackage().orEmpty())
                        }
                        localBackendRelease?.let { result.put("local_backend_release", it) }
                    }
                }
            }.getOrElse { error ->
                val message = when (error) {
                    is ActivityNotFoundException -> "No activity can handle this Android intent"
                    else -> error.message ?: error.javaClass.simpleName
                }
                errorJson(exitCode = 1, action = intentTaskAction, message = message)
            }
        }
    }

    private fun releaseLocalBackendForExternalActivity(
        context: Context,
        intentTaskAction: String,
    ): JSONObject? {
        if (intentTaskAction != INTENT_TASK_OPEN_URI) {
            return null
        }
        val priorStatus = OnDeviceBackendManager.currentStatus()
        OnDeviceBackendManager.stopAll()
        DeviceStateWriter.write(context)
        return JSONObject()
            .put("released", priorStatus.started)
            .put("backend", priorStatus.backendKind.persistedValue)
            .put("model", priorStatus.modelName)
            .put("base_url", priorStatus.baseUrl)
    }

    fun normalizeIntentTaskAction(action: String): String? {
        val normalized = action.trim().lowercase().replace("-", "_").replace(" ", "_")
        return INTENT_TASK_ACTION_SYNONYMS[normalized] ?: normalized.takeIf { it in INTENT_TASK_ACTIONS }
    }

    private fun buildIntent(context: Context, intentTaskAction: String, payload: JSONObject): BuiltIntent {
        val intent = Intent()
        val intentAction = payload.optString("intent_action").ifBlank {
            if (intentTaskAction == INTENT_TASK_OPEN_URI) Intent.ACTION_VIEW else ""
        }
        if (intentAction.isNotBlank()) {
            intent.action = intentAction
        }
        var resolvedOpenUri: ResolvedOpenUri? = null
        payload.optString("data_uri").takeIf { it.isNotBlank() }?.let { rawUri ->
            val resolved = resolveOpenUri(context, intentTaskAction, rawUri)
            resolvedOpenUri = resolved
            if (resolved.mimeType.isNullOrBlank()) {
                intent.data = resolved.uri
            } else {
                intent.setDataAndType(resolved.uri, resolved.mimeType)
            }
            if (resolved.grantReadPermission) {
                intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        }
        if (intentTaskAction == INTENT_TASK_OPEN_URI) {
            intent.putExtra(Browser.EXTRA_APPLICATION_ID, context.packageName)
        }
        if (intentTaskAction == INTENT_TASK_OPEN_URI && shouldAddBrowsableCategory(resolvedOpenUri?.uri ?: intent.data)) {
            intent.addCategory(Intent.CATEGORY_BROWSABLE)
        }
        val explicitPackageName = payload.optString("package_name").takeIf { it.isNotBlank() }
        explicitPackageName?.let { packageName ->
            intent.setPackage(packageName)
        }
        if (explicitPackageName == null && resolvedOpenUri?.preferBrowserPackage == true) {
            preferredBrowserPackage(context)?.let { intent.setPackage(it) }
        }
        resolveComponent(payload)?.let { component ->
            intent.component = component
        }
        payload.optString("category").takeIf { it.isNotBlank() }?.let { intent.addCategory(it) }
        payload.optJSONArray("categories")?.let { categories ->
            for (index in 0 until categories.length()) {
                categories.optString(index).takeIf { it.isNotBlank() }?.let { intent.addCategory(it) }
            }
        }
        payload.optJSONObject("extras")?.let { extras ->
            extras.keys().forEach { key ->
                when (val value = extras.opt(key)) {
                    is Boolean -> intent.putExtra(key, value)
                    is Int -> intent.putExtra(key, value)
                    is Long -> intent.putExtra(key, value)
                    is Double -> intent.putExtra(key, value)
                    is Number -> intent.putExtra(key, value.toLong())
                    is String -> intent.putExtra(key, value)
                }
            }
        }
        return BuiltIntent(intent, resolvedOpenUri)
    }

    private fun resolveOpenUri(context: Context, intentTaskAction: String, rawUri: String): ResolvedOpenUri {
        if (intentTaskAction != INTENT_TASK_OPEN_URI) {
            return ResolvedOpenUri(Uri.parse(rawUri))
        }
        val parsed = Uri.parse(rawUri)
        val localFile = localHermesFile(context, rawUri, parsed) ?: return ResolvedOpenUri(parsed)
        val contentUri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.files",
            localFile,
        )
        return ResolvedOpenUri(
            uri = contentUri,
            mimeType = mimeTypeFor(localFile),
            grantReadPermission = true,
            preferBrowserPackage = localFile.extension.equals("html", ignoreCase = true) ||
                localFile.extension.equals("htm", ignoreCase = true),
        )
    }

    private fun localHermesFile(context: Context, rawUri: String, parsed: Uri): File? {
        val candidate = when {
            parsed.scheme == "file" -> parsed.path?.let(::File)
            parsed.scheme.isNullOrBlank() && File(rawUri).isAbsolute -> File(rawUri)
            parsed.scheme.isNullOrBlank() -> {
                val state = HermesLinuxSubsystemBridge.ensureInstalled(context)
                val homePath = state.optString("home_path").ifBlank {
                    File(context.filesDir, "hermes-home/workspace").absolutePath
                }
                File(homePath, rawUri)
            }
            else -> null
        } ?: return null
        val canonical = candidate.canonicalFile
        val hermesRoot = File(context.filesDir, "hermes-home").canonicalFile
        return canonical.takeIf { file ->
            (file == hermesRoot || file.path.startsWith(hermesRoot.path + File.separator)) && file.isFile
        }
    }

    private fun mimeTypeFor(file: File): String {
        return when (file.extension.lowercase()) {
            "html", "htm" -> "text/html"
            "txt", "log", "md" -> "text/plain"
            "json" -> "application/json"
            "js" -> "application/javascript"
            "css" -> "text/css"
            "svg" -> "image/svg+xml"
            "png" -> "image/png"
            "jpg", "jpeg" -> "image/jpeg"
            "gif" -> "image/gif"
            else -> "*/*"
        }
    }

    private fun shouldAddBrowsableCategory(uri: Uri?): Boolean {
        return uri?.scheme?.lowercase() in BROWSABLE_URI_SCHEMES
    }

    private fun preferredBrowserPackage(context: Context): String? {
        val browserIntent = Intent(Intent.ACTION_VIEW, Uri.parse("https://example.com")).apply {
            addCategory(Intent.CATEGORY_BROWSABLE)
        }
        val packageManager = context.packageManager
        val resolved = packageManager.resolveActivity(browserIntent, PackageManager.MATCH_DEFAULT_ONLY)
            ?.activityInfo
            ?.packageName
            .orEmpty()

        val candidates = packageManager
            .queryIntentActivities(browserIntent, PackageManager.MATCH_DEFAULT_ONLY)
            .mapNotNull { it.activityInfo?.packageName }
            .distinct()
        return selectPreferredBrowserPackage(resolved, candidates)
    }

    internal fun selectPreferredBrowserPackage(resolvedPackage: String, candidatePackages: List<String>): String? {
        val resolved = resolvedPackage.takeUnless { it == "android" || it == "com.android.intentresolver" }
        if (!resolved.isNullOrBlank()) {
            return resolved
        }
        val preferredPackages = listOf(
            "com.android.chrome",
            "com.chrome.beta",
            "com.chrome.dev",
            "org.mozilla.firefox",
            "org.mozilla.firefox_beta",
            "com.brave.browser",
            "com.microsoft.emmx",
        )
        return preferredPackages.firstOrNull { it in candidatePackages } ?: candidatePackages.firstOrNull()
    }

    private fun validatePayload(payload: JSONObject): String? {
        val intentTaskAction = normalizeIntentTaskAction(payload.optString("intent_task_action"))
            ?: return "intent_task_action is required"
        for (key in STRING_KEYS) {
            val value = payload.optString(key)
            if (value.indexOf('\u0000') >= 0) {
                return "$key must not contain NUL bytes"
            }
            if (value.length > MAX_STRING_CHARS) {
                return "$key is too long"
            }
        }
        val categories = payload.optJSONArray("categories")
        if (categories != null) {
            if (categories.length() > MAX_CATEGORIES) {
                return "categories contains too many entries"
            }
            for (index in 0 until categories.length()) {
                val value = categories.optString(index)
                if (value.indexOf('\u0000') >= 0) {
                    return "categories must not contain NUL bytes"
                }
                if (value.length > MAX_STRING_CHARS) {
                    return "category is too long"
                }
            }
        }
        validateExtras(payload.optJSONObject("extras"))?.let { return it }
        if (payload.optString("component").isNotBlank() && resolveComponent(payload) == null) {
            return "component must use Android flattened component syntax such as com.example/.MainActivity"
        }
        if (payload.optString("class_name").isNotBlank() && payload.optString("package_name").isBlank()) {
            return "class_name intent tasks require package_name"
        }
        return when (intentTaskAction) {
            INTENT_TASK_OPEN_URI -> if (payload.optString("data_uri").isBlank()) {
                "open_uri intent tasks require data_uri"
            } else {
                null
            }
            INTENT_TASK_SEND_BROADCAST -> if (payload.optString("intent_action").isBlank()) {
                "send_broadcast intent tasks require intent_action"
            } else {
                null
            }
            INTENT_TASK_START_ACTIVITY -> if (
                payload.optString("intent_action").isBlank() &&
                payload.optString("data_uri").isBlank() &&
                payload.optString("component").isBlank() &&
                payload.optString("class_name").isBlank()
            ) {
                "start_activity intent tasks require intent_action, data_uri, component, or class_name"
            } else {
                null
            }
            else -> "Unsupported Android intent task action"
        }
    }

    private fun validateExtras(extras: JSONObject?): String? {
        if (extras == null) {
            return null
        }
        if (extras.length() > MAX_EXTRAS) {
            return "extras contains too many entries"
        }
        extras.keys().forEach { key ->
            if (key.isBlank() || key.indexOf('\u0000') >= 0) {
                return "extra keys must be non-empty strings without NUL bytes"
            }
            if (key.length > MAX_EXTRA_KEY_CHARS) {
                return "extra key is too long"
            }
            when (val value = extras.opt(key)) {
                null, JSONObject.NULL -> Unit
                is Boolean, is Number -> Unit
                is String -> {
                    if (value.indexOf('\u0000') >= 0) {
                        return "extra values must not contain NUL bytes"
                    }
                    if (value.length > MAX_EXTRA_VALUE_CHARS) {
                        return "extra value is too long"
                    }
                }
                else -> return "extras only support string, number, and boolean values"
            }
        }
        return null
    }

    private fun resolveComponent(payload: JSONObject): ComponentName? {
        val component = payload.optString("component").trim()
        if (component.isNotBlank()) {
            return ComponentName.unflattenFromString(component)
        }
        val packageName = payload.optString("package_name").trim()
        val className = payload.optString("class_name").trim()
        if (packageName.isBlank() || className.isBlank()) {
            return null
        }
        val resolvedClassName = if (className.startsWith(".")) "$packageName$className" else className
        return ComponentName(packageName, resolvedClassName)
    }

    private fun successJson(intentTaskAction: String, payload: JSONObject, message: String): JSONObject {
        return baseJson(intentTaskAction, payload)
            .put("success", true)
            .put("exit_code", 0)
            .put("message", message)
    }

    private fun errorJson(exitCode: Int, action: String, message: String): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("exit_code", exitCode)
            .put("action", action)
            .put("error", message)
            .put("available_intent_task_actions", JSONArray(INTENT_TASK_ACTIONS))
    }

    private fun baseJson(intentTaskAction: String, payload: JSONObject): JSONObject {
        val component = resolveComponent(payload)?.flattenToShortString().orEmpty()
        return JSONObject()
            .put("action", intentTaskAction)
            .put("intent_action", payload.optString("intent_action"))
            .put("data_uri", payload.optString("data_uri"))
            .put("package_name", payload.optString("package_name"))
            .put("class_name", payload.optString("class_name"))
            .put("component", component.ifBlank { payload.optString("component") })
            .put("category", payload.optString("category"))
            .put("categories_count", payload.optJSONArray("categories")?.length() ?: 0)
            .put("extras_count", payload.optJSONObject("extras")?.length() ?: 0)
            .put("extras", payload.optJSONObject("extras") ?: JSONObject())
    }

    private data class ResolvedOpenUri(
        val uri: Uri,
        val mimeType: String? = null,
        val grantReadPermission: Boolean = false,
        val preferBrowserPackage: Boolean = false,
    )

    private data class BuiltIntent(
        val intent: Intent,
        val resolvedOpenUri: ResolvedOpenUri?,
    )

    private const val INTENT_TASK_START_ACTIVITY = "start_activity"
    private const val INTENT_TASK_OPEN_URI = "open_uri"
    private const val INTENT_TASK_SEND_BROADCAST = "send_broadcast"
    private val BROWSABLE_URI_SCHEMES = setOf("http", "https")
    private val INTENT_TASK_ACTIONS = listOf(
        INTENT_TASK_START_ACTIVITY,
        INTENT_TASK_OPEN_URI,
        INTENT_TASK_SEND_BROADCAST,
    )
    private val INTENT_TASK_ACTION_SYNONYMS = mapOf(
        "activity" to INTENT_TASK_START_ACTIVITY,
        "launch_activity" to INTENT_TASK_START_ACTIVITY,
        "start" to INTENT_TASK_START_ACTIVITY,
        "view" to INTENT_TASK_OPEN_URI,
        "view_uri" to INTENT_TASK_OPEN_URI,
        "open_url" to INTENT_TASK_OPEN_URI,
        "url" to INTENT_TASK_OPEN_URI,
        "broadcast" to INTENT_TASK_SEND_BROADCAST,
        "send" to INTENT_TASK_SEND_BROADCAST,
    )
    private val STRING_KEYS = listOf(
        "intent_task_action",
        "intent_action",
        "data_uri",
        "package_name",
        "class_name",
        "component",
        "category",
    )
    private const val MAX_STRING_CHARS = 2_000
    private const val MAX_CATEGORIES = 8
    private const val MAX_EXTRAS = 32
    private const val MAX_EXTRA_KEY_CHARS = 128
    private const val MAX_EXTRA_VALUE_CHARS = 2_000
}
