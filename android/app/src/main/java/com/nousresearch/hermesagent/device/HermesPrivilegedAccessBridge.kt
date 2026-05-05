package com.nousresearch.hermesagent.device

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import org.json.JSONArray
import org.json.JSONObject
import rikka.shizuku.Shizuku
import java.io.File

private const val SHIZUKU_PACKAGE = "moe.shizuku.privileged.api"
private const val SUI_PACKAGE = "rikka.sui"
private const val SHIZUKU_DOWNLOAD_URL = "https://shizuku.rikka.app/download/"
private const val SHIZUKU_ADB_START_COMMAND =
    "adb shell sh /sdcard/Android/data/moe.shizuku.privileged.api/start.sh"
private const val SHIZUKU_PERMISSION_REQUEST_CODE = 2401

data class HermesPrivilegedAccessStatus(
    val shizukuInstalled: Boolean,
    val suiInstalled: Boolean,
    val shizukuBinderAlive: Boolean,
    val shizukuPermissionGranted: Boolean,
    val shizukuPermissionRationaleRequired: Boolean,
    val shizukuUid: Int?,
    val shizukuPrivilegeLabel: String,
    val rootBinaryVisible: Boolean,
    val adbStartCommand: String = SHIZUKU_ADB_START_COMMAND,
    val availablePrivilegedActions: List<String> = DEFAULT_PRIVILEGED_ACTIONS,
)

data class HermesPrivilegedActionResult(
    val success: Boolean,
    val action: String,
    val message: String,
)

private val DEFAULT_PRIVILEGED_ACTIONS = listOf(
    "open_developer_options",
    "open_wireless_debugging_settings",
    "open_shizuku_app",
    "open_shizuku_download",
    "request_shizuku_permission",
)

object HermesPrivilegedAccessBridge {
    fun readStatus(context: Context): HermesPrivilegedAccessStatus {
        val appContext = context.applicationContext
        val binderAlive = runCatching { Shizuku.pingBinder() }.getOrDefault(false)
        val uid = if (binderAlive) runCatching { Shizuku.getUid() }.getOrNull() else null
        val permissionGranted = if (binderAlive && !runCatching { Shizuku.isPreV11() }.getOrDefault(true)) {
            runCatching { Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED }.getOrDefault(false)
        } else {
            false
        }
        val rationaleRequired = if (binderAlive && !permissionGranted) {
            runCatching { Shizuku.shouldShowRequestPermissionRationale() }.getOrDefault(false)
        } else {
            false
        }

        return HermesPrivilegedAccessStatus(
            shizukuInstalled = isPackageInstalled(appContext, SHIZUKU_PACKAGE),
            suiInstalled = isPackageInstalled(appContext, SUI_PACKAGE),
            shizukuBinderAlive = binderAlive,
            shizukuPermissionGranted = permissionGranted,
            shizukuPermissionRationaleRequired = rationaleRequired,
            shizukuUid = uid,
            shizukuPrivilegeLabel = when (uid) {
                0 -> "root"
                2000 -> "adb_shell"
                null -> "unavailable"
                else -> "uid_$uid"
            },
            rootBinaryVisible = listOf("/system/bin/su", "/system/xbin/su", "/sbin/su")
                .any { path -> File(path).canExecute() },
        )
    }

    fun performAction(context: Context, action: String): HermesPrivilegedActionResult {
        val appContext = context.applicationContext
        return when (action) {
            "open_developer_options" -> launchIntent(
                appContext,
                action,
                Intent(Settings.ACTION_APPLICATION_DEVELOPMENT_SETTINGS),
                "Opened Android developer options",
            )
            "open_wireless_debugging_settings" -> launchIntent(
                appContext,
                action,
                Intent("android.settings.WIRELESS_DEBUGGING_SETTINGS"),
                "Opened Android wireless debugging settings",
            )
            "open_shizuku_app" -> launchPackage(appContext, action, SHIZUKU_PACKAGE, "Opened Shizuku")
            "open_shizuku_download" -> launchIntent(
                appContext,
                action,
                Intent(Intent.ACTION_VIEW, Uri.parse(SHIZUKU_DOWNLOAD_URL)),
                "Opened Shizuku download page",
            )
            "request_shizuku_permission" -> requestShizukuPermission(action)
            else -> HermesPrivilegedActionResult(
                success = false,
                action = action,
                message = "Unsupported privileged Android action: $action",
            )
        }
    }

    fun statusToJson(status: HermesPrivilegedAccessStatus): JSONObject {
        return JSONObject().apply {
            put("shizuku_installed", status.shizukuInstalled)
            put("sui_installed", status.suiInstalled)
            put("shizuku_binder_alive", status.shizukuBinderAlive)
            put("shizuku_permission_granted", status.shizukuPermissionGranted)
            put("shizuku_permission_rationale_required", status.shizukuPermissionRationaleRequired)
            put("shizuku_uid", status.shizukuUid ?: JSONObject.NULL)
            put("shizuku_privilege_label", status.shizukuPrivilegeLabel)
            put("root_binary_visible", status.rootBinaryVisible)
            put("adb_start_command", status.adbStartCommand)
            put("available_privileged_actions", JSONArray(status.availablePrivilegedActions))
        }
    }

    fun actionToJson(result: HermesPrivilegedActionResult): JSONObject {
        return JSONObject().apply {
            put("success", result.success)
            put("action", result.action)
            put("message", result.message)
        }
    }

    private fun requestShizukuPermission(action: String): HermesPrivilegedActionResult {
        val status = readStatus(com.nousresearch.hermesagent.HermesApplication.instance.applicationContext)
        if (!status.shizukuBinderAlive) {
            return HermesPrivilegedActionResult(
                success = false,
                action = action,
                message = "Shizuku is not running. Start Shizuku with root, ADB, or Android wireless debugging first.",
            )
        }
        if (status.shizukuPermissionGranted) {
            return HermesPrivilegedActionResult(
                success = true,
                action = action,
                message = "Shizuku permission is already granted",
            )
        }
        if (status.shizukuPermissionRationaleRequired) {
            return HermesPrivilegedActionResult(
                success = false,
                action = action,
                message = "Shizuku permission was denied. Open Shizuku and allow Hermes Agent manually.",
            )
        }
        return runCatching {
            Shizuku.requestPermission(SHIZUKU_PERMISSION_REQUEST_CODE)
            HermesPrivilegedActionResult(
                success = true,
                action = action,
                message = "Requested Shizuku permission for Hermes Agent",
            )
        }.getOrElse { error ->
            HermesPrivilegedActionResult(
                success = false,
                action = action,
                message = error.message ?: error.javaClass.simpleName,
            )
        }
    }

    private fun isPackageInstalled(context: Context, packageName: String): Boolean {
        return runCatching {
            context.packageManager.getPackageInfo(packageName, 0)
            true
        }.getOrDefault(false)
    }

    private fun launchPackage(context: Context, action: String, packageName: String, successMessage: String): HermesPrivilegedActionResult {
        val intent = context.packageManager.getLaunchIntentForPackage(packageName)
            ?: return HermesPrivilegedActionResult(
                success = false,
                action = action,
                message = "$packageName is not installed",
            )
        return launchIntent(context, action, intent, successMessage)
    }

    private fun launchIntent(context: Context, action: String, intent: Intent, successMessage: String): HermesPrivilegedActionResult {
        return runCatching {
            context.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            HermesPrivilegedActionResult(success = true, action = action, message = successMessage)
        }.getOrElse { error ->
            HermesPrivilegedActionResult(
                success = false,
                action = action,
                message = error.message ?: error.javaClass.simpleName,
            )
        }
    }
}
