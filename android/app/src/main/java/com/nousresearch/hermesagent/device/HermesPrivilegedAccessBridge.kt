package com.nousresearch.hermesagent.device

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.Uri
import android.os.IBinder
import android.provider.Settings
import org.json.JSONArray
import org.json.JSONObject
import rikka.shizuku.Shizuku
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
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
    "run_privileged_shell",
    "grant_runtime_permission",
    "revoke_runtime_permission",
    "force_stop_app",
    "enable_app",
    "disable_app",
    "set_app_enabled",
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

    fun handlesStructuredAction(action: String): Boolean {
        return normalizeStructuredAction(action) != null
    }

    fun performStructuredActionJson(context: Context, action: String, arguments: JSONObject = JSONObject()): String {
        val normalizedAction = normalizeStructuredAction(action)
            ?: return JSONObject()
                .put("success", false)
                .put("exit_code", 2)
                .put("action", action)
                .put("error", "Unsupported Shizuku app-management action: $action")
                .put("available_privileged_actions", JSONArray(DEFAULT_PRIVILEGED_ACTIONS))
                .toString()
        val appContext = context.applicationContext
        val packageName = stringArgument(arguments, "package_name", "packageName", "package", "app_package", "application_id")
            ?.trim()
            ?: return structuredError(normalizedAction, "requires package_name")
        if (!ANDROID_PACKAGE_OR_PERMISSION_REGEX.matches(packageName)) {
            return structuredError(normalizedAction, "package_name is not a valid Android package name", packageName = packageName)
        }
        if (normalizedAction in SELF_PROTECTING_ACTIONS && packageName == appContext.packageName) {
            return structuredError(normalizedAction, "Hermes will not disable or force-stop itself", packageName = packageName)
        }

        val permission = if (normalizedAction in PERMISSION_ACTIONS) {
            val value = stringArgument(arguments, "permission", "permission_name", "permissionName", "android_permission")
                ?.trim()
                ?: return structuredError(normalizedAction, "requires permission", packageName = packageName)
            if (!ANDROID_PACKAGE_OR_PERMISSION_REGEX.matches(value)) {
                return structuredError(
                    normalizedAction,
                    "permission is not a valid Android permission name",
                    packageName = packageName,
                    permission = value,
                )
            }
            value
        } else {
            null
        }

        val command = when (normalizedAction) {
            "grant_runtime_permission" -> "pm grant $packageName $permission"
            "revoke_runtime_permission" -> "pm revoke $packageName $permission"
            "force_stop_app" -> "am force-stop $packageName"
            "enable_app" -> "pm enable $packageName"
            "disable_app" -> "pm disable-user $packageName"
            "set_app_enabled" -> if (enabledArgument(arguments)) {
                "pm enable $packageName"
            } else {
                "pm disable-user $packageName"
            }
            else -> return structuredError(normalizedAction, "unsupported action after normalization", packageName = packageName)
        }
        val shellResult = JSONObject(runShellCommandJson(appContext, command, arguments.optInt("timeout_seconds", DEFAULT_SHELL_TIMEOUT_SECONDS)))
        return shellResult
            .put("action", normalizedAction)
            .put("package_name", packageName)
            .put("permission", permission ?: JSONObject.NULL)
            .put("adb_shell_command", command)
            .toString()
    }

    fun runShellCommandJson(context: Context, command: String, timeoutSeconds: Int = DEFAULT_SHELL_TIMEOUT_SECONDS): String {
        val status = readStatus(context)
        if (!status.shizukuBinderAlive) {
            return privilegedShellUnavailable(
                "Shizuku is not running. Start Shizuku with root, ADB, or Android wireless debugging first.",
                status,
            )
        }
        if (!status.shizukuPermissionGranted) {
            return privilegedShellUnavailable(
                "Shizuku permission is not granted to Hermes Agent.",
                status,
            )
        }
        if (command.trim().isBlank()) {
            return JSONObject()
                .put("success", false)
                .put("exit_code", 2)
                .put("error", "run_privileged_shell requires a command argument")
                .put("shizuku_privilege_label", status.shizukuPrivilegeLabel)
                .toString()
        }
        if (command.indexOf('\u0000') >= 0) {
            return JSONObject()
                .put("success", false)
                .put("exit_code", 2)
                .put("error", "run_privileged_shell command must not contain NUL bytes")
                .put("shizuku_privilege_label", status.shizukuPrivilegeLabel)
                .toString()
        }

        val appContext = context.applicationContext
        val componentName = ComponentName(appContext, HermesPrivilegedShellUserService::class.java)
        val args = Shizuku.UserServiceArgs(componentName)
            .tag("hermes_privileged_shell")
            .processNameSuffix("privileged_shell")
            .version(1)
            .debuggable((appContext.applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0)
            .daemon(false)
        val serviceRef = AtomicReference<IHermesPrivilegedShellService?>()
        val latch = CountDownLatch(1)
        val connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, service: IBinder) {
                serviceRef.set(IHermesPrivilegedShellService.Stub.asInterface(service))
                latch.countDown()
            }

            override fun onServiceDisconnected(name: ComponentName) {
                serviceRef.set(null)
            }
        }

        return runCatching {
            Shizuku.bindUserService(args, connection)
            if (!latch.await(SERVICE_CONNECT_TIMEOUT_SECONDS.toLong(), TimeUnit.SECONDS)) {
                runCatching { Shizuku.unbindUserService(args, connection, true) }
                return JSONObject()
                    .put("success", false)
                    .put("exit_code", 124)
                    .put("error", "Timed out while connecting to Hermes Shizuku user service")
                    .put("shizuku_privilege_label", status.shizukuPrivilegeLabel)
                    .toString()
            }
            val service = serviceRef.get()
                ?: return JSONObject()
                    .put("success", false)
                    .put("exit_code", -1)
                    .put("error", "Hermes Shizuku user service disconnected before command execution")
                    .put("shizuku_privilege_label", status.shizukuPrivilegeLabel)
                    .toString()
            val result = JSONObject(service.runCommand(command, timeoutSeconds))
                .put("shizuku_privilege_label", status.shizukuPrivilegeLabel)
            runCatching { Shizuku.unbindUserService(args, connection, true) }
            result.toString()
        }.getOrElse { error ->
            runCatching { Shizuku.unbindUserService(args, connection, true) }
            JSONObject()
                .put("success", false)
                .put("exit_code", -1)
                .put("error", error.message ?: error.javaClass.simpleName)
                .put("shizuku_privilege_label", status.shizukuPrivilegeLabel)
                .toString()
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

    private fun privilegedShellUnavailable(message: String, status: HermesPrivilegedAccessStatus): String {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 13)
            .put("error", message)
            .put("shizuku_binder_alive", status.shizukuBinderAlive)
            .put("shizuku_permission_granted", status.shizukuPermissionGranted)
            .put("shizuku_privilege_label", status.shizukuPrivilegeLabel)
            .put("available_privileged_actions", JSONArray(status.availablePrivilegedActions))
            .toString()
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

    private fun stringArgument(arguments: JSONObject, vararg keys: String): String? {
        return keys.firstNotNullOfOrNull { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                null
            } else {
                arguments.optString(key).takeIf { it.isNotBlank() }
            }
        }
    }

    private fun enabledArgument(arguments: JSONObject): Boolean {
        if (arguments.has("enabled") && !arguments.isNull("enabled")) {
            return arguments.optBoolean("enabled", true)
        }
        val state = stringArgument(arguments, "state", "enabled_state")?.lowercase().orEmpty()
        return state !in setOf("false", "off", "disabled", "disable", "0")
    }

    private fun normalizeStructuredAction(action: String): String? {
        val normalized = action.trim().lowercase().replace("-", "_").replace(" ", "_")
        return STRUCTURED_ACTION_SYNONYMS[normalized] ?: normalized.takeIf { it in STRUCTURED_PRIVILEGED_ACTIONS }
    }

    private fun structuredError(
        action: String,
        message: String,
        packageName: String? = null,
        permission: String? = null,
    ): String {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 2)
            .put("action", action)
            .put("error", message)
            .put("package_name", packageName ?: JSONObject.NULL)
            .put("permission", permission ?: JSONObject.NULL)
            .put("available_privileged_actions", JSONArray(DEFAULT_PRIVILEGED_ACTIONS))
            .toString()
    }

    private val STRUCTURED_PRIVILEGED_ACTIONS = setOf(
        "grant_runtime_permission",
        "revoke_runtime_permission",
        "force_stop_app",
        "enable_app",
        "disable_app",
        "set_app_enabled",
    )
    private val STRUCTURED_ACTION_SYNONYMS = mapOf(
        "grant_permission" to "grant_runtime_permission",
        "pm_grant" to "grant_runtime_permission",
        "revoke_permission" to "revoke_runtime_permission",
        "pm_revoke" to "revoke_runtime_permission",
        "force_stop_package" to "force_stop_app",
        "kill_app" to "force_stop_app",
        "enable_package" to "enable_app",
        "pm_enable" to "enable_app",
        "disable_package" to "disable_app",
        "disable_user_app" to "disable_app",
        "pm_disable" to "disable_app",
    )
    private val PERMISSION_ACTIONS = setOf("grant_runtime_permission", "revoke_runtime_permission")
    private val SELF_PROTECTING_ACTIONS = setOf("force_stop_app", "disable_app", "set_app_enabled")
    private val ANDROID_PACKAGE_OR_PERMISSION_REGEX = Regex("[A-Za-z][A-Za-z0-9_]*(\\.[A-Za-z0-9_]+)*")
}

private const val DEFAULT_SHELL_TIMEOUT_SECONDS = 30
private const val SERVICE_CONNECT_TIMEOUT_SECONDS = 10
