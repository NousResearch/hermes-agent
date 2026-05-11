package com.nousresearch.hermesagent.device

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.content.pm.ResolveInfo
import org.json.JSONObject
import java.util.Locale

object HermesAppControlBridge {
    fun launchApp(context: Context, packageName: String, appName: String): JSONObject {
        val appContext = context.applicationContext
        val target = resolveLaunchTarget(appContext, packageName, appName)
        if (target.error.isNotBlank()) {
            return errorJson(
                exitCode = target.errorCode,
                packageName = target.packageName,
                appName = target.appName,
                message = target.error,
                candidates = target.candidates,
            )
        }
        return launchPackage(
            context = appContext,
            packageName = target.packageName,
            appName = target.appName,
            resolvedLabel = target.resolvedLabel,
        )
    }

    fun launchPackage(context: Context, packageName: String): JSONObject {
        return launchPackage(context, packageName, appName = "", resolvedLabel = "")
    }

    private fun launchPackage(context: Context, packageName: String, appName: String, resolvedLabel: String): JSONObject {
        val appContext = context.applicationContext
        val trimmedPackageName = packageName.trim()
        if (trimmedPackageName.isBlank()) {
            return errorJson(
                exitCode = 64,
                packageName = trimmedPackageName,
                appName = appName,
                message = "launch_app requires a package_name argument",
            )
        }
        if (trimmedPackageName.indexOf('\u0000') >= 0) {
            return errorJson(
                exitCode = 64,
                packageName = trimmedPackageName,
                appName = appName,
                message = "launch_app package_name must not contain NUL bytes",
            )
        }

        val intent = appContext.packageManager.getLaunchIntentForPackage(trimmedPackageName)
            ?: return errorJson(
                exitCode = 1,
                packageName = trimmedPackageName,
                appName = appName,
                message = "$trimmedPackageName is not installed or does not expose a launcher activity",
            )

        return runCatching {
            appContext.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            DeviceStateWriter.write(appContext)
            JSONObject()
                .put("success", true)
                .put("exit_code", 0)
                .put("action", "launch_app")
                .put("package_name", trimmedPackageName)
                .put("app_name", appName)
                .put("resolved_app_label", resolvedLabel)
                .put("message", "Opened $trimmedPackageName")
        }.getOrElse { error ->
            val message = when (error) {
                is ActivityNotFoundException -> "$trimmedPackageName does not expose a launchable activity"
                else -> error.message ?: error.javaClass.simpleName
            }
            errorJson(
                exitCode = 1,
                packageName = trimmedPackageName,
                appName = appName,
                message = message,
            )
        }
    }

    private fun resolveLaunchTarget(context: Context, packageName: String, appName: String): LaunchTarget {
        val trimmedPackageName = packageName.trim()
        val trimmedAppName = appName.trim()
        if (trimmedPackageName.isNotBlank()) {
            return LaunchTarget(packageName = trimmedPackageName, appName = trimmedAppName)
        }
        if (trimmedAppName.isBlank()) {
            return LaunchTarget(
                appName = trimmedAppName,
                error = "launch_app requires package_name or app_name",
                errorCode = 64,
            )
        }
        if (trimmedAppName.indexOf('\u0000') >= 0) {
            return LaunchTarget(
                appName = trimmedAppName,
                error = "launch_app app_name must not contain NUL bytes",
                errorCode = 64,
            )
        }

        val packageIntent = context.packageManager.getLaunchIntentForPackage(trimmedAppName)
        if (packageIntent != null) {
            return LaunchTarget(packageName = trimmedAppName, appName = trimmedAppName)
        }

        val launchableApps = queryLaunchableApps(context)
        val exactMatches = launchableApps.filter { app ->
            app.label.equals(trimmedAppName, ignoreCase = true) ||
                app.packageName.equals(trimmedAppName, ignoreCase = true)
        }
        if (exactMatches.size == 1) {
            val app = exactMatches.first()
            return LaunchTarget(packageName = app.packageName, appName = trimmedAppName, resolvedLabel = app.label)
        }
        if (exactMatches.size > 1) {
            return LaunchTarget(
                appName = trimmedAppName,
                error = "launch_app app_name matched multiple launcher apps; pass package_name",
                candidates = candidatesJson(exactMatches),
                errorCode = 1,
            )
        }

        val normalizedQuery = trimmedAppName.lowercase(Locale.ROOT)
        val partialMatches = launchableApps.filter { app ->
            app.label.lowercase(Locale.ROOT).contains(normalizedQuery) ||
                app.packageName.lowercase(Locale.ROOT).contains(normalizedQuery)
        }
        if (partialMatches.size == 1) {
            val app = partialMatches.first()
            return LaunchTarget(packageName = app.packageName, appName = trimmedAppName, resolvedLabel = app.label)
        }
        if (partialMatches.size > 1) {
            return LaunchTarget(
                appName = trimmedAppName,
                error = "launch_app app_name matched multiple launcher apps; pass package_name",
                candidates = candidatesJson(partialMatches),
                errorCode = 1,
            )
        }

        return LaunchTarget(
            appName = trimmedAppName,
            error = "No launchable app matched app_name: $trimmedAppName",
            candidates = candidatesJson(launchableApps.take(8)),
            errorCode = 1,
        )
    }

    private fun queryLaunchableApps(context: Context): List<LaunchableApp> {
        val packageManager = context.packageManager
        val launcherIntent = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
        @Suppress("DEPRECATION")
        val resolvedActivities: List<ResolveInfo> = packageManager.queryIntentActivities(launcherIntent, 0)
        return resolvedActivities
            .mapNotNull { info ->
                val activityInfo = info.activityInfo ?: return@mapNotNull null
                val packageName = activityInfo.packageName ?: return@mapNotNull null
                val label = info.loadLabel(packageManager)?.toString()
                    ?: activityInfo.loadLabel(packageManager)?.toString()
                    ?: packageName
                LaunchableApp(packageName = packageName, label = label)
            }
            .distinctBy { it.packageName }
            .sortedWith(compareBy({ it.label.lowercase(Locale.ROOT) }, { it.packageName }))
    }

    private fun candidatesJson(apps: List<LaunchableApp>) = org.json.JSONArray().apply {
        apps.take(8).forEach { app ->
            put(
                JSONObject()
                    .put("package_name", app.packageName)
                    .put("app_label", app.label),
            )
        }
    }

    private fun errorJson(
        exitCode: Int,
        packageName: String,
        appName: String = "",
        message: String,
        candidates: org.json.JSONArray = org.json.JSONArray(),
    ): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("exit_code", exitCode)
            .put("action", "launch_app")
            .put("package_name", packageName)
            .put("app_name", appName)
            .put("candidates", candidates)
            .put("error", message)
    }

    private data class LaunchableApp(
        val packageName: String,
        val label: String,
    )

    private data class LaunchTarget(
        val packageName: String = "",
        val appName: String = "",
        val resolvedLabel: String = "",
        val error: String = "",
        val candidates: org.json.JSONArray = org.json.JSONArray(),
        val errorCode: Int = 0,
    )
}
