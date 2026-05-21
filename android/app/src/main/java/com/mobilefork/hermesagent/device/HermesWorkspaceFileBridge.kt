package com.mobilefork.hermesagent.device

import android.content.Context
import org.json.JSONObject
import java.io.File

object HermesWorkspaceFileBridge {
    fun writeTextJson(
        context: Context,
        rawPath: String,
        content: String,
        append: Boolean = false,
    ): JSONObject {
        if (rawPath.isBlank()) {
            return errorJson("file write requires a path argument", 2)
        }
        if (rawPath.indexOf('\u0000') >= 0) {
            return errorJson("file write path must not contain NUL bytes", 2)
        }
        val resolution = resolveTarget(context, rawPath)
        resolution.error?.let { return it }
        val target = resolution.target ?: return errorJson("Unable to resolve workspace path", 2)

        target.parentFile?.mkdirs()
        if (append) {
            target.appendText(content, Charsets.UTF_8)
        } else {
            target.writeText(content, Charsets.UTF_8)
        }

        return JSONObject()
            .put("exit_code", 0)
            .put("success", true)
            .put("path", target.absolutePath)
            .put("bytes", target.length())
            .put("append", append)
            .put("cwd", resolution.homeDir?.absolutePath.orEmpty())
    }

    fun deleteJson(context: Context, rawPath: String): JSONObject {
        if (rawPath.isBlank()) {
            return errorJson("file delete requires a path argument", 2)
        }
        if (rawPath.indexOf('\u0000') >= 0) {
            return errorJson("file delete path must not contain NUL bytes", 2)
        }
        val resolution = resolveTarget(context, rawPath)
        resolution.error?.let { return it }
        val target = resolution.target ?: return errorJson("Unable to resolve workspace path", 2)
        if (target == resolution.homeDir || target == resolution.appFilesDir) {
            return errorJson("file delete refuses to remove a workspace root", 13, target.absolutePath, resolution.homeDir)
        }
        val existed = target.exists()
        val deleted = when {
            !existed -> false
            target.isDirectory -> target.deleteRecursively()
            else -> target.delete()
        }
        return JSONObject()
            .put("exit_code", if (!existed || deleted) 0 else 1)
            .put("success", !existed || deleted)
            .put("path", target.absolutePath)
            .put("existed", existed)
            .put("deleted", deleted)
            .put("cwd", resolution.homeDir?.absolutePath.orEmpty())
    }

    private fun resolveTarget(context: Context, rawPath: String): FileResolution {
        val appContext = context.applicationContext
        val state = HermesLinuxSubsystemBridge.ensureInstalled(appContext)
        val homeDir = File(state.getString("home_path")).apply { mkdirs() }.canonicalFile
        val appFilesDir = appContext.filesDir.canonicalFile
        val target = if (File(rawPath).isAbsolute) {
            File(rawPath)
        } else {
            File(homeDir, rawPath)
        }.canonicalFile
        val allowedRoots = listOf(homeDir, appFilesDir)
        if (allowedRoots.none { root -> target == root || target.path.startsWith(root.path + File.separator) }) {
            return FileResolution(
                homeDir = homeDir,
                appFilesDir = appFilesDir,
                target = target,
                error = errorJson(
                    message = "file action can only target the Hermes app workspace",
                    exitCode = 13,
                    path = target.absolutePath,
                    homeDir = homeDir,
                ),
            )
        }
        return FileResolution(homeDir = homeDir, appFilesDir = appFilesDir, target = target)
    }

    private fun errorJson(
        message: String,
        exitCode: Int,
        path: String = "",
        homeDir: File? = null,
    ): JSONObject {
        return JSONObject()
            .put("exit_code", exitCode)
            .put("success", false)
            .put("error", message)
            .apply {
                if (path.isNotBlank()) put("path", path)
                if (homeDir != null) put("cwd", homeDir.absolutePath)
            }
    }

    private data class FileResolution(
        val homeDir: File? = null,
        val appFilesDir: File? = null,
        val target: File? = null,
        val error: JSONObject? = null,
    )
}
