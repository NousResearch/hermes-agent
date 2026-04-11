package com.nousresearch.hermesagent.device

import android.content.Context
import android.content.res.AssetManager
import android.os.Build
import android.system.Os
import org.json.JSONObject
import java.io.File

object HermesLinuxSubsystemBridge {
    private const val ASSET_ROOT = "hermes-linux"
    private const val STATE_FILE_NAME = "linux-subsystem-state.json"

    fun ensureInstalled(context: Context): JSONObject {
        readState(context)?.let { state ->
            val bashPath = state.optString("bash_path")
            if (bashPath.isNotBlank() && File(bashPath).isFile) {
                return state
            }
        }

        val androidAbi = selectAndroidAbi()
        val installRoot = File(context.filesDir, "hermes-home/linux/$androidAbi")
        val prefixDir = File(installRoot, "prefix")
        if (prefixDir.exists()) {
            prefixDir.deleteRecursively()
        }
        copyAssetTree(context.assets, "$ASSET_ROOT/$androidAbi/prefix", prefixDir)
        File(prefixDir, "home").mkdirs()
        File(prefixDir, "tmp").mkdirs()
        markExecutableTree(File(prefixDir, "bin"))
        markExecutableTree(File(prefixDir, "libexec"))

        val manifest = JSONObject(readAssetText(context.assets, "$ASSET_ROOT/$androidAbi/manifest.json"))
        recreateLinks(prefixDir, manifest)
        val state = JSONObject().apply {
            put("enabled", true)
            put("android_abi", androidAbi)
            put("termux_arch", manifest.optString("termux_arch"))
            put("prefix_path", prefixDir.absolutePath)
            put("bash_path", File(prefixDir, "bin/bash").absolutePath)
            put("bin_path", File(prefixDir, "bin").absolutePath)
            put("lib_path", File(prefixDir, "lib").absolutePath)
            put("home_path", File(prefixDir, "home").absolutePath)
            put("tmp_path", File(prefixDir, "tmp").absolutePath)
            put("root_packages", manifest.optJSONArray("root_packages"))
            put("packages", manifest.optJSONArray("packages"))
        }
        stateFile(context).apply {
            parentFile?.mkdirs()
            writeText(state.toString(), Charsets.UTF_8)
        }
        return state
    }

    fun readState(context: Context): JSONObject? {
        val stateFile = stateFile(context)
        if (!stateFile.isFile) {
            return null
        }
        return JSONObject(stateFile.readText(Charsets.UTF_8))
    }

    fun reset(context: Context) {
        File(context.filesDir, "hermes-home/linux").deleteRecursively()
    }

    private fun stateFile(context: Context): File {
        return File(context.filesDir, "hermes-home/linux/$STATE_FILE_NAME")
    }

    private fun selectAndroidAbi(): String {
        return when {
            Build.SUPPORTED_ABIS.contains("arm64-v8a") -> "arm64-v8a"
            Build.SUPPORTED_ABIS.contains("x86_64") -> "x86_64"
            else -> Build.SUPPORTED_ABIS.firstOrNull() ?: "arm64-v8a"
        }
    }

    private fun copyAssetTree(assets: AssetManager, assetPath: String, destination: File) {
        val children = assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            destination.parentFile?.mkdirs()
            assets.open(assetPath).use { input ->
                destination.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            return
        }
        destination.mkdirs()
        for (child in children) {
            copyAssetTree(assets, "$assetPath/$child", File(destination, child))
        }
    }

    private fun markExecutableTree(root: File) {
        if (!root.exists()) {
            return
        }
        root.walkTopDown().forEach { file ->
            if (file.isFile) {
                file.setExecutable(true, false)
            }
        }
    }

    private fun recreateLinks(prefixDir: File, manifest: JSONObject) {
        val links = manifest.optJSONArray("links") ?: return
        for (index in 0 until links.length()) {
            val item = links.optJSONObject(index) ?: continue
            val linkPath = item.optString("path")
            val targetPath = item.optString("target")
            if (linkPath.isBlank() || targetPath.isBlank()) {
                continue
            }
            val linkFile = File(prefixDir, linkPath)
            val targetFile = File(prefixDir, targetPath)
            if (!targetFile.exists()) {
                continue
            }
            linkFile.parentFile?.mkdirs()
            if (linkFile.exists()) {
                continue
            }
            runCatching {
                Os.symlink(targetFile.absolutePath, linkFile.absolutePath)
            }.onFailure {
                linkFile.writeBytes(targetFile.readBytes())
                linkFile.setExecutable(targetFile.canExecute(), false)
            }
        }
    }

    private fun readAssetText(assets: AssetManager, assetPath: String): String {
        return assets.open(assetPath).bufferedReader().use { it.readText() }
    }
}
