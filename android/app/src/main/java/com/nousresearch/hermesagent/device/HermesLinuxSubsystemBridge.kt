package com.nousresearch.hermesagent.device

import android.content.Context
import android.content.res.AssetManager
import android.os.Build
import android.system.Os
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

object HermesLinuxSubsystemBridge {
    private const val ASSET_ROOT = "hermes-linux"
    private const val STATE_FILE_NAME = "linux-subsystem-state.json"
    private const val EXECUTION_MODE = "embedded_termux"
    private const val SYSTEM_SHELL_MODE = "android_system_shell"
    private const val SYSTEM_SHELL_PATH = "/system/bin/sh"

    private data class ShellLaunchProbe(
        val ready: Boolean,
        val detail: String = "",
    )

    fun ensureInstalled(context: Context): JSONObject {
        val androidAbi = selectAndroidAbi()
        val currentAppVersionCode = appVersionCode(context)
        val currentAssetFingerprint = assetManifestSha256(context, androidAbi)
        readState(context)?.let { state ->
            if (state.optString("android_abi") != androidAbi) {
                reset(context)
                return@let
            }
            if (state.optLong("app_version_code", -1L) != currentAppVersionCode) {
                reset(context)
                return@let
            }
            if (state.optString("asset_manifest_sha256") != currentAssetFingerprint) {
                reset(context)
                return@let
            }
            if (state.optString("execution_mode") == SYSTEM_SHELL_MODE) {
                reset(context)
                return@let
            }
            val shellPath = state.optString("shell_path", state.optString("bash_path"))
            val bashFile = File(state.optString("bash_path", shellPath))
            val prefixDirPath = state.optString("prefix_path").ifBlank {
                bashFile.parentFile?.parentFile?.absolutePath.orEmpty()
            }
            if (prefixDirPath.isNotBlank()) {
                val prefixDir = File(prefixDirPath)
                File(prefixDir, "home").mkdirs()
                File(prefixDir, "tmp").mkdirs()
                markExecutableTree(File(prefixDir, "bin"))
                markExecutableTree(File(prefixDir, "libexec"))
            }
            val homeDir = File(state.optString("home_path").ifBlank { prefixDirPath })
            if (launchShellProbe(shellPath, homeDir, buildRunEnvironment(state)).ready) {
                return state
            }
            reset(context)
        }

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
        val bashPath = File(prefixDir, "bin/bash").absolutePath
        val nativeBashPath = File(context.applicationInfo.nativeLibraryDir.orEmpty(), "libhermes_android_bash.so").absolutePath
        val nativeLlamaServerPath = File(context.applicationInfo.nativeLibraryDir.orEmpty(), "libhermes_android_llama_server.so").absolutePath
        val embeddedState = JSONObject().apply {
            put("enabled", true)
            put("app_version_code", currentAppVersionCode)
            put("asset_manifest_sha256", currentAssetFingerprint)
            put("execution_mode", EXECUTION_MODE)
            put("android_abi", androidAbi)
            put("termux_arch", manifest.optString("termux_arch"))
            put("uses_termux", true)
            put("prefix_path", prefixDir.absolutePath)
            put("shell_path", nativeBashPath)
            put("bash_path", nativeBashPath)
            put("prefix_bash_path", bashPath)
            put("native_library_dir", context.applicationInfo.nativeLibraryDir.orEmpty())
            put("native_bash_path", nativeBashPath)
            put("native_llama_server_path", nativeLlamaServerPath)
            put("bin_path", File(prefixDir, "bin").absolutePath)
            put("lib_path", File(prefixDir, "lib").absolutePath)
            put("home_path", File(prefixDir, "home").absolutePath)
            put("tmp_path", File(prefixDir, "tmp").absolutePath)
            put("root_packages", manifest.optJSONArray("root_packages"))
            put("packages", manifest.optJSONArray("packages"))
        }
        val launchProbe = launchShellProbe(nativeBashPath, File(prefixDir, "home"), buildRunEnvironment(embeddedState))
        val state = if (launchProbe.ready) {
            embeddedState
        } else {
            installRoot.deleteRecursively()
            systemShellState(
                context = context,
                androidAbi = androidAbi,
                appVersionCode = currentAppVersionCode,
                assetManifestSha256 = currentAssetFingerprint,
                fallbackReason = launchProbe.detail,
            )
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
        val rawState = stateFile.readText(Charsets.UTF_8).trim()
        if (rawState.isBlank()) {
            stateFile.delete()
            return null
        }
        return runCatching { JSONObject(rawState) }.getOrElse {
            stateFile.delete()
            null
        }
    }

    fun reset(context: Context) {
        File(context.filesDir, "hermes-home/linux").deleteRecursively()
        File(context.filesDir, "hermes-home/native-shell").deleteRecursively()
    }

    fun buildRunEnvironment(state: JSONObject): Map<String, String> {
        val prefixPath = state.optString("prefix_path")
        val binPath = state.optString("bin_path")
        val libPath = state.optString("lib_path")
        val nativeLibraryDir = state.optString("native_library_dir")
        val homePath = state.optString("home_path").ifBlank { prefixPath }
        val tmpPath = state.optString("tmp_path").ifBlank { homePath.ifBlank { prefixPath } }
        return mapOf(
            "PREFIX" to prefixPath,
            "TERMUX_PREFIX" to prefixPath,
            "PATH" to listOf("/system/bin", "/system/xbin", System.getenv("PATH").orEmpty())
                .filter { it.isNotBlank() }
                .distinct()
                .joinToString(":"),
            "LD_LIBRARY_PATH" to listOf(nativeLibraryDir, libPath, System.getenv("LD_LIBRARY_PATH").orEmpty())
                .filter { it.isNotBlank() }
                .distinct()
                .joinToString(":"),
            "HOME" to homePath,
            "TMPDIR" to tmpPath,
            "ANDROID_DATA" to "/data",
            "ANDROID_ROOT" to "/system",
            "HERMES_ANDROID_EXECUTION_MODE" to state.optString("execution_mode"),
            "TERM" to "xterm-256color",
            "LANG" to "C.UTF-8",
        )
    }

    private fun stateFile(context: Context): File {
        return File(context.filesDir, "hermes-home/linux/$STATE_FILE_NAME")
    }

    private fun systemShellState(
        context: Context,
        androidAbi: String,
        appVersionCode: Long,
        assetManifestSha256: String,
        fallbackReason: String,
    ): JSONObject {
        val nativeRoot = File(context.filesDir, "hermes-home/native-shell")
        val homeDir = File(nativeRoot, "home").apply { mkdirs() }
        val tmpDir = File(nativeRoot, "tmp").apply { mkdirs() }
        return JSONObject().apply {
            put("enabled", true)
            put("app_version_code", appVersionCode)
            put("asset_manifest_sha256", assetManifestSha256)
            put("execution_mode", SYSTEM_SHELL_MODE)
            put("android_abi", androidAbi)
            put("termux_arch", "")
            put("uses_termux", false)
            put("prefix_path", nativeRoot.absolutePath)
            put("shell_path", SYSTEM_SHELL_PATH)
            put("bash_path", "")
            put("native_library_dir", context.applicationInfo.nativeLibraryDir.orEmpty())
            put("native_bash_path", nativeExecutablePath(context, "libhermes_android_bash.so"))
            put("native_llama_server_path", nativeExecutablePath(context, "libhermes_android_llama_server.so"))
            put("bin_path", "/system/bin")
            put("lib_path", "")
            put("home_path", homeDir.absolutePath)
            put("tmp_path", tmpDir.absolutePath)
            put("root_packages", JSONArray())
            put("packages", JSONArray())
            put("fallback_reason", fallbackReason.take(1200))
        }
    }

    private fun launchShellProbe(
        shellPath: String,
        workingDirectory: File,
        environment: Map<String, String>,
    ): ShellLaunchProbe {
        if (shellPath.isBlank()) {
            return ShellLaunchProbe(false, "shell path is blank")
        }
        if (!shellPath.startsWith("/system/") && !File(shellPath).canExecute()) {
            return ShellLaunchProbe(false, "shell is not executable: $shellPath")
        }
        return runCatching {
            workingDirectory.mkdirs()
            val process = ProcessBuilder(shellPath, "-c", "exit 0")
                .directory(workingDirectory)
                .redirectErrorStream(true)
                .apply { environment().putAll(environment) }
                .start()
            if (!process.waitFor(5, TimeUnit.SECONDS)) {
                process.destroy()
                if (!process.waitFor(1, TimeUnit.SECONDS)) {
                    process.destroyForcibly()
                }
                return@runCatching ShellLaunchProbe(false, "shell launch timed out: $shellPath")
            }
            val output = BufferedReader(InputStreamReader(process.inputStream)).use { reader ->
                generateSequence { reader.readLine() }
                    .take(40)
                    .joinToString("\n")
                    .take(1200)
            }
            if (process.exitValue() == 0) {
                ShellLaunchProbe(true)
            } else {
                ShellLaunchProbe(false, "shell exited ${process.exitValue()}: $output")
            }
        }.getOrElse { error ->
            ShellLaunchProbe(false, error.message ?: error.javaClass.simpleName)
        }
    }

    private fun selectAndroidAbi(): String {
        val supportedAbis = Build.SUPPORTED_ABIS.toList()
        return supportedAbis.firstOrNull { it == "arm64-v8a" || it == "x86_64" }
            ?: supportedAbis.firstOrNull()
            ?: "arm64-v8a"
    }

    @Suppress("DEPRECATION")
    private fun appVersionCode(context: Context): Long {
        val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            packageInfo.longVersionCode
        } else {
            packageInfo.versionCode.toLong()
        }
    }

    private fun assetManifestSha256(context: Context, androidAbi: String): String {
        return runCatching {
            val payload = readAssetText(context.assets, "$ASSET_ROOT/$androidAbi/manifest.json")
            MessageDigest.getInstance("SHA-256")
                .digest(payload.toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it) }
        }.getOrDefault("")
    }

    private fun nativeExecutablePath(context: Context, name: String): String {
        val nativeLibraryDir = context.applicationInfo.nativeLibraryDir.orEmpty()
        if (nativeLibraryDir.isBlank()) {
            return ""
        }
        return File(nativeLibraryDir, name).absolutePath
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
            val linkPath = normalizeAssetRelativePath(item.optString("path"))
            val targetPath = normalizeAssetRelativePath(item.optString("target"))
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

    private fun normalizeAssetRelativePath(value: String): String {
        val parts = value
            .replace('\\', '/')
            .trim()
            .trimStart('/')
            .split('/')
            .filter { it.isNotBlank() }
        if (parts.any { it == "." || it == ".." }) {
            return ""
        }
        return parts.joinToString("/")
    }

    private fun readAssetText(assets: AssetManager, assetPath: String): String {
        return assets.open(assetPath).bufferedReader().use { it.readText() }
    }
}
