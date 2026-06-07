package com.mobilefork.hermesagent.device

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
    private const val RUNTIME_LAYOUT_VERSION = 6
    private const val NATIVE_EXEC_ROOT_NAME = "native-exec"
    private const val PYTHON_BINARY_NAME = "python3.13"
    private val NATIVE_EXECUTABLE_NAMES = mapOf(
        "bin/bash" to "libhermes_android_bash.so",
        "bin/llama-server" to "libhermes_android_llama_server.so",
        "bin/llama-server-bionic" to "libhermes_android_llama_server_bionic_spawn.so",
    )

    private data class ShellLaunchProbe(
        val ready: Boolean,
        val detail: String = "",
    )

    fun ensureInstalled(context: Context): JSONObject {
        val androidAbi = selectAndroidAbi()
        val currentAppVersionCode = appVersionCode(context)
        val currentAssetFingerprint = assetManifestSha256(context, androidAbi)
        val currentNativeLibraryDir = context.applicationInfo.nativeLibraryDir.orEmpty()
        readState(context)?.let { state ->
            var stateChanged = false
            if (state.optString("android_abi") != androidAbi) {
                reset(context)
                return@let
            }
            if (state.optString("native_library_dir") != currentNativeLibraryDir) {
                refreshNativeRuntimePaths(context, androidAbi, state) ?: run {
                    reset(context)
                    return@let
                }
                stateChanged = true
            }
            if (state.optInt("runtime_layout_version", 0) != RUNTIME_LAYOUT_VERSION) {
                reset(context)
                return@let
            }
            if (state.optLong("app_version_code", -1L) != currentAppVersionCode) {
                state.put("app_version_code", currentAppVersionCode)
                stateChanged = true
            }
            if (state.optString("asset_manifest_sha256") != currentAssetFingerprint) {
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
                val refreshedState = attachSandboxCatalog(state)
                if (stateChanged) {
                    stateFile(context).writeText(refreshedState.toString(), Charsets.UTF_8)
                }
                return refreshedState
            }
            reset(context)
        }

        val installRoot = File(context.filesDir, "hermes-home/linux/$androidAbi")
        val prefixDir = File(installRoot, "prefix")
        if (prefixDir.exists()) {
            prefixDir.deleteRecursively()
        }
        val state = runCatching {
            val manifest = JSONObject(readAssetText(context.assets, "$ASSET_ROOT/$androidAbi/manifest.json"))
            copyAssetFiles(context.assets, "$ASSET_ROOT/$androidAbi/prefix", prefixDir, manifest)
            File(prefixDir, "home").mkdirs()
            File(prefixDir, "tmp").mkdirs()
            markExecutableTree(File(prefixDir, "bin"))
            markExecutableTree(File(prefixDir, "libexec"))

            recreateLinks(prefixDir, manifest)
            val nativeExecRoot = recreateNativeExecutableShims(context, installRoot, prefixDir, manifest)
            val nativeBinDir = File(nativeExecRoot, "bin")
            val nativeLibexecDir = File(nativeExecRoot, "libexec")
            val prefixBashPath = File(prefixDir, "bin/bash").absolutePath
            val nativeBashPath = nativeExecutablePath(context, "libhermes_android_bash.so")
            val bashPath = nativeBashPath
                .takeIf { it.isNotBlank() && File(it).canExecute() }
                ?: prefixBashPath
            val llamaServerPath = nativeExecutablePath(context, "libhermes_android_llama_server.so")
            val bionicLlamaServerPath = nativeExecutablePath(context, "libhermes_android_llama_server_bionic_spawn.so")
            val binPath = listOf(nativeBinDir, File(prefixDir, "bin"))
                .filter { it.isDirectory }
                .joinToString(":") { it.absolutePath }
            val embeddedState = JSONObject().apply {
                put("enabled", true)
                put("runtime_layout_version", RUNTIME_LAYOUT_VERSION)
                put("app_version_code", currentAppVersionCode)
                put("asset_manifest_sha256", currentAssetFingerprint)
                put("execution_mode", EXECUTION_MODE)
                put("android_abi", androidAbi)
                put("termux_arch", manifest.optString("termux_arch"))
                put("uses_termux", true)
                put("prefix_path", prefixDir.absolutePath)
                put("shell_path", bashPath)
                put("bash_path", bashPath)
                put("prefix_bash_path", prefixBashPath)
                put("native_library_dir", context.applicationInfo.nativeLibraryDir.orEmpty())
                put("app_package_name", context.packageName)
                put("native_bash_path", nativeBashPath)
                put("native_llama_server_path", llamaServerPath)
                put("bionic_llama_server_path", bionicLlamaServerPath)
                put("native_bin_path", nativeBinDir.absolutePath)
                put("native_libexec_path", nativeLibexecDir.absolutePath)
                put("python_path", File(nativeBinDir, PYTHON_BINARY_NAME).absolutePath)
                put("bin_path", binPath.ifBlank { File(prefixDir, "bin").absolutePath })
                put("lib_path", File(prefixDir, "lib").absolutePath)
                put("home_path", File(prefixDir, "home").absolutePath)
                put("tmp_path", File(prefixDir, "tmp").absolutePath)
                put("root_packages", manifest.optJSONArray("root_packages"))
                put("packages", manifest.optJSONArray("packages"))
            }
            val launchProbe = launchShellProbe(bashPath, File(prefixDir, "home"), buildRunEnvironment(embeddedState))
            if (launchProbe.ready) {
                attachSandboxCatalog(embeddedState)
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
        }.getOrElse { exc ->
            installRoot.deleteRecursively()
            systemShellState(
                context = context,
                androidAbi = androidAbi,
                appVersionCode = currentAppVersionCode,
                assetManifestSha256 = currentAssetFingerprint,
                fallbackReason = "Embedded Linux assets unavailable: ${exc.message ?: exc::class.java.simpleName}",
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

    private fun refreshNativeRuntimePaths(context: Context, androidAbi: String, state: JSONObject): JSONObject? {
        val prefixDir = File(state.optString("prefix_path")).takeIf { it.isDirectory } ?: return null
        val installRoot = prefixDir.parentFile ?: return null
        val manifest = runCatching {
            JSONObject(readAssetText(context.assets, "$ASSET_ROOT/$androidAbi/manifest.json"))
        }.getOrNull() ?: return null
        val nativeExecRoot = recreateNativeExecutableShims(context, installRoot, prefixDir, manifest)
        val nativeBinDir = File(nativeExecRoot, "bin")
        val nativeLibexecDir = File(nativeExecRoot, "libexec")
        val prefixBashPath = File(prefixDir, "bin/bash").absolutePath
        val nativeBashPath = nativeExecutablePath(context, "libhermes_android_bash.so")
        val bashPath = nativeBashPath
            .takeIf { it.isNotBlank() && File(it).canExecute() }
            ?: prefixBashPath
        val llamaServerPath = nativeExecutablePath(context, "libhermes_android_llama_server.so")
        val bionicLlamaServerPath = nativeExecutablePath(context, "libhermes_android_llama_server_bionic_spawn.so")
        val binPath = listOf(nativeBinDir, File(prefixDir, "bin"))
            .filter { it.isDirectory }
            .joinToString(":") { it.absolutePath }

        return state
            .put("shell_path", bashPath)
            .put("bash_path", bashPath)
            .put("prefix_bash_path", prefixBashPath)
            .put("native_library_dir", context.applicationInfo.nativeLibraryDir.orEmpty())
            .put("app_package_name", context.packageName)
            .put("native_bash_path", nativeBashPath)
            .put("native_llama_server_path", llamaServerPath)
            .put("bionic_llama_server_path", bionicLlamaServerPath)
            .put("native_bin_path", nativeBinDir.absolutePath)
            .put("native_libexec_path", nativeLibexecDir.absolutePath)
            .put("python_path", File(nativeBinDir, PYTHON_BINARY_NAME).absolutePath)
            .put("bin_path", binPath.ifBlank { File(prefixDir, "bin").absolutePath })
            .put("lib_path", File(prefixDir, "lib").absolutePath)
            .put("home_path", File(prefixDir, "home").absolutePath)
            .put("tmp_path", File(prefixDir, "tmp").absolutePath)
            .put("root_packages", manifest.optJSONArray("root_packages"))
            .put("packages", manifest.optJSONArray("packages"))
    }

    fun buildRunEnvironment(state: JSONObject): Map<String, String> {
        val prefixPath = state.optString("prefix_path")
        val binPath = state.optString("bin_path")
        val libPath = state.optString("lib_path")
        val nativeLibexecPath = state.optString("native_libexec_path")
        val nativeLibraryDir = state.optString("native_library_dir")
        val appPackageName = state.optString("app_package_name").ifBlank { "com.nousresearch.hermesagent" }
        val nativeExecutableDir = state.optString("shell_path")
            .takeUnless { it.startsWith("/system/") }
            ?.let { File(it).parent.orEmpty() }
            .orEmpty()
        val homePath = state.optString("home_path").ifBlank { prefixPath }
        val tmpPath = state.optString("tmp_path").ifBlank { homePath.ifBlank { prefixPath } }
        val pythonLibPath = File(prefixPath, "lib/python3.13").absolutePath
        val prootLoaderPath = shellPathUnder(prefixPath, "libexec/proot/loader")
        val prootLoader32Path = shellPathUnder(prefixPath, "libexec/proot/loader32")
        return mapOf(
            "PREFIX" to prefixPath,
            "TERMUX_PREFIX" to prefixPath,
            "TERMUX_APP__PACKAGE_NAME" to appPackageName,
            "TERMUX_APP__APP_VERSION_NAME" to "Hermes",
            "TERMUX_VERSION" to "Hermes",
            "TERMUX__PREFIX" to prefixPath,
            "TERMUX__HOME" to homePath,
            "PATH" to listOf(binPath, "/system/bin", "/system/xbin", System.getenv("PATH").orEmpty())
                .filter { it.isNotBlank() }
                .distinct()
                .joinToString(":"),
            "LD_LIBRARY_PATH" to listOf(libPath, nativeExecutableDir, nativeLibraryDir, System.getenv("LD_LIBRARY_PATH").orEmpty())
                .filter { it.isNotBlank() }
                .distinct()
                .joinToString(":"),
            "HOME" to homePath,
            "TMPDIR" to tmpPath,
            "PROOT_TMP_DIR" to tmpPath,
            "PROOT_LOADER" to prootLoaderPath,
            "PROOT_LOADER_32" to prootLoader32Path,
            "PROOT_NO_SECCOMP" to "1",
            "ANDROID_DATA" to "/data",
            "ANDROID_ROOT" to "/system",
            "HERMES_ANDROID_EXECUTION_MODE" to state.optString("execution_mode"),
            "HERMES_ANDROID_SHELL" to SYSTEM_SHELL_PATH,
            "HERMES_ANDROID_NATIVE_SHELL" to state.optString("shell_path"),
            "HERMES_ANDROID_LINUX_BASH" to state.optString("shell_path").ifBlank { SYSTEM_SHELL_PATH },
            "HERMES_ANDROID_LINUX_NATIVE_BASH" to state.optString("shell_path"),
            "HERMES_ANDROID_LINUX_PYTHON" to state.optString("python_path").ifBlank { PYTHON_BINARY_NAME },
            "PYTHONHOME" to prefixPath,
            "PYTHONPATH" to listOf(
                pythonLibPath,
                File(pythonLibPath, "site-packages").absolutePath,
            ).joinToString(":"),
            "SSL_CERT_FILE" to File(prefixPath, "etc/tls/cert.pem").absolutePath,
            "REQUESTS_CA_BUNDLE" to File(prefixPath, "etc/tls/cert.pem").absolutePath,
            "CURL_CA_BUNDLE" to File(prefixPath, "etc/tls/cert.pem").absolutePath,
            "GIT_EXEC_PATH" to nativeLibexecPath
                .takeIf { it.isNotBlank() }
                ?.let { File(it, "git-core").absolutePath }
                .orEmpty(),
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
        val manifest = runCatching {
            JSONObject(readAssetText(context.assets, "$ASSET_ROOT/$androidAbi/manifest.json"))
        }.getOrNull()
        val nativeRoot = File(context.filesDir, "hermes-home/native-shell")
        val homeDir = File(nativeRoot, "home").apply { mkdirs() }
        val tmpDir = File(nativeRoot, "tmp").apply { mkdirs() }
        return JSONObject().apply {
            put("enabled", true)
            put("runtime_layout_version", RUNTIME_LAYOUT_VERSION)
            put("app_version_code", appVersionCode)
            put("asset_manifest_sha256", assetManifestSha256)
            put("execution_mode", SYSTEM_SHELL_MODE)
            put("android_abi", androidAbi)
            put("termux_arch", manifest?.optString("termux_arch").orEmpty().ifBlank { androidAbi })
            put("uses_termux", false)
            put("prefix_path", nativeRoot.absolutePath)
            put("shell_path", SYSTEM_SHELL_PATH)
            put("bash_path", SYSTEM_SHELL_PATH)
            put("native_library_dir", context.applicationInfo.nativeLibraryDir.orEmpty())
            put("app_package_name", context.packageName)
            put("native_bash_path", nativeExecutablePath(context, "libhermes_android_bash.so"))
            put("native_llama_server_path", nativeExecutablePath(context, "libhermes_android_llama_server.so"))
            put("bionic_llama_server_path", nativeExecutablePath(context, "libhermes_android_llama_server_bionic_spawn.so"))
            put("python_path", "")
            put("bin_path", "/system/bin")
            put("lib_path", "")
            put("home_path", homeDir.absolutePath)
            put("tmp_path", tmpDir.absolutePath)
            put("root_packages", manifest?.optJSONArray("root_packages") ?: JSONArray())
            put("packages", manifest?.optJSONArray("packages") ?: JSONArray())
            attachSandboxCatalog(this)
            put("fallback_reason", fallbackReason.take(1200))
        }
    }

    private fun attachSandboxCatalog(state: JSONObject): JSONObject {
        return state
            .put("downloadable_linux_sandboxes", HermesLinuxSandboxCatalog.distroCatalog())
            .put("recommended_linux_sandboxes", HermesLinuxSandboxCatalog.recommendedSandboxIds())
            .put("desktop_environment_catalog", HermesLinuxSandboxCatalog.desktopCatalog())
            .put("linux_sandbox_agent_summary", HermesLinuxSandboxCatalog.agentSummary())
    }

    fun commandWithEmbeddedToolAliases(state: JSONObject, command: String): String {
        if (!state.optBoolean("uses_termux", false)) {
            return command
        }
        val prefixPath = state.optString("prefix_path")
        if (prefixPath.isBlank()) {
            return command
        }
        val homePath = state.optString("home_path").ifBlank { File(prefixPath, "home").absolutePath }
        val tmpPath = state.optString("tmp_path").ifBlank { File(prefixPath, "tmp").absolutePath }
        val appPackageName = state.optString("app_package_name").ifBlank { "com.nousresearch.hermesagent" }
        val pythonPath = state.optString("python_path").ifBlank { PYTHON_BINARY_NAME }
        val prootLoaderPath = shellPathUnder(prefixPath, "libexec/proot/loader")
        val prootLoader32Path = shellPathUnder(prefixPath, "libexec/proot/loader32")
        val prootDistroScript = File(prefixPath, "bin/proot-distro").absolutePath
        val runtimeLibraryPath = listOf(
            state.optString("lib_path"),
            state.optString("native_library_dir"),
            state.optString("shell_path")
                .takeUnless { it.startsWith("/system/") }
                ?.let { File(it).parent.orEmpty() }
                .orEmpty(),
        )
            .filter { it.isNotBlank() }
            .distinct()
            .joinToString(":")
        val prelude = listOf(
            "export TERMUX_APP__PACKAGE_NAME=${shellQuote(appPackageName)}",
            "export TERMUX_APP__APP_VERSION_NAME=Hermes",
            "export TERMUX_VERSION=Hermes",
            "export TERMUX__PREFIX=${shellQuote(prefixPath)}",
            "export TERMUX__HOME=${shellQuote(homePath)}",
            "export TMPDIR=${shellQuote(tmpPath)}",
            "export PROOT_TMP_DIR=${shellQuote(tmpPath)}",
            "export PROOT_LOADER=${shellQuote(prootLoaderPath)}",
            "export PROOT_LOADER_32=${shellQuote(prootLoader32Path)}",
            "export PROOT_NO_SECCOMP=${shellQuote("1")}",
            "export LD_LIBRARY_PATH=${shellQuote(runtimeLibraryPath)}",
            "proot-distro() { case \"\${1:-}\" in login|sh|run) local _pd_cmd=\"\$1\"; shift; command ${shellQuote(pythonPath)} ${shellQuote(prootDistroScript)} \"\$_pd_cmd\" -e \"LD_LIBRARY_PATH=\$LD_LIBRARY_PATH\" -e \"PROOT_TMP_DIR=\$PROOT_TMP_DIR\" -e \"PROOT_LOADER=\$PROOT_LOADER\" -e \"PROOT_LOADER_32=\$PROOT_LOADER_32\" -e \"PROOT_NO_SECCOMP=\$PROOT_NO_SECCOMP\" \"\$@\" ;; *) command ${shellQuote(pythonPath)} ${shellQuote(prootDistroScript)} \"\$@\" ;; esac; }",
            "pd() { proot-distro \"\$@\"; }",
        ).joinToString("; ")
        return "$prelude; $command"
    }

    internal fun shellQuote(value: String): String {
        if (value.isEmpty()) {
            return "''"
        }
        return "'" + value.replace("'", "'\"'\"'") + "'"
    }

    private fun shellPathUnder(basePath: String, relativePath: String): String {
        return basePath.trimEnd('/') + "/" + relativePath.trimStart('/')
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

    private fun copyAssetFiles(assets: AssetManager, assetPath: String, destination: File, manifest: JSONObject) {
        val files = manifest.optJSONArray("files")
        if (files == null || files.length() == 0) {
            copyAssetTree(assets, assetPath, destination)
            return
        }
        destination.mkdirs()
        for (index in 0 until files.length()) {
            val relativePath = normalizeAssetRelativePath(files.optString(index))
            if (relativePath.isBlank()) {
                continue
            }
            val outputFile = File(destination, relativePath)
            outputFile.parentFile?.mkdirs()
            assets.open("$assetPath/$relativePath").use { input ->
                outputFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
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

    private fun recreateNativeExecutableShims(
        context: Context,
        installRoot: File,
        prefixDir: File,
        manifest: JSONObject,
    ): File {
        val nativeExecRoot = File(installRoot, NATIVE_EXEC_ROOT_NAME)
        if (nativeExecRoot.exists()) {
            nativeExecRoot.deleteRecursively()
        }
        nativeExecRoot.mkdirs()
        val linkMap = manifestLinkMap(manifest)
        listOf(File(prefixDir, "bin"), File(prefixDir, "libexec")).forEach { root ->
            if (!root.exists()) {
                return@forEach
            }
            root.walkTopDown()
                .filter { it.isFile }
                .forEach { file ->
                    val relativePath = file.relativeTo(prefixDir).invariantSeparatorsPath
                    createNativeExecutableShim(context, nativeExecRoot, relativePath, linkMap)
                }
        }
        linkMap.keys.forEach { relativePath ->
            createNativeExecutableShim(context, nativeExecRoot, relativePath, linkMap)
        }
        return nativeExecRoot
    }

    private fun createNativeExecutableShim(
        context: Context,
        nativeExecRoot: File,
        relativePath: String,
        linkMap: Map<String, String>,
    ) {
        val normalizedPath = normalizeAssetRelativePath(relativePath)
        if (!isNativeExecutableShimPath(normalizedPath)) {
            return
        }
        val targetRelativePath = resolveNativeExecutableTarget(normalizedPath, linkMap)
        val targetFile = File(nativeExecutablePath(context, nativeExecutableName(targetRelativePath)))
        if (!targetFile.isFile || !targetFile.canExecute()) {
            return
        }
        val shim = File(nativeExecRoot, normalizedPath)
        shim.parentFile?.mkdirs()
        if (shim.exists()) {
            shim.delete()
        }
        runCatching {
            Os.symlink(targetFile.absolutePath, shim.absolutePath)
        }
    }

    private fun manifestLinkMap(manifest: JSONObject): Map<String, String> {
        val links = manifest.optJSONArray("links") ?: return emptyMap()
        return buildMap {
            for (index in 0 until links.length()) {
                val item = links.optJSONObject(index) ?: continue
                val linkPath = normalizeAssetRelativePath(item.optString("path"))
                val targetPath = normalizeAssetRelativePath(item.optString("target"))
                if (linkPath.isBlank() || targetPath.isBlank()) {
                    continue
                }
                put(linkPath, targetPath)
            }
        }
    }

    private fun resolveNativeExecutableTarget(
        relativePath: String,
        linkMap: Map<String, String>,
    ): String {
        var current = relativePath
        repeat(12) {
            current = linkMap[current] ?: return current
        }
        return current
    }

    private fun isNativeExecutableShimPath(relativePath: String): Boolean {
        return relativePath.startsWith("bin/") || relativePath.startsWith("libexec/")
    }

    private fun nativeExecutableName(relativePath: String): String {
        NATIVE_EXECUTABLE_NAMES[relativePath]?.let { return it }
        val safeName = relativePath
            .replace('\\', '/')
            .replace(Regex("[^0-9A-Za-z_]+"), "_")
            .trim('_')
            .ifBlank { "command" }
        return "libhermes_exec_$safeName.so"
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
