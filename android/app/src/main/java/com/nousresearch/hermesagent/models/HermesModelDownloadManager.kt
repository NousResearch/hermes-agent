package com.nousresearch.hermesagent.models

import android.app.ActivityManager
import android.app.DownloadManager
import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Environment
import android.text.format.Formatter
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import java.util.UUID

data class ModelDownloadInspection(
    val title: String,
    val sourceUrl: String,
    val destinationFileName: String,
    val totalBytes: Long,
    val totalBytesLabel: String,
    val deviceRamBytes: Long,
    val deviceRamLabel: String,
    val ramWarning: String,
    val supportsResume: Boolean,
    val abiSummary: String,
)

data class ModelDownloadDraft(
    val repoOrUrl: String,
    val filePath: String,
    val revision: String,
    val runtimeFlavor: String,
)

object HermesModelDownloadManager {
    private const val HUGGING_FACE_BASE = "https://huggingface.co"
    private const val HUGGING_FACE_API = "$HUGGING_FACE_BASE/api/models/"
    private val LITERT_ALIAS_REPOS = mapOf(
        "google/gemma-4-e2b" to "litert-community/gemma-4-E2B-it-litert-lm",
        "google/gemma-4-e2b-it" to "litert-community/gemma-4-E2B-it-litert-lm",
        "google/gemma-4-e4b" to "litert-community/gemma-4-E4B-it-litert-lm",
        "google/gemma-4-e4b-it" to "litert-community/gemma-4-E4B-it-litert-lm",
        "qwen/qwen2.5-1.5b-instruct" to "litert-community/Qwen2.5-1.5B-Instruct",
        "deepseek-ai/deepseek-r1-distill-qwen-1.5b" to "litert-community/DeepSeek-R1-Distill-Qwen-1.5B",
        "microsoft/phi-4-mini-instruct" to "litert-community/Phi-4-mini-instruct",
    )
    private val GGUF_RECOMMENDED_REPOS = mapOf(
        "qwen/qwen2.5-1.5b-instruct" to "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "deepseek-ai/deepseek-r1-distill-qwen-1.5b" to "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "microsoft/phi-4-mini-instruct" to "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "nvidia/llama-3.1-nemotron-nano-8b-v1" to "bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF",
        "nvidia/nemotron-cascade-8b" to "bartowski/nvidia_Nemotron-Cascade-8B-GGUF",
        "nvidia/nemotron-cascade-8b-thinking" to "bartowski/nvidia_Nemotron-Cascade-8B-Thinking-GGUF",
    )

    private data class NetworkPolicySnapshot(
        val label: String,
        val isMetered: Boolean,
        val isRoaming: Boolean,
    )

    fun modelsDirectory(context: Context): File {
        return (context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
            ?: File(context.filesDir, "downloads")).resolve("models").apply { mkdirs() }
    }

    fun inspectCandidate(context: Context, draft: ModelDownloadDraft, hfToken: String): ModelDownloadInspection {
        val resolvedUrl = resolveDownloadUrl(draft, hfToken)
        val head = headProbe(resolvedUrl, hfToken)
        val totalBytes = head.contentLength.coerceAtLeast(0L)
        val memoryInfo = ActivityManager.MemoryInfo()
        (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memoryInfo)
        val ramWarning = if (totalBytes > 0L && totalBytes > memoryInfo.totalMem) {
            "Warning: this download is larger than your phone RAM. Downloading is allowed, but local inference may require a smaller quant or an external runtime."
        } else {
            ""
        }
        val destinationName = destinationFileName(draft.repoOrUrl, draft.filePath, resolvedUrl)
        return ModelDownloadInspection(
            title = destinationName,
            sourceUrl = resolvedUrl,
            destinationFileName = destinationName,
            totalBytes = totalBytes,
            totalBytesLabel = if (totalBytes > 0) Formatter.formatShortFileSize(context, totalBytes) else "unknown size",
            deviceRamBytes = memoryInfo.totalMem,
            deviceRamLabel = Formatter.formatShortFileSize(context, memoryInfo.totalMem),
            ramWarning = ramWarning,
            supportsResume = head.acceptRanges,
            abiSummary = android.os.Build.SUPPORTED_ABIS.joinToString(),
        )
    }

    fun enqueueDownload(
        context: Context,
        store: LocalModelDownloadStore,
        draft: ModelDownloadDraft,
        hfToken: String,
        dataSaverMode: Boolean,
        allowMetered: Boolean = !dataSaverMode,
        allowRoaming: Boolean = false,
    ): LocalModelDownloadRecord {
        val record = enqueueDownloadRecord(
            context = context,
            draft = draft,
            hfToken = hfToken,
            dataSaverMode = dataSaverMode,
            allowMetered = allowMetered,
            allowRoaming = allowRoaming,
        )
        store.upsertDownload(record)
        return record
    }

    fun restartDownloadOnMobileData(
        context: Context,
        store: LocalModelDownloadStore,
        recordId: String,
        hfToken: String,
    ): LocalModelDownloadRecord? {
        val existing = store.findDownload(recordId) ?: return null
        val downloadManager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        runCatching { downloadManager.remove(existing.downloadManagerId) }
        runCatching { File(existing.destinationPath).delete() }
        val restarted = enqueueDownloadRecord(
            context = context,
            draft = ModelDownloadDraft(
                repoOrUrl = existing.repoOrUrl,
                filePath = existing.filePath,
                revision = existing.revision,
                runtimeFlavor = existing.runtimeFlavor,
            ),
            hfToken = hfToken,
            dataSaverMode = false,
            allowMetered = true,
            allowRoaming = true,
            recordId = existing.id,
        )
        store.upsertDownload(restarted)
        return restarted
    }

    private fun enqueueDownloadRecord(
        context: Context,
        draft: ModelDownloadDraft,
        hfToken: String,
        dataSaverMode: Boolean,
        allowMetered: Boolean,
        allowRoaming: Boolean,
        recordId: String = UUID.randomUUID().toString(),
    ): LocalModelDownloadRecord {
        val inspection = inspectCandidate(context, draft, hfToken)
        val targetFile = modelsDirectory(context).resolve(uniqueFileName(modelsDirectory(context), inspection.destinationFileName))
        val request = DownloadManager.Request(Uri.parse(inspection.sourceUrl)).apply {
            setTitle("Hermes model: ${inspection.title}")
            setDescription("Downloading a local model for Hermes")
            setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            setVisibleInDownloadsUi(true)
            setAllowedOverRoaming(allowRoaming)
            setAllowedOverMetered(allowMetered)
            if (looksLikeHuggingFaceResource(inspection.sourceUrl) && hfToken.isNotBlank()) {
                addRequestHeader("Authorization", "Bearer $hfToken")
            }
            setDestinationUri(Uri.fromFile(targetFile))
        }
        val downloadManager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        val downloadId = downloadManager.enqueue(request)
        return LocalModelDownloadRecord(
            id = recordId,
            title = inspection.title,
            sourceUrl = inspection.sourceUrl,
            repoOrUrl = draft.repoOrUrl,
            filePath = draft.filePath,
            revision = draft.revision,
            runtimeFlavor = draft.runtimeFlavor,
            destinationFileName = inspection.destinationFileName,
            destinationPath = targetFile.absolutePath,
            downloadManagerId = downloadId,
            totalBytes = inspection.totalBytes,
            downloadedBytes = 0L,
            status = "queued",
            statusMessage = queuedStatusMessage(
                context = context,
                dataSaverMode = dataSaverMode,
                allowMetered = allowMetered,
                allowRoaming = allowRoaming,
            ),
            ramWarning = inspection.ramWarning,
            supportsResume = inspection.supportsResume,
            allowMetered = allowMetered,
            allowRoaming = allowRoaming,
        )
    }

    fun refreshDownloads(context: Context, store: LocalModelDownloadStore): List<LocalModelDownloadRecord> {
        val refreshed = store.loadDownloads().map { refreshRecord(context, it) }
        store.saveDownloads(refreshed)
        return refreshed
    }

    fun refreshRecord(context: Context, record: LocalModelDownloadRecord): LocalModelDownloadRecord {
        val downloadManager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        val query = DownloadManager.Query().setFilterById(record.downloadManagerId)
        downloadManager.query(query)?.use { cursor ->
            if (!cursor.moveToFirst()) {
                return record.copy(
                    status = if (File(record.destinationPath).exists()) "completed" else "missing",
                    statusMessage = if (File(record.destinationPath).exists()) "Download file is present on disk" else "Android no longer reports this download",
                    updatedAtEpochMs = System.currentTimeMillis(),
                )
            }
            val status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
            val downloadedBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
            val totalBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)).coerceAtLeast(record.totalBytes)
            val reason = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))
            val localUri = cursor.getString(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_LOCAL_URI)).orEmpty()
            val statusPair = statusLabel(context, record, status, reason)
            return record.copy(
                destinationPath = localUri.removePrefix("file://").ifBlank { record.destinationPath },
                downloadedBytes = downloadedBytes,
                totalBytes = totalBytes,
                status = statusPair.first,
                statusMessage = statusPair.second,
                updatedAtEpochMs = System.currentTimeMillis(),
            )
        }
        return record
    }

    fun removeDownload(context: Context, store: LocalModelDownloadStore, recordId: String) {
        val existing = store.findDownload(recordId) ?: return
        val downloadManager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        runCatching { downloadManager.remove(existing.downloadManagerId) }
        runCatching { File(existing.destinationPath).delete() }
        store.removeDownload(recordId)
    }

    fun setPreferredDownload(store: LocalModelDownloadStore, recordId: String) {
        store.setPreferredDownloadId(recordId)
    }

    private fun resolveDownloadUrl(draft: ModelDownloadDraft, hfToken: String): String {
        val trimmed = draft.repoOrUrl.trim()
        val explicitFilePath = draft.filePath.trim().trim('/')
        val requestedRevision = draft.revision.trim().ifBlank { "main" }
        parseHuggingFaceReference(trimmed)?.let { parsedReference ->
            val reference = normalizeReferenceForRuntime(
                reference = parsedReference,
                runtimeFlavor = draft.runtimeFlavor,
                explicitFilePath = explicitFilePath,
            )
            val resolvedRevision = reference.revision ?: requestedRevision
            val resolvedFilePath = explicitFilePath.ifBlank {
                reference.filePath ?: findCompatibleRepoFile(
                    repoId = reference.repoId,
                    revision = resolvedRevision,
                    runtimeFlavor = draft.runtimeFlavor,
                    hfToken = hfToken,
                )
            }
            return "$HUGGING_FACE_BASE/${reference.repoId}/resolve/$resolvedRevision/$resolvedFilePath?download=true"
        }
        if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
            return trimmed
        }
        val repo = trimmed.removePrefix("hf://").trim('/').ifBlank {
            throw IllegalArgumentException("Enter a Hugging Face repo or a direct model URL")
        }
        val resolvedFilePath = explicitFilePath.ifBlank {
            findCompatibleRepoFile(
                repoId = repo,
                revision = requestedRevision,
                runtimeFlavor = draft.runtimeFlavor,
                hfToken = hfToken,
            )
        }
        return "$HUGGING_FACE_BASE/$repo/resolve/$requestedRevision/$resolvedFilePath?download=true"
    }

    private fun parseHuggingFaceReference(repoOrUrl: String): HuggingFaceReference? {
        val trimmed = repoOrUrl.trim()
        if (trimmed.startsWith("hf://")) {
            val repoId = trimmed.removePrefix("hf://").trim('/').ifBlank { return null }
            return HuggingFaceReference(repoId = repoId)
        }
        if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) {
            return HuggingFaceReference(repoId = trimmed.trim('/').ifBlank { return null })
        }
        val uri = Uri.parse(trimmed)
        val host = uri.host.orEmpty().lowercase(Locale.US)
        if (!host.contains("huggingface.co") && !host.contains("hf.co")) {
            return null
        }
        val segments = uri.pathSegments.filter { it.isNotBlank() }
        if (segments.size < 2) {
            return null
        }
        val repoId = "${segments[0]}/${segments[1]}"
        if (segments.size >= 5 && segments[2] in setOf("blob", "resolve")) {
            val filePath = segments.drop(4).joinToString("/")
            return HuggingFaceReference(
                repoId = repoId,
                revision = segments[3].ifBlank { null },
                filePath = filePath.ifBlank { null },
            )
        }
        return HuggingFaceReference(repoId = repoId)
    }

    private fun normalizeReferenceForRuntime(
        reference: HuggingFaceReference,
        runtimeFlavor: String,
        explicitFilePath: String,
    ): HuggingFaceReference {
        if (explicitFilePath.isNotBlank() || reference.filePath != null || !runtimeFlavor.equals("LiteRT-LM", ignoreCase = true)) {
            return reference
        }
        val aliasRepoId = liteRtAlias(reference.repoId) ?: return reference
        return reference.copy(
            repoId = aliasRepoId,
            revision = null,
            filePath = null,
        )
    }

    private fun findCompatibleRepoFile(
        repoId: String,
        revision: String,
        runtimeFlavor: String,
        hfToken: String,
    ): String {
        val siblings = loadRepoFiles(repoId = repoId, revision = revision, hfToken = hfToken)
        val compatible = siblings
            .filter { isCompatibleRepoFile(it, runtimeFlavor) }
            .sortedWith(compareBy<String> { compatibleFileRank(it, runtimeFlavor) }.thenBy { it.lowercase(Locale.US) })

        return compatible.firstOrNull()
            ?: throw IllegalArgumentException(noCompatibleArtifactMessage(repoId, runtimeFlavor))
    }

    private fun noCompatibleArtifactMessage(repoId: String, runtimeFlavor: String): String {
        val aliasRepoId = liteRtAlias(repoId)
        val normalizedRepoId = repoId.lowercase(Locale.US)
        val ggufSuggestion = GGUF_RECOMMENDED_REPOS[normalizedRepoId]
        return when {
            runtimeFlavor.equals("GGUF", ignoreCase = true) && normalizedRepoId in GEMMA4_SOURCE_REPOS ->
                "No compatible GGUF artifact found in huggingface.co/$repoId. Google AI Edge Gallery runs Gemma 4 from LiteRT-LM repos like ${aliasRepoId ?: "litert-community/gemma-4-E2B-it-litert-lm"}, not from the raw google model page."

            runtimeFlavor.equals("GGUF", ignoreCase = true) && ggufSuggestion != null ->
                "No compatible GGUF artifact found in huggingface.co/$repoId. Try the GGUF-ready repo $ggufSuggestion instead."

            runtimeFlavor.equals("LiteRT-LM", ignoreCase = true) && aliasRepoId != null ->
                "No compatible LiteRT-LM artifact found in huggingface.co/$repoId. Try the mobile-ready repo $aliasRepoId instead."

            runtimeFlavor.equals("LiteRT-LM", ignoreCase = true) && ("nemotron" in normalizedRepoId || "cascade" in normalizedRepoId) ->
                "No compatible LiteRT-LM artifact found in huggingface.co/$repoId. Hermes currently recommends llama.cpp + GGUF for Nemotron / Cascade families."

            else -> "No compatible $runtimeFlavor artifact found in huggingface.co/$repoId"
        }
    }

    private fun liteRtAlias(repoId: String): String? {
        return LITERT_ALIAS_REPOS[repoId.lowercase(Locale.US)]
    }

    private fun loadRepoFiles(repoId: String, revision: String, hfToken: String): List<String> {
        val metadata = fetchRepoMetadata(repoId = repoId, revision = revision, hfToken = hfToken)
        val siblings = metadata.optJSONArray("siblings") ?: return emptyList()
        return buildList {
            for (index in 0 until siblings.length()) {
                val item = siblings.optJSONObject(index) ?: continue
                val fileName = item.optString("rfilename").ifBlank { item.optString("path") }
                if (fileName.isNotBlank()) {
                    add(fileName)
                }
            }
        }
    }

    private fun fetchRepoMetadata(repoId: String, revision: String, hfToken: String): JSONObject {
        val revisionEndpoint = if (revision.isBlank() || revision == "main") {
            null
        } else {
            "$HUGGING_FACE_API$repoId/revision/${Uri.encode(revision)}"
        }
        val endpoints = listOfNotNull(revisionEndpoint, "$HUGGING_FACE_API$repoId")
        var lastFailure: Exception? = null
        for (endpoint in endpoints) {
            try {
                return getJson(endpoint, hfToken)
            } catch (error: Exception) {
                lastFailure = error
            }
        }
        throw IllegalArgumentException(lastFailure?.message ?: "Unable to inspect huggingface.co/$repoId")
    }

    private fun getJson(url: String, hfToken: String): JSONObject {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            requestMethod = "GET"
            connectTimeout = 15_000
            readTimeout = 15_000
            setRequestProperty("Accept", "application/json")
            if (looksLikeHuggingFaceResource(url) && hfToken.isNotBlank()) {
                setRequestProperty("Authorization", "Bearer $hfToken")
            }
        }
        return try {
            val responseCode = connection.responseCode
            if (responseCode !in 200..299) {
                val errorBody = connection.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
                val detail = errorBody.take(160).ifBlank { "HTTP $responseCode" }
                throw IllegalArgumentException("Unable to inspect huggingface.co metadata: $detail")
            }
            JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
        } finally {
            connection.disconnect()
        }
    }

    private fun isCompatibleRepoFile(path: String, runtimeFlavor: String): Boolean {
        val lower = path.lowercase(Locale.US)
        return when (runtimeFlavor.uppercase(Locale.US)) {
            "LITERT-LM" -> lower.endsWith(".litertlm")
            else -> lower.endsWith(".gguf")
        }
    }

    private fun compatibleFileRank(path: String, runtimeFlavor: String): Int {
        val lower = path.lowercase(Locale.US)
        return when (runtimeFlavor.uppercase(Locale.US)) {
            "LITERT-LM" -> 0
            else -> when {
                "q4_k_m" in lower -> 0
                "q4_k_s" in lower -> 1
                "iq4" in lower -> 2
                "q5_k_m" in lower -> 3
                "q5" in lower -> 4
                "q6" in lower -> 5
                "q8" in lower -> 6
                else -> 7
            }
        }
    }

    private fun headProbe(sourceUrl: String, hfToken: String): HeadProbeResult {
        val connection = (URL(sourceUrl).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            requestMethod = "HEAD"
            connectTimeout = 15_000
            readTimeout = 15_000
            if (looksLikeHuggingFaceResource(sourceUrl) && hfToken.isNotBlank()) {
                setRequestProperty("Authorization", "Bearer $hfToken")
            }
        }
        return try {
            HeadProbeResult(
                contentLength = connection.contentLengthLong,
                acceptRanges = connection.getHeaderField("Accept-Ranges").orEmpty().contains("bytes", ignoreCase = true),
            )
        } finally {
            connection.disconnect()
        }
    }

    private fun destinationFileName(repoOrUrl: String, filePath: String, sourceUrl: String): String {
        val raw = when {
            filePath.isNotBlank() -> filePath.substringAfterLast('/')
            repoOrUrl.startsWith("http://") || repoOrUrl.startsWith("https://") -> Uri.parse(repoOrUrl).lastPathSegment
            else -> Uri.parse(sourceUrl).lastPathSegment
        }.orEmpty().substringBefore('?')
        return sanitizeFileName(raw.ifBlank { "model.bin" })
    }

    private fun uniqueFileName(directory: File, desiredName: String): String {
        val existing = directory.resolve(desiredName)
        if (!existing.exists()) {
            return desiredName
        }
        val dotIndex = desiredName.lastIndexOf('.')
        val stem = if (dotIndex > 0) desiredName.substring(0, dotIndex) else desiredName
        val ext = if (dotIndex > 0) desiredName.substring(dotIndex) else ""
        var counter = 1
        while (true) {
            val candidate = "$stem-$counter$ext"
            if (!directory.resolve(candidate).exists()) {
                return candidate
            }
            counter += 1
        }
    }

    private fun sanitizeFileName(name: String): String {
        return name.replace(Regex("[^A-Za-z0-9._-]"), "_").lowercase(Locale.US)
    }

    private fun looksLikeHuggingFaceResource(url: String): Boolean {
        return url.contains("huggingface.co", ignoreCase = true) || url.contains("hf.co", ignoreCase = true)
    }

    private fun activeNetworkPolicySnapshot(context: Context): NetworkPolicySnapshot {
        val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
        val capabilities = connectivityManager?.getNetworkCapabilities(connectivityManager.activeNetwork)
        if (capabilities == null) {
            return NetworkPolicySnapshot(label = "offline", isMetered = false, isRoaming = false)
        }
        val label = when {
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "Wi‑Fi"
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "cellular"
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ethernet"
            else -> "network"
        }
        val isMetered = !capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED)
        val isRoaming = capabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) &&
            !capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_ROAMING)
        return NetworkPolicySnapshot(label = label, isMetered = isMetered, isRoaming = isRoaming)
    }

    private fun queuedStatusMessage(
        context: Context,
        dataSaverMode: Boolean,
        allowMetered: Boolean,
        allowRoaming: Boolean,
    ): String {
        val network = activeNetworkPolicySnapshot(context)
        return when {
            !allowRoaming && network.isRoaming ->
                "Queued while roaming is blocked for this download. Use Wi‑Fi or restart on mobile data to allow roaming."
            dataSaverMode || !allowMetered ->
                "Queued with Data saver mode: large transfers wait for Wi‑Fi / unmetered connectivity"
            else -> "Queued in Android DownloadManager over ${network.label}"
        }
    }

    private fun statusLabel(
        context: Context,
        record: LocalModelDownloadRecord,
        status: Int,
        reason: Int,
    ): Pair<String, String> {
        val network = activeNetworkPolicySnapshot(context)
        return when (status) {
            DownloadManager.STATUS_PENDING -> "queued" to when {
                !record.allowRoaming && network.isRoaming ->
                    "Waiting because roaming downloads are blocked. Restart on mobile data or switch to Wi‑Fi."
                !record.allowMetered && network.isMetered ->
                    "Waiting for Wi‑Fi / unmetered connectivity. Restart on mobile data to continue over cellular."
                else -> "Waiting for Android to start the transfer"
            }
            DownloadManager.STATUS_RUNNING -> "downloading" to "Downloading in the background with system-managed resume support"
            DownloadManager.STATUS_PAUSED -> "paused" to when {
                !record.allowRoaming && network.isRoaming ->
                    "Paused because Android treats the current connection as roaming. Restart on mobile data or switch to Wi‑Fi."
                !record.allowMetered && (reason == DownloadManager.PAUSED_QUEUED_FOR_WIFI || network.isMetered) ->
                    "Paused until Wi‑Fi / unmetered connectivity is available. Restart on mobile data below to continue over cellular."
                reason == DownloadManager.PAUSED_WAITING_FOR_NETWORK -> "Paused until network connectivity returns"
                reason == DownloadManager.PAUSED_QUEUED_FOR_WIFI -> "Paused until Wi‑Fi / unmetered connectivity is available"
                reason == DownloadManager.PAUSED_WAITING_TO_RETRY -> "Paused while Android retries the connection"
                reason == DownloadManager.PAUSED_UNKNOWN -> "Paused by Android"
                else -> "Paused by Android"
            }
            DownloadManager.STATUS_SUCCESSFUL -> "completed" to "Download completed and saved locally"
            DownloadManager.STATUS_FAILED -> "failed" to when (reason) {
                DownloadManager.ERROR_HTTP_DATA_ERROR -> "Network transfer failed"
                DownloadManager.ERROR_INSUFFICIENT_SPACE -> "Download failed: insufficient storage"
                DownloadManager.ERROR_TOO_MANY_REDIRECTS -> "Download failed: too many redirects"
                DownloadManager.ERROR_FILE_ALREADY_EXISTS -> "Download failed: file already exists"
                else -> "Download failed (reason $reason)"
            }
            else -> "unknown" to "Android reported an unknown download state"
        }
    }

    private data class HeadProbeResult(
        val contentLength: Long,
        val acceptRanges: Boolean,
    )

    private data class HuggingFaceReference(
        val repoId: String,
        val revision: String? = null,
        val filePath: String? = null,
    )

    private val GEMMA4_SOURCE_REPOS = setOf(
        "google/gemma-4-e2b",
        "google/gemma-4-e2b-it",
        "google/gemma-4-e4b",
        "google/gemma-4-e4b-it",
    )
}
