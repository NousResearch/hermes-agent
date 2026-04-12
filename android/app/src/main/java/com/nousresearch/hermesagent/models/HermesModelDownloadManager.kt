package com.nousresearch.hermesagent.models

import android.app.ActivityManager
import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.text.format.Formatter
import com.nousresearch.hermesagent.data.LocalModelDownloadRecord
import com.nousresearch.hermesagent.data.LocalModelDownloadStore
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

    fun modelsDirectory(context: Context): File {
        return (context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
            ?: File(context.filesDir, "downloads")).resolve("models").apply { mkdirs() }
    }

    fun inspectCandidate(context: Context, draft: ModelDownloadDraft, hfToken: String): ModelDownloadInspection {
        val resolvedUrl = resolveDownloadUrl(draft.repoOrUrl, draft.filePath, draft.revision)
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
    ): LocalModelDownloadRecord {
        val inspection = inspectCandidate(context, draft, hfToken)
        val targetFile = modelsDirectory(context).resolve(uniqueFileName(modelsDirectory(context), inspection.destinationFileName))
        val request = DownloadManager.Request(Uri.parse(inspection.sourceUrl)).apply {
            setTitle("Hermes model: ${inspection.title}")
            setDescription("Downloading a local model for Hermes")
            setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            setVisibleInDownloadsUi(true)
            setAllowedOverRoaming(false)
            setAllowedOverMetered(!dataSaverMode)
            if (looksLikeHuggingFaceResource(inspection.sourceUrl) && hfToken.isNotBlank()) {
                addRequestHeader("Authorization", "Bearer $hfToken")
            }
            setDestinationUri(Uri.fromFile(targetFile))
        }
        val downloadManager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        val downloadId = downloadManager.enqueue(request)
        val record = LocalModelDownloadRecord(
            id = UUID.randomUUID().toString(),
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
            statusMessage = if (dataSaverMode) {
                "Queued with Data saver mode: large transfers wait for Wi‑Fi / unmetered connectivity"
            } else {
                "Queued in Android DownloadManager"
            },
            ramWarning = inspection.ramWarning,
            supportsResume = inspection.supportsResume,
        )
        store.upsertDownload(record)
        return record
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
            val statusPair = statusLabel(status, reason)
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

    private fun resolveDownloadUrl(repoOrUrl: String, filePath: String, revision: String): String {
        val trimmed = repoOrUrl.trim()
        if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
            return trimmed
        }
        val repo = trimmed.removePrefix("hf://").trim('/').ifBlank {
            throw IllegalArgumentException("Enter a Hugging Face repo or a direct model URL")
        }
        val resolvedRevision = revision.trim().ifBlank { "main" }
        val resolvedFilePath = filePath.trim().trim('/').ifBlank {
            throw IllegalArgumentException("Enter the file path inside the repo")
        }
        return "$HUGGING_FACE_BASE/$repo/resolve/$resolvedRevision/$resolvedFilePath?download=true"
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

    private fun statusLabel(status: Int, reason: Int): Pair<String, String> {
        return when (status) {
            DownloadManager.STATUS_PENDING -> "queued" to "Waiting for Android to start the transfer"
            DownloadManager.STATUS_RUNNING -> "downloading" to "Downloading in the background with system-managed resume support"
            DownloadManager.STATUS_PAUSED -> "paused" to when (reason) {
                DownloadManager.PAUSED_WAITING_FOR_NETWORK -> "Paused until network connectivity returns"
                DownloadManager.PAUSED_QUEUED_FOR_WIFI -> "Paused until Wi‑Fi / unmetered connectivity is available"
                DownloadManager.PAUSED_WAITING_TO_RETRY -> "Paused while Android retries the connection"
                DownloadManager.PAUSED_UNKNOWN -> "Paused by Android"
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
}
