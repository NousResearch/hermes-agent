package com.nousresearch.hermesagent.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

data class LocalModelDownloadRecord(
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val sourceUrl: String,
    val repoOrUrl: String,
    val filePath: String,
    val revision: String,
    val runtimeFlavor: String,
    val destinationFileName: String,
    val destinationPath: String,
    val downloadManagerId: Long,
    val totalBytes: Long = 0,
    val downloadedBytes: Long = 0,
    val status: String = "queued",
    val statusMessage: String = "Queued",
    val ramWarning: String = "",
    val supportsResume: Boolean = true,
    val allowMetered: Boolean = true,
    val allowRoaming: Boolean = false,
    val updatedAtEpochMs: Long = System.currentTimeMillis(),
)

class LocalModelDownloadStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun loadDownloads(): List<LocalModelDownloadRecord> {
        val raw = preferences.getString(KEY_DOWNLOADS_JSON, null).orEmpty()
        if (raw.isBlank()) {
            return emptyList()
        }
        return runCatching {
            val array = JSONArray(raw)
            buildList {
                for (index in 0 until array.length()) {
                    val item = array.optJSONObject(index) ?: continue
                    add(item.toRecord())
                }
            }
        }.getOrDefault(emptyList())
    }

    fun saveDownloads(downloads: List<LocalModelDownloadRecord>) {
        val array = JSONArray()
        downloads.forEach { array.put(it.toJson()) }
        preferences.edit().putString(KEY_DOWNLOADS_JSON, array.toString()).apply()
    }

    fun upsertDownload(download: LocalModelDownloadRecord) {
        val current = loadDownloads().toMutableList()
        val index = current.indexOfFirst { it.id == download.id }
        if (index >= 0) {
            current[index] = download
        } else {
            current += download
        }
        saveDownloads(current.sortedByDescending { it.updatedAtEpochMs })
    }

    fun removeDownload(recordId: String) {
        val remaining = loadDownloads().filterNot { it.id == recordId }
        saveDownloads(remaining)
        if (preferredDownloadId() == recordId) {
            setPreferredDownloadId("")
        }
    }

    fun findDownload(recordId: String): LocalModelDownloadRecord? {
        return loadDownloads().firstOrNull { it.id == recordId }
    }

    fun setPreferredDownloadId(recordId: String) {
        preferences.edit().putString(KEY_PREFERRED_DOWNLOAD_ID, recordId).apply()
    }

    fun preferredDownloadId(): String {
        return preferences.getString(KEY_PREFERRED_DOWNLOAD_ID, "").orEmpty()
    }

    private fun LocalModelDownloadRecord.toJson(): JSONObject {
        return JSONObject().apply {
            put("id", id)
            put("title", title)
            put("sourceUrl", sourceUrl)
            put("repoOrUrl", repoOrUrl)
            put("filePath", filePath)
            put("revision", revision)
            put("runtimeFlavor", runtimeFlavor)
            put("destinationFileName", destinationFileName)
            put("destinationPath", destinationPath)
            put("downloadManagerId", downloadManagerId)
            put("totalBytes", totalBytes)
            put("downloadedBytes", downloadedBytes)
            put("status", status)
            put("statusMessage", statusMessage)
            put("ramWarning", ramWarning)
            put("supportsResume", supportsResume)
            put("allowMetered", allowMetered)
            put("allowRoaming", allowRoaming)
            put("updatedAtEpochMs", updatedAtEpochMs)
        }
    }

    private fun JSONObject.toRecord(): LocalModelDownloadRecord {
        return LocalModelDownloadRecord(
            id = optString("id", UUID.randomUUID().toString()),
            title = optString("title", "Downloaded model"),
            sourceUrl = optString("sourceUrl", ""),
            repoOrUrl = optString("repoOrUrl", ""),
            filePath = optString("filePath", ""),
            revision = optString("revision", "main"),
            runtimeFlavor = optString("runtimeFlavor", "GGUF"),
            destinationFileName = optString("destinationFileName", "model.bin"),
            destinationPath = optString("destinationPath", ""),
            downloadManagerId = optLong("downloadManagerId", -1L),
            totalBytes = optLong("totalBytes", 0L),
            downloadedBytes = optLong("downloadedBytes", 0L),
            status = optString("status", "queued"),
            statusMessage = optString("statusMessage", "Queued"),
            ramWarning = optString("ramWarning", ""),
            supportsResume = optBoolean("supportsResume", true),
            allowMetered = optBoolean("allowMetered", true),
            allowRoaming = optBoolean("allowRoaming", false),
            updatedAtEpochMs = optLong("updatedAtEpochMs", System.currentTimeMillis()),
        )
    }

    companion object {
        private const val PREFS_NAME = "hermes_android_local_model_downloads"
        private const val KEY_DOWNLOADS_JSON = "downloads_json"
        private const val KEY_PREFERRED_DOWNLOAD_ID = "preferred_download_id"
    }
}
