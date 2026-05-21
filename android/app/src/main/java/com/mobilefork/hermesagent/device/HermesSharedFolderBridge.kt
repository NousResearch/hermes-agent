package com.mobilefork.hermesagent.device

import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import com.mobilefork.hermesagent.HermesApplication
import com.mobilefork.hermesagent.data.DeviceCapabilityStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.ArrayDeque

object HermesSharedFolderBridge {
    private const val DEFAULT_MIME_TYPE = "text/plain"
    private const val MAX_LIST_LIMIT = 200

    @JvmStatic
    fun listEntriesJson(relativePath: String?, recursive: Boolean, limit: Int): String {
        return runCatching {
            val root = sharedRoot() ?: return errorJson("No shared folder grant is available yet")
            val normalizedPath = normalizePath(relativePath.orEmpty())
            val target = findExistingDocument(root, segmentsFor(normalizedPath))
                ?: return errorJson("Shared folder path not found: ${normalizedPath.ifBlank { "/" }}")
            if (!target.isDirectory) {
                return errorJson("Shared folder path is not a directory: ${normalizedPath.ifBlank { "/" }}")
            }

            val cappedLimit = limit.coerceIn(1, MAX_LIST_LIMIT)
            val entries = mutableListOf<JSONObject>()
            var truncated = false

            if (recursive) {
                val queue = ArrayDeque<Pair<DocumentFile, String>>()
                queue.add(target to normalizedPath)
                while (queue.isNotEmpty() && entries.size < cappedLimit) {
                    val (directory, basePath) = queue.removeFirst()
                    for (child in sortedChildren(directory)) {
                        val childName = child.name.orEmpty()
                        if (childName.isBlank()) {
                            continue
                        }
                        val childRelativePath = joinPath(basePath, childName)
                        entries.add(entryJson(child, childRelativePath))
                        if (entries.size >= cappedLimit) {
                            truncated = true
                            break
                        }
                        if (child.isDirectory) {
                            queue.add(child to childRelativePath)
                        }
                    }
                }
            } else {
                for (child in sortedChildren(target)) {
                    val childName = child.name.orEmpty()
                    if (childName.isBlank()) {
                        continue
                    }
                    entries.add(entryJson(child, joinPath(normalizedPath, childName)))
                    if (entries.size >= cappedLimit) {
                        truncated = sortedChildren(target).size > cappedLimit
                        break
                    }
                }
            }

            JSONObject().apply {
                put("shared_tree_label", sharedFolderLabel())
                put("requested_path", normalizedPath)
                put("recursive", recursive)
                put("entries", JSONArray(entries))
                put("truncated", truncated)
            }.toString()
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    @JvmStatic
    fun readTextFileJson(relativePath: String, maxChars: Int): String {
        return runCatching {
            val root = sharedRoot() ?: return errorJson("No shared folder grant is available yet")
            val normalizedPath = normalizeRequiredFilePath(relativePath)
            val target = findExistingDocument(root, segmentsFor(normalizedPath))
                ?: return errorJson("Shared folder file not found: $normalizedPath")
            if (!target.isFile) {
                return errorJson("Shared folder path is not a file: $normalizedPath")
            }

            val bytes = appContext().contentResolver.openInputStream(target.uri)?.use { it.readBytes() }
                ?: return errorJson("Unable to open shared folder file: $normalizedPath")
            val content = bytes.toString(Charsets.UTF_8)
            val cappedChars = maxChars.coerceAtLeast(1)
            if (content.length > cappedChars) {
                return errorJson("Shared folder file is too large for one read (${content.length} chars > $cappedChars)")
            }

            JSONObject().apply {
                put("relative_path", normalizedPath)
                put("content", content)
                put("size_bytes", bytes.size)
                put("line_count", if (content.isEmpty()) 0 else content.count { it == '\n' } + 1)
            }.toString()
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    @JvmStatic
    fun writeTextFileJson(relativePath: String, content: String, createDirectories: Boolean): String {
        return runCatching {
            val root = sharedRoot() ?: return errorJson("No shared folder grant is available yet")
            val normalizedPath = normalizeRequiredFilePath(relativePath)
            val segments = segmentsFor(normalizedPath)
            val parentSegments = segments.dropLast(1)
            val fileName = segments.last()
            val parent = if (createDirectories) {
                ensureDirectory(root, parentSegments)
            } else {
                findExistingDocument(root, parentSegments)
                    ?: return errorJson("Shared folder parent directory not found: ${parentSegments.joinToString("/")}")
            }

            if (!parent.isDirectory) {
                return errorJson("Shared folder parent is not a directory: ${parentSegments.joinToString("/")}")
            }

            val existing = parent.findFile(fileName)
            if (existing?.isDirectory == true) {
                return errorJson("Cannot overwrite directory with file content: $normalizedPath")
            }
            val target = existing ?: parent.createFile(DEFAULT_MIME_TYPE, fileName)
                ?: return errorJson("Unable to create shared folder file: $normalizedPath")

            appContext().contentResolver.openOutputStream(target.uri, "wt")?.use { output ->
                output.write(content.toByteArray(Charsets.UTF_8))
            } ?: return errorJson("Unable to open shared folder file for writing: $normalizedPath")

            JSONObject().apply {
                put("success", true)
                put("relative_path", normalizedPath)
                put("chars_written", content.length)
            }.toString()
        }.getOrElse { error ->
            errorJson(error.message ?: error.javaClass.simpleName)
        }
    }

    private fun appContext() = HermesApplication.instance.applicationContext

    private fun sharedRoot(): DocumentFile? {
        val sharedUri = DeviceCapabilityStore(appContext()).load().sharedFolderUri.ifBlank { return null }
        return DocumentFile.fromTreeUri(appContext(), Uri.parse(sharedUri))
    }

    private fun sharedFolderLabel(): String {
        return DeviceCapabilityStore(appContext()).load().sharedFolderLabel
    }

    private fun normalizeRequiredFilePath(relativePath: String): String {
        val normalized = normalizePath(relativePath)
        if (normalized.isBlank()) {
            throw IOException("A relative file path is required")
        }
        return normalized
    }

    private fun normalizePath(relativePath: String): String {
        return segmentsFor(relativePath).joinToString("/")
    }

    private fun segmentsFor(relativePath: String): List<String> {
        val sanitized = mutableListOf<String>()
        for (segment in relativePath.replace('\\', '/').split('/')) {
            val trimmed = segment.trim()
            when {
                trimmed.isEmpty() || trimmed == "." -> continue
                trimmed == ".." -> throw IOException("Parent directory traversal is not allowed")
                else -> sanitized.add(trimmed)
            }
        }
        return sanitized
    }

    private fun findExistingDocument(root: DocumentFile, segments: List<String>): DocumentFile? {
        var current = root
        for (segment in segments) {
            current = current.findFile(segment) ?: return null
        }
        return current
    }

    private fun ensureDirectory(root: DocumentFile, segments: List<String>): DocumentFile {
        var current = root
        for (segment in segments) {
            val existing = current.findFile(segment)
            current = when {
                existing == null -> current.createDirectory(segment)
                    ?: throw IOException("Unable to create shared folder directory: $segment")
                existing.isDirectory -> existing
                else -> throw IOException("Shared folder path segment is not a directory: $segment")
            }
        }
        return current
    }

    private fun sortedChildren(directory: DocumentFile): List<DocumentFile> {
        return directory.listFiles().sortedWith(
            compareBy<DocumentFile>({ !it.isDirectory }, { it.name.orEmpty().lowercase() })
        )
    }

    private fun entryJson(document: DocumentFile, relativePath: String): JSONObject {
        return JSONObject().apply {
            put("name", document.name.orEmpty())
            put("relative_path", relativePath)
            put("kind", if (document.isDirectory) "directory" else "file")
            put("mime_type", document.type.orEmpty())
            put("size_bytes", document.length())
            put("last_modified_epoch_ms", document.lastModified())
        }
    }

    private fun joinPath(base: String, child: String): String {
        return if (base.isBlank()) child else "$base/$child"
    }

    private fun errorJson(message: String): String {
        return JSONObject().apply {
            put("error", message)
        }.toString()
    }
}
