package com.mobilefork.hermesagent.device

import android.net.Uri
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.URLDecoder
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlin.concurrent.thread

object HermesLocalFileHttpServer {
    private const val HOST = "127.0.0.1"
    private const val TTL_MILLIS = 10 * 60 * 1000L
    private val lock = Any()
    private val sharedFiles = ConcurrentHashMap<String, SharedFile>()
    @Volatile private var server: RunningServer? = null

    fun shareFile(file: File, mimeType: String): Uri {
        val canonical = file.canonicalFile
        val running = ensureRunning()
        pruneExpired()
        val token = UUID.randomUUID().toString()
        sharedFiles[token] = SharedFile(
            file = canonical,
            mimeType = mimeType,
            expiresAtMillis = System.currentTimeMillis() + TTL_MILLIS,
        )
        val encodedName = URLEncoder.encode(canonical.name, StandardCharsets.UTF_8.name()).replace("+", "%20")
        return Uri.parse("http://$HOST:${running.port}/files/$token/$encodedName")
    }

    private fun ensureRunning(): RunningServer {
        server?.let { return it }
        synchronized(lock) {
            server?.let { return it }
            val socket = ServerSocket(0, 50, InetAddress.getByName(HOST))
            val running = RunningServer(socket.localPort, socket)
            thread(name = "HermesLocalFileHttpServer", isDaemon = true) {
                runServer(socket)
            }
            server = running
            return running
        }
    }

    private fun runServer(serverSocket: ServerSocket) {
        while (!serverSocket.isClosed) {
            runCatching {
                serverSocket.accept().use(::handleClient)
            }
        }
    }

    private fun handleClient(socket: Socket) {
        socket.soTimeout = 5_000
        val input = BufferedReader(InputStreamReader(socket.getInputStream(), StandardCharsets.US_ASCII))
        val requestLine = input.readLine().orEmpty()
        while (true) {
            val line = input.readLine() ?: break
            if (line.isEmpty()) break
        }
        val parts = requestLine.split(" ")
        if (parts.size < 2) {
            socket.sendHttp(400, "Bad Request", "text/plain", "Bad request".toByteArray(StandardCharsets.UTF_8))
            return
        }
        val method = parts[0].uppercase()
        if (method != "GET" && method != "HEAD") {
            socket.sendHttp(405, "Method Not Allowed", "text/plain", "Method not allowed".toByteArray(StandardCharsets.UTF_8))
            return
        }
        val token = tokenFromPath(parts[1])
        val shared = token?.let(sharedFiles::get)
        if (shared == null || shared.expiresAtMillis < System.currentTimeMillis() || !shared.file.isFile) {
            socket.sendHttp(404, "Not Found", "text/plain", "Not found".toByteArray(StandardCharsets.UTF_8))
            return
        }
        val body = if (method == "HEAD") null else shared.file.readBytes()
        socket.sendHttp(
            statusCode = 200,
            reason = "OK",
            mimeType = contentType(shared.mimeType),
            body = body,
            contentLength = shared.file.length(),
        )
    }

    private fun tokenFromPath(rawPath: String): String? {
        val path = rawPath.substringBefore("?")
        if (!path.startsWith("/files/")) return null
        return URLDecoder.decode(path.removePrefix("/files/").substringBefore("/"), StandardCharsets.UTF_8.name())
            .takeIf { it.isNotBlank() }
    }

    private fun Socket.sendHttp(
        statusCode: Int,
        reason: String,
        mimeType: String,
        body: ByteArray?,
        contentLength: Long = body?.size?.toLong() ?: 0L,
    ) {
        val header = buildString {
            append("HTTP/1.1 ").append(statusCode).append(' ').append(reason).append("\r\n")
            append("Content-Type: ").append(mimeType).append("\r\n")
            append("Content-Length: ").append(contentLength).append("\r\n")
            append("Cache-Control: no-store\r\n")
            append("Connection: close\r\n")
            append("\r\n")
        }
        val output = getOutputStream()
        output.write(header.toByteArray(StandardCharsets.US_ASCII))
        if (body != null) {
            output.write(body)
        }
        output.flush()
    }

    private fun contentType(mimeType: String): String {
        return when {
            mimeType.startsWith("text/") -> "$mimeType; charset=utf-8"
            mimeType == "application/javascript" || mimeType == "application/json" -> "$mimeType; charset=utf-8"
            else -> mimeType
        }
    }

    private fun pruneExpired() {
        val now = System.currentTimeMillis()
        sharedFiles.entries.removeIf { it.value.expiresAtMillis < now }
    }

    private data class SharedFile(
        val file: File,
        val mimeType: String,
        val expiresAtMillis: Long,
    )

    private data class RunningServer(
        val port: Int,
        val socket: ServerSocket,
    )
}
