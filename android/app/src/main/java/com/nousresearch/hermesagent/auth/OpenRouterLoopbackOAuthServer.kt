package com.nousresearch.hermesagent.auth

import android.content.Context
import android.net.Uri
import com.nousresearch.hermesagent.data.AuthSession
import com.nousresearch.hermesagent.data.AuthSessionStore
import com.nousresearch.hermesagent.data.PendingAuthRequest
import com.nousresearch.hermesagent.device.DeviceStateWriter
import java.io.BufferedReader
import java.io.OutputStreamWriter
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.net.SocketTimeoutException
import kotlin.concurrent.thread

data class OpenRouterLoopbackServerStart(
    val started: Boolean,
    val callbackUrl: String = "",
    val errorName: String = "",
    val handle: OpenRouterLoopbackOAuthServer.Handle? = null,
)

object OpenRouterLoopbackOAuthServer {
    const val DEFAULT_PORT = 3000
    private const val CALLBACK_HOST = "localhost"
    private const val CALLBACK_PATH = "/hermes/openrouter/callback"
    private const val SERVER_TIMEOUT_MS = 5 * 60 * 1000

    @Volatile
    private var currentHandle: Handle? = null

    fun callbackUrlForState(state: String, port: Int = DEFAULT_PORT): String {
        return Uri.Builder()
            .scheme("http")
            .encodedAuthority("$CALLBACK_HOST:$port")
            .path(CALLBACK_PATH)
            .appendQueryParameter("method", "openrouter")
            .appendQueryParameter("provider", "openrouter")
            .appendQueryParameter("state", state)
            .build()
            .toString()
    }

    fun start(
        context: Context,
        pending: PendingAuthRequest,
        port: Int = DEFAULT_PORT,
        exchangeUrl: String = OpenRouterOAuthClient.DEFAULT_EXCHANGE_URL,
    ): OpenRouterLoopbackServerStart {
        return start(
            port = port,
            state = pending.state,
            callbackHandler = { callbackUri ->
                handleOpenRouterCallback(
                    context = context.applicationContext,
                    pending = pending,
                    callbackUri = callbackUri,
                    exchangeUrl = exchangeUrl,
                )
            },
        )
    }

    internal fun start(
        port: Int,
        state: String,
        callbackHandler: (Uri) -> AuthSession,
    ): OpenRouterLoopbackServerStart {
        stopCurrent()
        return try {
            val socket = ServerSocket(port).apply {
                soTimeout = SERVER_TIMEOUT_MS
            }
            val actualPort = socket.localPort
            val callbackUrl = callbackUrlForState(state, actualPort)
            val handle = Handle(socket, callbackUrl)
            currentHandle = handle
            handle.worker = thread(
                name = "HermesOpenRouterLoopbackOAuth",
                isDaemon = true,
            ) {
                runServer(handle, actualPort, callbackHandler)
            }
            OpenRouterLoopbackServerStart(
                started = true,
                callbackUrl = callbackUrl,
                handle = handle,
            )
        } catch (error: Exception) {
            OpenRouterLoopbackServerStart(
                started = false,
                callbackUrl = callbackUrlForState(state, port),
                errorName = error.javaClass.simpleName,
            )
        }
    }

    fun stopCurrent() {
        currentHandle?.stop()
        currentHandle = null
    }

    private fun handleOpenRouterCallback(
        context: Context,
        pending: PendingAuthRequest,
        callbackUri: Uri,
        exchangeUrl: String,
    ): AuthSession {
        val store = AuthSessionStore(context)
        val session = OpenRouterOAuthClient.exchangeCallbackForSession(
            uri = callbackUri,
            pending = pending,
            exchangeUrl = exchangeUrl,
        )
        store.clearPendingRequest()
        store.saveSession(session)
        if (session.signedIn) {
            AuthRuntimeApplier.apply(context, session)
        }
        DeviceStateWriter.write(context)
        return session
    }

    private fun runServer(
        handle: Handle,
        port: Int,
        callbackHandler: (Uri) -> AuthSession,
    ) {
        try {
            handle.serverSocket.accept().use { client ->
                serveClient(client, port, callbackHandler)
            }
        } catch (_: SocketTimeoutException) {
            // The pending browser auth window expired without a callback.
        } catch (_: SocketException) {
            // Stopping the handle closes the socket and interrupts accept.
        } finally {
            handle.stop()
            if (currentHandle === handle) {
                currentHandle = null
            }
        }
    }

    private fun serveClient(
        client: Socket,
        port: Int,
        callbackHandler: (Uri) -> AuthSession,
    ) {
        val reader = BufferedReader(client.getInputStream().reader(Charsets.UTF_8))
        val requestLine = reader.readLine().orEmpty()
        while (reader.readLine()?.isNotBlank() == true) {
            // Drain headers so the browser can receive the response cleanly.
        }
        val target = requestLine.split(" ").getOrNull(1).orEmpty()
        val callbackUri = parseRequestTarget(target, port)
        if (callbackUri.path != CALLBACK_PATH) {
            writeResponse(
                client = client,
                status = "404 Not Found",
                body = htmlPage("Hermes sign-in", "This local Hermes sign-in route was not found."),
            )
            return
        }

        val session = runCatching { callbackHandler(callbackUri) }.getOrElse { error ->
            writeResponse(
                client = client,
                status = "500 Internal Server Error",
                body = htmlPage(
                    "Hermes sign-in failed",
                    "Hermes received the OpenRouter callback, but could not save it (${error.javaClass.simpleName}). Return to Hermes and try again, or paste an API key in Settings.",
                ),
            )
            return
        }
        val title = if (session.signedIn) "Hermes sign-in complete" else "Hermes sign-in failed"
        val message = if (session.signedIn) {
            "OpenRouter is connected. You can return to Hermes."
        } else {
            session.status.ifBlank { "OpenRouter did not return a usable API key. Return to Hermes and try again." }
        }
        writeResponse(client, status = "200 OK", body = htmlPage(title, message))
    }

    private fun parseRequestTarget(target: String, port: Int): Uri {
        val normalizedTarget = target.ifBlank { "/" }
        return if (normalizedTarget.startsWith("http://") || normalizedTarget.startsWith("https://")) {
            Uri.parse(normalizedTarget)
        } else {
            Uri.parse("http://$CALLBACK_HOST:$port$normalizedTarget")
        }
    }

    private fun writeResponse(client: Socket, status: String, body: String) {
        val bytes = body.toByteArray(Charsets.UTF_8)
        OutputStreamWriter(client.getOutputStream(), Charsets.UTF_8).use { writer ->
            writer.write("HTTP/1.1 $status\r\n")
            writer.write("Content-Type: text/html; charset=utf-8\r\n")
            writer.write("Cache-Control: no-store\r\n")
            writer.write("Connection: close\r\n")
            writer.write("Content-Length: ${bytes.size}\r\n")
            writer.write("\r\n")
            writer.write(body)
            writer.flush()
        }
    }

    private fun htmlPage(title: String, message: String): String {
        return """
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>${title.escapeHtml()}</title>
              <style>
                body { font-family: sans-serif; margin: 2rem; line-height: 1.45; color: #10131f; }
                main { max-width: 32rem; margin: 0 auto; }
              </style>
            </head>
            <body>
              <main>
                <h1>${title.escapeHtml()}</h1>
                <p>${message.escapeHtml()}</p>
              </main>
            </body>
            </html>
        """.trimIndent()
    }

    private fun String.escapeHtml(): String {
        return replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("'", "&#39;")
    }

    class Handle internal constructor(
        internal val serverSocket: ServerSocket,
        val callbackUrl: String,
    ) {
        internal var worker: Thread? = null

        fun stop() {
            runCatching { serverSocket.close() }
        }
    }
}
