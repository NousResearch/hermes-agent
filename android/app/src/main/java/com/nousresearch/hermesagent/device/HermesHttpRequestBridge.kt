package com.nousresearch.hermesagent.device

import android.content.Context
import com.nousresearch.hermesagent.data.HermesNetworkPolicy
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.Locale
import java.util.concurrent.TimeUnit

object HermesHttpRequestBridge {
    fun payloadFromArguments(arguments: JSONObject, allowVariableUrl: Boolean = true): JSONObject {
        val method = normalizeMethod(firstString(arguments, "http_method", "method", "request_method", "verb").orEmpty())
            ?: DEFAULT_METHOD
        val url = firstString(arguments, "url", "uri", "endpoint", "request_url", "command")?.trim()
            ?: throw IllegalArgumentException("http_request requires a url")
        rejectNul(url, "url")
        require(url.length <= MAX_URL_CHARS) { "url must be $MAX_URL_CHARS characters or shorter" }
        if (!allowVariableUrl || !looksLikeVariableReference(url)) {
            require(isHttpUrl(url)) { "http_request url must start with http:// or https://" }
        }

        val payload = JSONObject()
            .put("method", method)
            .put("url", url)

        firstString(arguments, "body", "data", "request_body", "payload", allowEmpty = true)?.let { body ->
            rejectNul(body, "body")
            require(body.length <= MAX_REQUEST_BODY_CHARS) {
                "body must be $MAX_REQUEST_BODY_CHARS characters or shorter"
            }
            payload.put("body", body)
        }
        firstString(arguments, "content_type", "contentType", "mime_type")?.let { contentType ->
            rejectNul(contentType, "content_type")
            require(contentType.length <= MAX_HEADER_VALUE_CHARS) {
                "content_type must be $MAX_HEADER_VALUE_CHARS characters or shorter"
            }
            payload.put("content_type", contentType)
        }
        headersFromArguments(arguments)?.let { payload.put("headers", it) }
        timeoutSeconds(arguments)?.let { payload.put("timeout_seconds", it) }
        firstString(arguments, "save_response_variable", "response_variable", "body_variable")?.let { payload.put("save_response_variable", it) }
        firstString(arguments, "save_status_variable", "status_variable", "code_variable")?.let { payload.put("save_status_variable", it) }
        return payload
    }

    fun performHttpRequestJson(context: Context, payload: JSONObject): JSONObject {
        val method = normalizeMethod(payload.optString("method"))
            ?: return errorJson("Unsupported HTTP method: ${payload.optString("method")}")
        val url = payload.optString("url").trim()
        if (!isHttpUrl(url)) {
            return errorJson("http_request url must start with http:// or https://")
        }
        if (HermesNetworkPolicy.isExternalNetworkBlocked(context, url)) {
            return errorJson(HermesNetworkPolicy.offlineBlockedMessage("HTTP automation request"))
                .put("method", method)
                .put("url", url)
        }
        if (url.length > MAX_URL_CHARS || url.indexOf('\u0000') >= 0) {
            return errorJson("http_request url is invalid")
        }
        val bodyText = payload.optString("body", "")
        if (bodyText.length > MAX_REQUEST_BODY_CHARS || bodyText.indexOf('\u0000') >= 0) {
            return errorJson("http_request body is invalid")
        }
        val headers = runCatching { headersFromPayload(payload) }.getOrElse { error ->
            return errorJson(error.message ?: "HTTP headers are invalid")
        }
        val timeoutSeconds = payload.optInt("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
            .coerceIn(MIN_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS)
        val requestBuilder = Request.Builder().url(url)
        headers.keys().forEach { name ->
            requestBuilder.header(name, headers.optString(name))
        }
        val requestBody = when {
            method in METHODS_WITH_OPTIONAL_BODY && (bodyText.isNotEmpty() || method in METHODS_REQUIRING_BODY) -> {
                val contentType = payload.optString("content_type").ifBlank { DEFAULT_CONTENT_TYPE }
                bodyText.toRequestBody(contentType.toMediaType())
            }
            else -> null
        }
        requestBuilder.method(method, requestBody)
        val client = BASE_CLIENT.newBuilder()
            .callTimeout(timeoutSeconds.toLong(), TimeUnit.SECONDS)
            .connectTimeout(timeoutSeconds.toLong(), TimeUnit.SECONDS)
            .readTimeout(timeoutSeconds.toLong(), TimeUnit.SECONDS)
            .build()
        return try {
            client.newCall(requestBuilder.build()).execute().use { response ->
                val responseBody = response.body?.string().orEmpty()
                val truncatedBody = responseBody.take(MAX_RESPONSE_BODY_CHARS)
                val success = response.code in 200..399
                JSONObject()
                    .put("success", success)
                    .put("exit_code", if (success) 0 else response.code)
                    .put("action", "http_request")
                    .put("method", method)
                    .put("url", url)
                    .put("status_code", response.code)
                    .put("http_success", success)
                    .put("body", truncatedBody)
                    .put("body_truncated", responseBody.length > MAX_RESPONSE_BODY_CHARS)
                    .put("content_type", response.header("Content-Type").orEmpty())
                    .put("headers", responseHeadersJson(response))
                    .put("message", "HTTP $method returned ${response.code}")
            }
        } catch (error: IOException) {
            errorJson(error.message ?: error.javaClass.simpleName)
                .put("method", method)
                .put("url", url)
        } catch (error: IllegalArgumentException) {
            errorJson(error.message ?: error.javaClass.simpleName)
                .put("method", method)
                .put("url", url)
        }
    }

    fun normalizeMethod(value: String): String? {
        val normalized = value.trim().uppercase(Locale.US)
        if (normalized.isBlank()) {
            return null
        }
        return normalized.takeIf { it in SUPPORTED_METHODS }
    }

    fun isHttpUrl(value: String): Boolean {
        val lower = value.trim().lowercase(Locale.US)
        return lower.startsWith("http://") || lower.startsWith("https://")
    }

    private fun headersFromArguments(arguments: JSONObject): JSONObject? {
        val rawHeaders = arguments.optJSONObject("headers")
            ?: arguments.optJSONObject("request_headers")
            ?: JSONObject()
        val headerName = firstString(arguments, "header_name", "request_header_name")
        val headerValue = firstString(arguments, "header_value", "request_header_value", allowEmpty = true)
        if (!headerName.isNullOrBlank()) {
            rawHeaders.put(headerName, headerValue.orEmpty())
        }
        val headers = JSONObject()
        rawHeaders.keys().forEach { rawName ->
            val name = rawName.trim()
            val value = rawHeaders.optString(rawName)
            rejectHeader(name, value)
            headers.put(name, value.take(MAX_HEADER_VALUE_CHARS))
        }
        return headers.takeIf { it.length() > 0 }
    }

    private fun headersFromPayload(payload: JSONObject): JSONObject {
        val raw = payload.optJSONObject("headers") ?: return JSONObject()
        val headers = JSONObject()
        raw.keys().forEach { rawName ->
            val name = rawName.trim()
            val value = raw.optString(rawName)
            rejectHeader(name, value)
            headers.put(name, value.take(MAX_HEADER_VALUE_CHARS))
        }
        return headers
    }

    private fun responseHeadersJson(response: okhttp3.Response): JSONObject {
        val output = JSONObject()
        response.headers.names().take(MAX_RESPONSE_HEADERS).forEach { name ->
            output.put(name, response.headers.values(name).joinToString(", ").take(MAX_HEADER_VALUE_CHARS))
        }
        return output
    }

    private fun timeoutSeconds(arguments: JSONObject): Int? {
        val raw = firstString(arguments, "timeout_seconds", "timeout", "read_timeout_seconds") ?: return null
        val value = raw.trim().toIntOrNull() ?: throw IllegalArgumentException("timeout_seconds must be an integer")
        require(value in MIN_TIMEOUT_SECONDS..MAX_TIMEOUT_SECONDS) {
            "timeout_seconds must be between $MIN_TIMEOUT_SECONDS and $MAX_TIMEOUT_SECONDS"
        }
        return value
    }

    private fun firstString(arguments: JSONObject, vararg keys: String, allowEmpty: Boolean = false): String? {
        return keys.firstNotNullOfOrNull { key ->
            if (!arguments.has(key) || arguments.isNull(key)) {
                null
            } else {
                arguments.optString(key).takeIf { allowEmpty || it.isNotBlank() }
            }
        }
    }

    private fun rejectHeader(name: String, value: String) {
        require(name.isNotBlank() && name.length <= MAX_HEADER_NAME_CHARS) { "HTTP header name is invalid" }
        require(HEADER_NAME_REGEX.matches(name)) { "HTTP header name is invalid: $name" }
        rejectNul(value, "HTTP header value")
        require('\n' !in value && '\r' !in value) { "HTTP header value must not contain newlines" }
    }

    private fun rejectNul(value: String, label: String) {
        require(value.indexOf('\u0000') < 0) { "$label must not contain NUL bytes" }
    }

    private fun looksLikeVariableReference(value: String): Boolean {
        return value.contains('%') || value.contains("{{")
    }

    private fun errorJson(message: String): JSONObject {
        return JSONObject()
            .put("success", false)
            .put("exit_code", 1)
            .put("action", "http_request")
            .put("error", message)
    }

    private val BASE_CLIENT = OkHttpClient.Builder()
        .followRedirects(true)
        .followSslRedirects(true)
        .build()
    private val SUPPORTED_METHODS = setOf("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE")
    private val METHODS_WITH_OPTIONAL_BODY = setOf("POST", "PUT", "PATCH", "DELETE")
    private val METHODS_REQUIRING_BODY = setOf("POST", "PUT", "PATCH")
    private val HEADER_NAME_REGEX = Regex("[A-Za-z0-9!#$%&'*+.^_`|~-]+")
    private const val DEFAULT_METHOD = "GET"
    private const val DEFAULT_CONTENT_TYPE = "text/plain; charset=utf-8"
    private const val MIN_TIMEOUT_SECONDS = 1
    private const val DEFAULT_TIMEOUT_SECONDS = 15
    private const val MAX_TIMEOUT_SECONDS = 60
    private const val MAX_URL_CHARS = 2_000
    private const val MAX_REQUEST_BODY_CHARS = 131_072
    private const val MAX_RESPONSE_BODY_CHARS = 65_536
    private const val MAX_HEADER_NAME_CHARS = 80
    private const val MAX_HEADER_VALUE_CHARS = 1_000
    private const val MAX_RESPONSE_HEADERS = 64
}
