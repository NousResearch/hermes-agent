package com.mobilefork.hermesagent.device

import android.content.Context
import android.os.Build
import android.os.Process
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlin.system.exitProcess

object HermesCrashLogStore {
    @Volatile
    private var installed = false

    @Volatile
    private var previousHandler: Thread.UncaughtExceptionHandler? = null

    fun install(context: Context) {
        val appContext = context.applicationContext
        synchronized(this) {
            if (installed) return
            val current = Thread.getDefaultUncaughtExceptionHandler()
            if (current is HermesCrashHandler) {
                installed = true
                return
            }
            previousHandler = current
            Thread.setDefaultUncaughtExceptionHandler(HermesCrashHandler(appContext, current))
            installed = true
        }
        Thread(
            { appendDiagnosticEvent(appContext, "info", "crash_capture_armed") },
            "HermesCrashDiagnosticsInit",
        ).apply { isDaemon = true }.start()
    }

    fun statusSnapshot(context: Context): CrashLogSnapshot {
        val appContext = context.applicationContext
        val crashFile = lastCrashFile(appContext)
        val logFile = diagnosticsLogFile(appContext)
        val crash = readLastCrashJson(appContext)
        val capturedAtLabel = crash?.optString("captured_at").orEmpty()
        val exceptionType = crash?.optString("exception_type").orEmpty()
        val message = crash?.optString("message").orEmpty()
        val stackTrace = crash?.optString("stack_trace").orEmpty()
        val previewLines = buildList {
            if (exceptionType.isNotBlank()) add(exceptionType)
            if (message.isNotBlank()) add(message)
            stackTrace.lineSequence()
                .firstOrNull { it.trimStart().startsWith("at ") || it.trimStart().startsWith("Caused by:") }
                ?.let { add(it.trim()) }
        }.map { redactForDiagnostics(it).text }.take(MAX_PREVIEW_LINES)
        val statusLabel = if (crash != null) {
            val suffix = if (exceptionType.isBlank()) "" else " ($exceptionType)"
            "Last crash captured ${capturedAtLabel.ifBlank { "recently" }}$suffix"
        } else {
            "No crash captured"
        }
        return CrashLogSnapshot(
            hasLastCrash = crash != null,
            statusLabel = statusLabel,
            capturedAtLabel = capturedAtLabel,
            exceptionType = exceptionType,
            previewLines = previewLines,
            exportFileName = "hermes-diagnostics-logs.txt",
            logBytes = if (logFile.isFile) logFile.length() else 0L,
            lastCrashBytes = if (crashFile.isFile) crashFile.length() else 0L,
            captureInstalled = installed,
        )
    }

    fun statusJson(context: Context): JSONObject {
        val snapshot = statusSnapshot(context)
        return JSONObject()
            .put("success", true)
            .put("action", "crash_log_status")
            .put("capture_installed", snapshot.captureInstalled)
            .put("last_crash_present", snapshot.hasLastCrash)
            .put("status_label", snapshot.statusLabel)
            .put("captured_at", snapshot.capturedAtLabel)
            .put("exception_type", snapshot.exceptionType)
            .put("display_preview", JSONArray(snapshot.previewLines))
            .put("diagnostics_log_bytes", snapshot.logBytes)
            .put("last_crash_bytes", snapshot.lastCrashBytes)
            .put("export_action", "diagnostics_log_export")
            .put("suggested_export_file_name", snapshot.exportFileName)
            .put("pii_filter", redactionPolicyJson())
    }

    fun exportJson(context: Context): JSONObject {
        val exportText = exportLogsText(context)
        return JSONObject()
            .put("success", true)
            .put("action", "diagnostics_log_export")
            .put("suggested_file_name", "hermes-diagnostics-logs.txt")
            .put("mime_type", "text/plain")
            .put("content", exportText)
            .put("content_bytes", exportText.toByteArray(Charsets.UTF_8).size)
            .put("pii_filter", redactionPolicyJson())
    }

    fun exportLogsText(context: Context): String {
        val appContext = context.applicationContext
        val snapshot = statusSnapshot(appContext)
        val builder = StringBuilder()
            .appendLine("Hermes diagnostics logs")
            .appendLine("Generated: ${formatTimestamp(System.currentTimeMillis())}")
            .appendLine("Crash capture installed: ${snapshot.captureInstalled}")
            .appendLine("PII filter: emails, bearer/API tokens, phone-like numbers, and obvious user paths are redacted.")
            .appendLine()
            .appendLine("Status: ${snapshot.statusLabel}")

        readLastCrashJson(appContext)?.let { crash ->
            builder
                .appendLine()
                .appendLine("Last crash")
                .appendLine(crash.toString(2))
        }

        val logFile = diagnosticsLogFile(appContext)
        if (logFile.isFile) {
            builder
                .appendLine()
                .appendLine("Recent diagnostic events")
                .appendLine(logFile.readText().takeLast(MAX_EXPORT_LOG_CHARS))
        }

        return redactForDiagnostics(builder.toString()).text
    }

    fun clearLastCrash(context: Context) {
        lastCrashFile(context.applicationContext).delete()
        appendDiagnosticEvent(context.applicationContext, "info", "last_crash_cleared")
    }

    internal fun recordCrashForTest(
        context: Context,
        throwable: Throwable,
        threadName: String = "test-thread",
        nowMs: Long = System.currentTimeMillis(),
    ): JSONObject {
        return recordCrash(context.applicationContext, threadName, throwable, nowMs)
    }

    internal fun clearAllForTest(context: Context) {
        val appContext = context.applicationContext
        lastCrashFile(appContext).delete()
        diagnosticsLogFile(appContext).delete()
    }

    private fun recordCrash(context: Context, threadName: String, throwable: Throwable, nowMs: Long): JSONObject {
        val rawStackTrace = stackTraceString(throwable).take(MAX_STACK_TRACE_CHARS)
        val redactedMessage = redactForDiagnostics(throwable.message.orEmpty()).text
        val redactedStackTrace = redactForDiagnostics(rawStackTrace).text
        val redactedThreadName = redactForDiagnostics(threadName).text
        val crash = JSONObject()
            .put("kind", "uncaught_exception")
            .put("captured_at_ms", nowMs)
            .put("captured_at", formatTimestamp(nowMs))
            .put("thread", redactedThreadName)
            .put("exception_type", throwable.javaClass.name)
            .put("message", redactedMessage)
            .put("stack_trace", redactedStackTrace)
            .put(
                "android",
                JSONObject()
                    .put("sdk_int", Build.VERSION.SDK_INT)
                    .put("release", Build.VERSION.RELEASE.orEmpty())
                    .put("manufacturer", Build.MANUFACTURER.orEmpty())
                    .put("model", Build.MODEL.orEmpty())
                    .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList())),
            )
            .put("pii_filter", redactionPolicyJson())

        val dir = diagnosticsDir(context)
        dir.mkdirs()
        lastCrashFile(context).writeText(crash.toString(2), Charsets.UTF_8)
        appendDiagnosticEvent(context, "error", "uncaught_exception", crash)
        return crash
    }

    private fun appendDiagnosticEvent(
        context: Context,
        level: String,
        message: String,
        payload: JSONObject? = null,
        nowMs: Long = System.currentTimeMillis(),
    ) {
        runCatching {
            val event = JSONObject()
                .put("timestamp_ms", nowMs)
                .put("timestamp", formatTimestamp(nowMs))
                .put("level", level)
                .put("message", redactForDiagnostics(message).text)
            if (payload != null) {
                event.put("payload", JSONObject(redactForDiagnostics(payload.toString()).text))
            }

            val logFile = diagnosticsLogFile(context)
            logFile.parentFile?.mkdirs()
            val existing = if (logFile.isFile) logFile.readText(Charsets.UTF_8) else ""
            val combined = (existing + event.toString() + "\n").takeLast(MAX_LOG_BYTES)
            logFile.writeText(combined, Charsets.UTF_8)
        }
    }

    private fun readLastCrashJson(context: Context): JSONObject? {
        val file = lastCrashFile(context.applicationContext)
        if (!file.isFile) return null
        return runCatching {
            JSONObject(redactForDiagnostics(file.readText(Charsets.UTF_8)).text)
        }.getOrNull()
    }

    private fun diagnosticsDir(context: Context): File = File(context.filesDir, DIAGNOSTICS_DIR)

    private fun lastCrashFile(context: Context): File = File(diagnosticsDir(context), LAST_CRASH_FILE)

    private fun diagnosticsLogFile(context: Context): File = File(diagnosticsDir(context), DIAGNOSTICS_LOG_FILE)

    private fun stackTraceString(throwable: Throwable): String {
        val writer = StringWriter()
        throwable.printStackTrace(PrintWriter(writer))
        return writer.toString()
    }

    private fun redactionPolicyJson(): JSONObject {
        return JSONObject()
            .put("redacted_before_write", true)
            .put("redacted_before_display", true)
            .put("redacted_before_export", true)
            .put(
                "patterns",
                JSONArray()
                    .put("email")
                    .put("authorization_bearer")
                    .put("api_key_or_token_assignment")
                    .put("common_secret_token")
                    .put("phone_like_number")
                    .put("obvious_user_path"),
            )
    }

    internal fun redactForDiagnostics(value: String): RedactionResult {
        var text = value
        val counts = linkedMapOf<String, Int>()

        fun apply(key: String, regex: Regex, replacement: (MatchResult) -> String) {
            var count = 0
            text = regex.replace(text) { match ->
                count += 1
                replacement(match)
            }
            if (count > 0) {
                counts[key] = (counts[key] ?: 0) + count
            }
        }

        apply("authorization_bearer", AUTHORIZATION_BEARER_REGEX) {
            "${it.groupValues[1]}[REDACTED_BEARER]"
        }
        apply("json_secret", JSON_SECRET_REGEX) {
            "${it.groupValues[1]}[REDACTED_SECRET]${it.groupValues[3]}"
        }
        apply("assignment_secret", ASSIGNMENT_SECRET_REGEX) {
            "${it.groupValues[1]}[REDACTED_SECRET]"
        }
        apply("bearer_token", STANDALONE_BEARER_REGEX) {
            "Bearer [REDACTED_BEARER]"
        }
        apply("jwt", JWT_REGEX) { "[REDACTED_TOKEN]" }
        apply("openai_style_key", OPENAI_STYLE_KEY_REGEX) { "[REDACTED_API_KEY]" }
        apply("google_style_key", GOOGLE_STYLE_KEY_REGEX) { "[REDACTED_API_KEY]" }
        apply("github_style_token", GITHUB_STYLE_TOKEN_REGEX) { "[REDACTED_TOKEN]" }
        apply("email", EMAIL_REGEX) { "[REDACTED_EMAIL]" }
        apply("phone_context", PHONE_CONTEXT_REGEX) {
            "${it.groupValues[1]}[REDACTED_PHONE]"
        }
        apply("phone_like", E164_PHONE_REGEX) { "[REDACTED_PHONE]" }
        apply("phone_like", NORTH_AMERICAN_PHONE_REGEX) { "[REDACTED_PHONE]" }
        apply("windows_user_path", WINDOWS_USER_PATH_REGEX) {
            "${it.groupValues[1]}[REDACTED_USER]"
        }
        apply("mac_user_path", MAC_USER_PATH_REGEX) {
            "${it.groupValues[1]}[REDACTED_USER]"
        }
        apply("linux_user_path", LINUX_USER_PATH_REGEX) {
            "${it.groupValues[1]}[REDACTED_USER]"
        }
        apply("shared_storage_path", SHARED_STORAGE_PATH_REGEX) {
            "${it.groupValues[1]}/[REDACTED_PATH]"
        }

        return RedactionResult(text = text, counts = counts)
    }

    private fun formatTimestamp(timestampMs: Long): String {
        return SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }.format(Date(timestampMs))
    }

    data class CrashLogSnapshot(
        val hasLastCrash: Boolean,
        val statusLabel: String,
        val capturedAtLabel: String,
        val exceptionType: String,
        val previewLines: List<String>,
        val exportFileName: String,
        val logBytes: Long,
        val lastCrashBytes: Long,
        val captureInstalled: Boolean,
    )

    data class RedactionResult(
        val text: String,
        val counts: Map<String, Int>,
    )

    private class HermesCrashHandler(
        private val context: Context,
        private val delegate: Thread.UncaughtExceptionHandler?,
    ) : Thread.UncaughtExceptionHandler {
        override fun uncaughtException(thread: Thread, throwable: Throwable) {
            runCatching {
                recordCrash(context, thread.name.orEmpty(), throwable, System.currentTimeMillis())
            }
            val downstream = delegate
            if (downstream != null) {
                downstream.uncaughtException(thread, throwable)
            } else {
                Process.killProcess(Process.myPid())
                exitProcess(10)
            }
        }
    }

    private const val DIAGNOSTICS_DIR = "hermes-diagnostics"
    private const val LAST_CRASH_FILE = "last-crash.json"
    private const val DIAGNOSTICS_LOG_FILE = "diagnostics-log.jsonl"
    private const val MAX_STACK_TRACE_CHARS = 24_000
    private const val MAX_LOG_BYTES = 128 * 1024
    private const val MAX_EXPORT_LOG_CHARS = 96 * 1024
    private const val MAX_PREVIEW_LINES = 4

    private val EMAIL_REGEX = Regex("""(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b""")
    private val AUTHORIZATION_BEARER_REGEX = Regex("""(?i)\b(authorization\s*[:=]\s*bearer\s+)([^\s"',;\\]+)""")
    private val STANDALONE_BEARER_REGEX = Regex("""(?i)\bBearer\s+([A-Za-z0-9._~+/=-]{8,})""")
    private val JSON_SECRET_REGEX = Regex(
        """(?i)("(?:api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?token|auth[_-]?token|token|secret|password|authorization)"\s*:\s*")([^"]*)(")""",
    )
    private val ASSIGNMENT_SECRET_REGEX = Regex(
        """(?i)\b((?:api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?token|auth[_-]?token|bearer[_-]?token|password|secret|client[_-]?secret|[A-Z0-9_]+_API_KEY|[A-Z0-9_]+_TOKEN)\s*[:=]\s*['"]?)([^\s'",;]+)""",
    )
    private val OPENAI_STYLE_KEY_REGEX = Regex("""\bsk-(?:or-v1-)?[A-Za-z0-9_-]{8,}\b""")
    private val GOOGLE_STYLE_KEY_REGEX = Regex("""\bAIza[0-9A-Za-z_-]{20,}\b""")
    private val GITHUB_STYLE_TOKEN_REGEX = Regex("""\bgh[pousr]_[A-Za-z0-9_]{20,}\b""")
    private val JWT_REGEX = Regex("""\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b""")
    private val PHONE_CONTEXT_REGEX = Regex("""(?i)\b((?:phone|mobile|tel|sms)\s*[:=]\s*)\+?[0-9][0-9 .()/-]{7,}[0-9]""")
    private val E164_PHONE_REGEX = Regex("""(?<![A-Za-z0-9])\+\d[\d .()-]{8,}\d(?![A-Za-z0-9])""")
    private val NORTH_AMERICAN_PHONE_REGEX = Regex("""(?<![A-Za-z0-9])(?:\+?1[\s.-]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[\s.-]?[2-9]\d{2}[\s.-]?\d{4}(?![A-Za-z0-9])""")
    private val WINDOWS_USER_PATH_REGEX = Regex("""(?i)\b([A-Z]:\\Users\\)([^\\\r\n]+)""")
    private val MAC_USER_PATH_REGEX = Regex("""(?i)(/Users/)([^/\s"']+)""")
    private val LINUX_USER_PATH_REGEX = Regex("""(?i)(/home/)([^/\s"']+)""")
    private val SHARED_STORAGE_PATH_REGEX = Regex("""(?i)(/storage/emulated/\d+/(?:Download|Documents|Pictures|Movies|Music|DCIM))(/[^\s"']*)?""")
}
