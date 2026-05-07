package com.nousresearch.hermesagent.device

import org.json.JSONArray
import org.json.JSONObject
import org.w3c.dom.Element
import org.w3c.dom.Node
import java.io.ByteArrayInputStream
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.util.Base64
import java.util.UUID
import javax.xml.parsers.DocumentBuilderFactory

data class HermesTaskerImportResult(
    val bundle: JSONObject,
    val taskCount: Int,
    val importedActionCount: Int,
    val skippedActions: JSONArray,
)

object HermesTaskerImportBridge {
    fun bundleFromArguments(arguments: JSONObject): HermesTaskerImportResult? {
        val raw = firstString(
            arguments,
            "tasker_xml",
            "tasker_data",
            "tasker_export",
            "tasker_bundle",
            "tasker_project_xml",
            "tasker_task_xml",
        ) ?: firstString(arguments, "tasker_data_uri", "tasker_uri", "data_uri")
            ?.let(::decodeTaskerDataUri)
            ?: firstString(arguments, "tasker_xml_base64", "tasker_base64")
                ?.let(::decodeBase64Text)
            ?: return null
        return parse(raw)
    }

    fun parse(rawInput: String): HermesTaskerImportResult {
        val xml = rawInput.trim()
        require(xml.startsWith("<")) { "Tasker import expects raw XML, a data:text/xml URI, or base64 XML" }
        require(xml.length <= MAX_TASKER_XML_CHARS) { "Tasker XML import is limited to $MAX_TASKER_XML_CHARS characters" }
        require(xml.indexOf('\u0000') < 0) { "Tasker XML cannot contain NUL bytes" }
        require(!DOCTYPE_REGEX.containsMatchIn(xml)) { "Tasker XML cannot include DOCTYPE declarations" }
        require(!ENTITY_REGEX.containsMatchIn(xml)) { "Tasker XML cannot include ENTITY declarations" }
        val builderFactory = DocumentBuilderFactory.newInstance().apply {
            isNamespaceAware = false
            runCatching { isXIncludeAware = false }
            runCatching { setExpandEntityReferences(false) }
            runCatching { setFeature("http://apache.org/xml/features/disallow-doctype-decl", true) }
            runCatching { setFeature("http://xml.org/sax/features/external-general-entities", false) }
            runCatching { setFeature("http://xml.org/sax/features/external-parameter-entities", false) }
        }
        val document = builderFactory.newDocumentBuilder().parse(
            ByteArrayInputStream(xml.toByteArray(StandardCharsets.UTF_8)),
        )
        val records = JSONArray()
        val skipped = JSONArray()
        val taskElements = document.getElementsByTagName("Task")
        var importedActions = 0
        for (taskIndex in 0 until taskElements.length) {
            val task = taskElements.item(taskIndex) as? Element ?: continue
            val taskName = directText(task, "nme")
                .ifBlank { task.getAttribute("sr") }
                .ifBlank { "Tasker task ${taskIndex + 1}" }
                .take(MAX_LABEL_CHARS)
            val actions = directChildren(task, "Action")
            actions.forEachIndexed { actionIndex, action ->
                val code = directText(action, "code").trim().toIntOrNull()
                if (code == null) {
                    skipped.put(skippedAction(taskName, actionIndex, "missing_code", "Tasker action has no numeric code"))
                    return@forEachIndexed
                }
                val record = recordFromTaskerAction(taskName, taskIndex, actionIndex, code, action)
                if (record == null) {
                    skipped.put(
                        skippedAction(
                            taskName = taskName,
                            actionIndex = actionIndex,
                            code = code.toString(),
                            reason = "Unsupported or unsafe Tasker action code",
                        ),
                    )
                    return@forEachIndexed
                }
                records.put(record)
                importedActions += 1
            }
        }
        val variables = parseVariables(document.documentElement)
        require(records.length() > 0 || variables.length() > 0) {
            "Tasker XML did not contain any supported safe actions or variables"
        }
        return HermesTaskerImportResult(
            bundle = JSONObject()
                .put("kind", "hermes_android_automation_bundle")
                .put("schema_version", 1)
                .put("source", "tasker_xml")
                .put("automations", records)
                .put("variables", variables)
                .put("tasker_task_count", taskElements.length)
                .put("tasker_imported_action_count", importedActions)
                .put("tasker_skipped_actions", skipped),
            taskCount = taskElements.length,
            importedActionCount = importedActions,
            skippedActions = skipped,
        )
    }

    private fun recordFromTaskerAction(
        taskName: String,
        taskIndex: Int,
        actionIndex: Int,
        code: Int,
        action: Element,
    ): JSONObject? {
        val base = JSONObject()
            .put("id", taskerRecordId(taskName, taskIndex, actionIndex, code))
            .put("label", "$taskName: ${TASKER_ACTION_LABELS[code] ?: "Tasker action $code"}".take(MAX_LABEL_CHARS))
            .put("trigger_type", TRIGGER_MANUAL)
            .put("enabled", false)
        return when (code) {
            TASKER_RUN_SHELL -> {
                val command = argText(action, 0).trim()
                if (command.isBlank() || command.indexOf('\u0000') >= 0) return null
                base.put("action_type", ACTION_TYPE_SHELL)
                    .put("command", command)
            }
            TASKER_WRITE_FILE -> {
                val path = argText(action, 0).trim()
                val content = argText(action, 1)
                if (path.isBlank() || path.indexOf('\u0000') >= 0 || content.indexOf('\u0000') >= 0) return null
                base.put("action_type", ACTION_TYPE_FILE_WRITE)
                    .put(
                        "command",
                        JSONObject()
                            .put("path", path)
                            .put("content", content)
                            .put("append", argBoolean(action, 2))
                            .toString(),
                    )
            }
            TASKER_DELETE_FILE -> {
                val path = argText(action, 0).trim()
                if (path.isBlank() || path.indexOf('\u0000') >= 0) return null
                base.put("action_type", ACTION_TYPE_FILE_DELETE)
                    .put("command", path)
            }
            TASKER_LAUNCH_APP -> {
                val packageName = argText(action, 0).trim()
                if (!ANDROID_PACKAGE_REGEX.matches(packageName)) return null
                base.put("action_type", ACTION_TYPE_APP_LAUNCH)
                    .put("command", packageName)
            }
            TASKER_BROWSE_URL -> {
                val uri = argText(action, 0).trim()
                if (!isSafeUri(uri)) return null
                base.put("action_type", ACTION_TYPE_INTENT)
                    .put(
                        "command",
                        JSONObject()
                            .put("intent_task_action", "open_uri")
                            .put("data_uri", uri)
                            .toString(),
                    )
            }
            TASKER_NOTIFY -> {
                val title = argText(action, 0).take(MAX_NOTIFICATION_FIELD_CHARS)
                val text = argText(action, 1).take(MAX_NOTIFICATION_TEXT_CHARS)
                if (title.isBlank() && text.isBlank()) return null
                base.put("action_type", ACTION_TYPE_NOTIFICATION_ACTION)
                    .put(
                        "command",
                        JSONObject()
                            .put("notification_action", "post")
                            .put("title", title)
                            .put("text", text)
                            .put("notification_id", 1000 + actionIndex)
                            .put("only_alert_once", true)
                            .toString(),
                    )
            }
            else -> null
        }
    }

    private fun parseVariables(root: Element): JSONObject {
        val variables = JSONObject()
        val variableNodes = root.getElementsByTagName("Variable")
        for (index in 0 until variableNodes.length) {
            val variable = variableNodes.item(index) as? Element ?: continue
            val name = directText(variable, "nme")
                .ifBlank { directText(variable, "name") }
                .ifBlank { variable.getAttribute("name") }
            val value = directText(variable, "val")
                .ifBlank { directText(variable, "value") }
                .ifBlank { variable.getAttribute("value") }
            val normalized = HermesAutomationStore.normalizeVariableName(name) ?: continue
            if (value.indexOf('\u0000') < 0) {
                variables.put(normalized, value.take(MAX_VARIABLE_VALUE_CHARS))
            }
        }
        return variables
    }

    private fun argText(action: Element, index: Int): String {
        val sr = "arg$index"
        directChildren(action).forEach { child ->
            if (child.getAttribute("sr") != sr) {
                return@forEach
            }
            return when (child.tagName) {
                "Int", "Long", "Float", "Double" -> child.getAttribute("val").ifBlank { child.textContent.orEmpty() }
                "App" -> directText(child, "pkg").ifBlank { child.getAttribute("pkg") }.ifBlank { child.textContent.orEmpty() }
                else -> child.textContent.orEmpty()
            }
        }
        return ""
    }

    private fun argBoolean(action: Element, index: Int): Boolean {
        return when (argText(action, index).trim().lowercase()) {
            "1", "true", "yes", "on" -> true
            else -> false
        }
    }

    private fun directText(parent: Element, tagName: String): String {
        return directChildren(parent, tagName).firstOrNull()?.textContent.orEmpty()
    }

    private fun directChildren(parent: Element, tagName: String? = null): List<Element> {
        val children = mutableListOf<Element>()
        val nodes = parent.childNodes
        for (index in 0 until nodes.length) {
            val child = nodes.item(index)
            if (child.nodeType != Node.ELEMENT_NODE) {
                continue
            }
            val element = child as Element
            if (tagName == null || element.tagName == tagName) {
                children += element
            }
        }
        return children
    }

    private fun skippedAction(taskName: String, actionIndex: Int, code: String, reason: String): JSONObject {
        return JSONObject()
            .put("task_name", taskName)
            .put("action_index", actionIndex)
            .put("code", code)
            .put("reason", reason)
    }

    private fun taskerRecordId(taskName: String, taskIndex: Int, actionIndex: Int, code: Int): String {
        val source = "$taskName:$taskIndex:$actionIndex:$code"
        val uuid = UUID.nameUUIDFromBytes(source.toByteArray(StandardCharsets.UTF_8))
            .toString()
            .replace("-", "")
            .take(16)
        return "tasker_$uuid"
    }

    private fun firstString(arguments: JSONObject, vararg keys: String): String? {
        keys.forEach { key ->
            if (arguments.has(key) && !arguments.isNull(key)) {
                val value = arguments.optString(key)
                if (value.isNotBlank()) {
                    return value
                }
            }
        }
        return null
    }

    private fun decodeTaskerDataUri(value: String): String {
        val trimmed = value.trim()
        if (!trimmed.startsWith("data:", ignoreCase = true)) {
            return URLDecoder.decode(trimmed, StandardCharsets.UTF_8.name())
        }
        val comma = trimmed.indexOf(',')
        require(comma >= 0) { "Tasker data URI is missing a comma separator" }
        val metadata = trimmed.substring(0, comma).lowercase()
        val body = trimmed.substring(comma + 1)
        return if (";base64" in metadata) {
            decodeBase64Text(body)
        } else {
            URLDecoder.decode(body, StandardCharsets.UTF_8.name())
        }
    }

    private fun decodeBase64Text(value: String): String {
        return String(Base64.getDecoder().decode(value.trim()), StandardCharsets.UTF_8)
    }

    private fun isSafeUri(uri: String): Boolean {
        if (uri.isBlank() || uri.indexOf('\u0000') >= 0) {
            return false
        }
        val lower = uri.lowercase()
        return lower.startsWith("http://") ||
            lower.startsWith("https://") ||
            lower.startsWith("geo:") ||
            lower.startsWith("mailto:") ||
            lower.startsWith("tel:")
    }

    private const val TASKER_LAUNCH_APP = 20
    private const val TASKER_BROWSE_URL = 104
    private const val TASKER_RUN_SHELL = 123
    private const val TASKER_DELETE_FILE = 406
    private const val TASKER_WRITE_FILE = 410
    private const val TASKER_NOTIFY = 523
    private const val MAX_TASKER_XML_CHARS = 512_000
    private const val MAX_LABEL_CHARS = 80
    private const val MAX_VARIABLE_VALUE_CHARS = 4_000
    private const val MAX_NOTIFICATION_FIELD_CHARS = 120
    private const val MAX_NOTIFICATION_TEXT_CHARS = 2_000
    private val ANDROID_PACKAGE_REGEX = Regex("[A-Za-z][A-Za-z0-9_]*(\\.[A-Za-z][A-Za-z0-9_]*)+")
    private val DOCTYPE_REGEX = Regex("<!\\s*DOCTYPE", RegexOption.IGNORE_CASE)
    private val ENTITY_REGEX = Regex("<!\\s*ENTITY", RegexOption.IGNORE_CASE)
    private val TASKER_ACTION_LABELS = mapOf(
        TASKER_LAUNCH_APP to "Launch App",
        TASKER_BROWSE_URL to "Browse URL",
        TASKER_RUN_SHELL to "Run Shell",
        TASKER_DELETE_FILE to "Delete File",
        TASKER_WRITE_FILE to "Write File",
        TASKER_NOTIFY to "Notify",
    )
}
