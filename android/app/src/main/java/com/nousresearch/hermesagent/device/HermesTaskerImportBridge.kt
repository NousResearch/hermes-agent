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
            TASKER_WAIT -> {
                val durationMs = waitDurationMsFromTaskerAction(action) ?: return null
                base.put("action_type", ACTION_TYPE_WAIT)
                    .put(
                        "command",
                        JSONObject()
                            .put("duration_ms", durationMs)
                            .toString(),
                    )
            }
            TASKER_VIBRATE -> {
                val durationMs = vibrationDurationMsFromTaskerAction(action) ?: return null
                base.put("action_type", ACTION_TYPE_VIBRATION_ACTION)
                    .put(
                        "command",
                        JSONObject()
                            .put("vibration_action", "vibrate")
                            .put("duration_ms", durationMs)
                            .toString(),
                    )
            }
            TASKER_VIBRATE_PATTERN -> {
                val pattern = vibrationPatternMsFromTaskerAction(action) ?: return null
                val command = JSONObject()
                    .put("vibration_action", "vibrate")
                    .put("duration_ms", pattern.sum())
                if (pattern.size == 1) {
                    command.put("duration_ms", pattern.first())
                } else {
                    command.put("pattern_ms", JSONArray(pattern))
                }
                base.put("action_type", ACTION_TYPE_VIBRATION_ACTION)
                    .put("command", command.toString())
            }
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
            TASKER_VARIABLE_SET -> {
                val name = argText(action, 0).trim()
                val normalized = HermesAutomationStore.normalizeVariableName(name) ?: return null
                val value = argText(action, 1)
                if (value.indexOf('\u0000') >= 0) return null
                base.put("action_type", ACTION_TYPE_VARIABLE_ACTION)
                    .put(
                        "command",
                        JSONObject()
                            .put("variable_action", "set")
                            .put("name", normalized)
                            .put("value", value.take(MAX_VARIABLE_VALUE_CHARS))
                            .toString(),
                    )
            }
            TASKER_VARIABLE_CLEAR -> {
                val name = argText(action, 0).trim()
                val normalized = HermesAutomationStore.normalizeVariableName(name) ?: return null
                base.put("action_type", ACTION_TYPE_VARIABLE_ACTION)
                    .put(
                        "command",
                        JSONObject()
                            .put("variable_action", "clear")
                            .put("name", normalized)
                            .toString(),
                    )
            }
            TASKER_SET_CLIPBOARD -> {
                val text = argText(action, 0)
                if (text.indexOf('\u0000') >= 0) return null
                base.put("action_type", ACTION_TYPE_CLIPBOARD_ACTION)
                    .put(
                        "command",
                        JSONObject()
                            .put("clipboard_action", "set")
                            .put("text", text.take(MAX_CLIPBOARD_TEXT_CHARS))
                            .put("label", "Tasker import")
                            .toString(),
                    )
            }
            TASKER_FLASH -> {
                val text = argText(action, 0)
                if (text.isBlank() || text.indexOf('\u0000') >= 0) return null
                base.put("action_type", ACTION_TYPE_TOAST_ACTION)
                    .put(
                        "command",
                        JSONObject()
                            .put("text", text.take(MAX_TOAST_TEXT_CHARS))
                            .put("long", argBoolean(action, 1))
                            .toString(),
                    )
            }
            in TASKER_GLOBAL_UI_ACTIONS -> {
                base.put("action_type", ACTION_TYPE_UI_ACTION)
                    .put(
                        "command",
                        JSONObject()
                            .put("ui_action", TASKER_GLOBAL_UI_ACTIONS.getValue(code))
                            .toString(),
                    )
            }
            in TASKER_SETTINGS_ACTIONS -> {
                base.put("action_type", ACTION_TYPE_SYSTEM_ACTION)
                    .put("command", TASKER_SETTINGS_ACTIONS.getValue(code))
            }
            TASKER_CUSTOM_SETTING -> {
                customSettingRecordFromTaskerAction(base, action) ?: return null
            }
            in TASKER_SHIZUKU_FIXED_ACTIONS -> {
                shizukuRecordFromTaskerAction(base, code, action) ?: return null
            }
            else -> null
        }
    }

    private fun shizukuRecordFromTaskerAction(base: JSONObject, code: Int, action: Element): JSONObject? {
        val payload = JSONObject()
            .put("shizuku_action", TASKER_SHIZUKU_FIXED_ACTIONS.getValue(code))
        if (code in TASKER_SHIZUKU_TOGGLE_ACTIONS) {
            val enabled = taskerToggleEnabledFromAction(action) ?: return null
            payload.put("target_enabled", enabled)
        }
        if (code == TASKER_DO_NOT_DISTURB) {
            val mode = taskerDndModeFromAction(action) ?: return null
            payload.put("dnd_mode", mode)
        }
        return base.put("action_type", ACTION_TYPE_SHIZUKU_ACTION)
            .put("use_shizuku", true)
            .put("command", payload.toString())
    }

    private fun customSettingRecordFromTaskerAction(base: JSONObject, action: Element): JSONObject? {
        val namespace = taskerCustomSettingNamespace(action) ?: return null
        val name = argText(action, 1).trim()
        val value = argText(action, 2)
        if (!isSafeSettingName(name) ||
            value.isBlank() ||
            value.indexOf('\u0000') >= 0 ||
            value.trim() == TASKER_CUSTOM_SETTING_TOGGLE
        ) {
            return null
        }
        val payload = JSONObject()
            .put("shizuku_action", "set_custom_setting")
            .put("setting_namespace", namespace)
            .put("setting_name", name)
            .put("setting_value", value.take(MAX_CUSTOM_SETTING_VALUE_CHARS))
        return base.put("action_type", ACTION_TYPE_SHIZUKU_ACTION)
            .put("use_shizuku", true)
            .put("command", payload.toString())
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

    private fun taskerToggleEnabledFromAction(action: Element): Boolean? {
        return when (argText(action, 0).trim().lowercase()) {
            "1", "true", "yes", "on", "enabled", "enable" -> true
            "0", "false", "no", "off", "disabled", "disable" -> false
            else -> null
        }
    }

    private fun taskerDndModeFromAction(action: Element): String? {
        return when (argText(action, 0).trim().lowercase()) {
            "0", "none", "total_silence", "no_interruptions" -> "on"
            "1", "priority", "important_interruptions" -> "priority"
            "2", "all", "off", "disable", "disabled" -> "off"
            "3", "alarms", "alarms_only" -> "alarms"
            else -> null
        }
    }

    private fun taskerCustomSettingNamespace(action: Element): String? {
        return when (argText(action, 0).trim().lowercase()) {
            "0", "system" -> "system"
            "1", "secure" -> "secure"
            "2", "global" -> "global"
            else -> null
        }
    }

    private fun waitDurationMsFromTaskerAction(action: Element): Long? {
        val total = listOf(
            taskerWaitComponentMs(action, 0, 1L) ?: return null,
            taskerWaitComponentMs(action, 1, 1_000L) ?: return null,
            taskerWaitComponentMs(action, 2, 60_000L) ?: return null,
            taskerWaitComponentMs(action, 3, 3_600_000L) ?: return null,
            taskerWaitComponentMs(action, 4, 86_400_000L) ?: return null,
        ).sum()
        return total.takeIf { it in 1L..MAX_WAIT_DURATION_MS }
    }

    private fun taskerWaitComponentMs(action: Element, index: Int, factor: Long): Long? {
        val raw = argText(action, index).trim()
        val value = if (raw.isBlank()) 0L else raw.toLongOrNull() ?: return null
        if (value < 0L || value > MAX_WAIT_DURATION_MS / factor) {
            return null
        }
        return value * factor
    }

    private fun vibrationDurationMsFromTaskerAction(action: Element): Long? {
        val raw = argText(action, 0).trim().ifBlank { "200" }
        val durationMs = raw.toLongOrNull() ?: return null
        return durationMs.takeIf { it in 1L..MAX_VIBRATION_TOTAL_MS }
    }

    private fun vibrationPatternMsFromTaskerAction(action: Element): List<Long>? {
        val raw = argText(action, 0).trim()
        if (raw.isBlank() || raw.indexOf('\u0000') >= 0) {
            return null
        }
        val pattern = raw
            .split(Regex("[,\\s]+"))
            .filter { it.isNotBlank() }
            .map { it.toLongOrNull() ?: return null }
        if (pattern.isEmpty() || pattern.size > MAX_VIBRATION_PATTERN_ENTRIES) {
            return null
        }
        if (pattern.any { it < 0L || it > MAX_VIBRATION_TOTAL_MS } || pattern.none { it > 0L }) {
            return null
        }
        return pattern.takeIf { it.sum() <= MAX_VIBRATION_TOTAL_MS }
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

    private fun isSafeSettingName(name: String): Boolean {
        return name.length in 1..MAX_CUSTOM_SETTING_NAME_CHARS &&
            name.indexOf('\u0000') < 0 &&
            SETTINGS_NAME_REGEX.matches(name)
    }

    private const val TASKER_LAUNCH_APP = 20
    private const val TASKER_GO_HOME = 25
    private const val TASKER_WAIT = 30
    private const val TASKER_VIBRATE = 61
    private const val TASKER_VIBRATE_PATTERN = 62
    private const val TASKER_BROWSE_URL = 104
    private const val TASKER_SET_CLIPBOARD = 105
    private const val TASKER_WIFI_TETHER = 113
    private const val TASKER_RUN_SHELL = 123
    private const val TASKER_POWER_MODE = 175
    private const val TASKER_DEVELOPER_SETTINGS = 197
    private const val TASKER_DEVICE_INFO_SETTINGS = 198
    private const val TASKER_ADD_ACCOUNT_SETTINGS = 199
    private const val TASKER_ALL_SETTINGS = 200
    private const val TASKER_AIRPLANE_MODE_SETTINGS = 201
    private const val TASKER_APN_SETTINGS = 202
    private const val TASKER_DATE_SETTINGS = 203
    private const val TASKER_INTERNAL_STORAGE_SETTINGS = 204
    private const val TASKER_WIFI_SETTINGS = 206
    private const val TASKER_LOCATION_SETTINGS = 208
    private const val TASKER_INPUT_METHOD_SETTINGS = 210
    private const val TASKER_SYNC_SETTINGS = 211
    private const val TASKER_WIFI_IP_SETTINGS = 212
    private const val TASKER_WIRELESS_SETTINGS = 214
    private const val TASKER_APP_SETTINGS = 216
    private const val TASKER_BLUETOOTH_SETTINGS = 218
    private const val TASKER_QUICK_SETTINGS = 219
    private const val TASKER_MOBILE_DATA_SETTINGS = 220
    private const val TASKER_DISPLAY_SETTINGS = 222
    private const val TASKER_LOCALE_SETTINGS = 224
    private const val TASKER_APP_MANAGE_SETTINGS = 226
    private const val TASKER_MEMORY_CARD_SETTINGS = 227
    private const val TASKER_NETWORK_OPERATOR_SETTINGS = 228
    private const val TASKER_QUICK_LAUNCH_SETTINGS = 229
    private const val TASKER_SECURITY_SETTINGS = 230
    private const val TASKER_SEARCH_SETTINGS = 231
    private const val TASKER_SOUND_SETTINGS = 232
    private const val TASKER_DICTIONARY_SETTINGS = 234
    private const val TASKER_CUSTOM_SETTING = 235
    private const val TASKER_ACCESSIBILITY_SETTINGS = 236
    private const val TASKER_NOTIFICATION_LISTENER_SETTINGS = 237
    private const val TASKER_PRIVACY_SETTINGS = 238
    private const val TASKER_PRINT_SETTINGS = 239
    private const val TASKER_BACK_BUTTON = 245
    private const val TASKER_SHOW_RECENTS = 247
    private const val TASKER_TURN_OFF = 248
    private const val TASKER_POWER_USAGE_SETTINGS = 257
    private const val TASKER_BLUETOOTH = 294
    private const val TASKER_DO_NOT_DISTURB = 312
    private const val TASKER_AIRPLANE_MODE = 333
    private const val TASKER_NOTIFICATION_SETTINGS = 337
    private const val TASKER_DELETE_FILE = 406
    private const val TASKER_WRITE_FILE = 410
    private const val TASKER_WIFI = 425
    private const val TASKER_MOBILE_DATA = 433
    private const val TASKER_NOTIFY = 523
    private const val TASKER_VARIABLE_SET = 547
    private const val TASKER_FLASH = 548
    private const val TASKER_VARIABLE_CLEAR = 549
    private const val TASKER_END_CALL = 733
    private const val TASKER_NFC_SETTINGS = 956
    private const val MAX_TASKER_XML_CHARS = 512_000
    private const val MAX_LABEL_CHARS = 80
    private const val MAX_VARIABLE_VALUE_CHARS = 4_000
    private const val MAX_CUSTOM_SETTING_NAME_CHARS = 128
    private const val MAX_CUSTOM_SETTING_VALUE_CHARS = 512
    private const val MAX_WAIT_DURATION_MS = 60_000L
    private const val MAX_VIBRATION_TOTAL_MS = 60_000L
    private const val MAX_VIBRATION_PATTERN_ENTRIES = 32
    private const val MAX_NOTIFICATION_FIELD_CHARS = 120
    private const val MAX_NOTIFICATION_TEXT_CHARS = 2_000
    private const val MAX_CLIPBOARD_TEXT_CHARS = 8_000
    private const val MAX_TOAST_TEXT_CHARS = 1_000
    private val ANDROID_PACKAGE_REGEX = Regex("[A-Za-z][A-Za-z0-9_]*(\\.[A-Za-z][A-Za-z0-9_]*)+")
    private val SETTINGS_NAME_REGEX = Regex("[A-Za-z0-9_.:-]+")
    private const val TASKER_CUSTOM_SETTING_TOGGLE = "=:=toggle=:="
    private val DOCTYPE_REGEX = Regex("<!\\s*DOCTYPE", RegexOption.IGNORE_CASE)
    private val ENTITY_REGEX = Regex("<!\\s*ENTITY", RegexOption.IGNORE_CASE)
    private val TASKER_GLOBAL_UI_ACTIONS = mapOf(
        TASKER_GO_HOME to "home",
        TASKER_BACK_BUTTON to "back",
        TASKER_SHOW_RECENTS to "recents",
        TASKER_QUICK_SETTINGS to "quick_settings",
    )
    private val TASKER_SHIZUKU_FIXED_ACTIONS = mapOf(
        TASKER_TURN_OFF to "turn_screen_off",
        TASKER_BLUETOOTH to "set_bluetooth_enabled",
        TASKER_DO_NOT_DISTURB to "set_dnd_mode",
        TASKER_AIRPLANE_MODE to "set_airplane_mode_enabled",
        TASKER_POWER_MODE to "set_power_save_mode",
        TASKER_WIFI_TETHER to "set_wifi_tethering_enabled",
        TASKER_WIFI to "set_wifi_enabled",
        TASKER_MOBILE_DATA to "set_mobile_data_enabled",
        TASKER_END_CALL to "end_call",
    )
    private val TASKER_SHIZUKU_TOGGLE_ACTIONS = setOf(
        TASKER_AIRPLANE_MODE,
        TASKER_BLUETOOTH,
        TASKER_POWER_MODE,
        TASKER_WIFI_TETHER,
        TASKER_WIFI,
        TASKER_MOBILE_DATA,
    )
    private val TASKER_SETTINGS_ACTIONS = mapOf(
        TASKER_DEVELOPER_SETTINGS to "open_developer_options",
        TASKER_DEVICE_INFO_SETTINGS to "open_device_info_settings",
        TASKER_ADD_ACCOUNT_SETTINGS to "open_add_account_settings",
        TASKER_ALL_SETTINGS to "open_all_settings",
        TASKER_AIRPLANE_MODE_SETTINGS to "open_airplane_mode_settings",
        TASKER_APN_SETTINGS to "open_apn_settings",
        TASKER_DATE_SETTINGS to "open_date_settings",
        TASKER_INTERNAL_STORAGE_SETTINGS to "open_internal_storage_settings",
        TASKER_WIFI_SETTINGS to "open_wifi_panel",
        TASKER_LOCATION_SETTINGS to "open_location_settings",
        TASKER_INPUT_METHOD_SETTINGS to "open_input_method_settings",
        TASKER_SYNC_SETTINGS to "open_sync_settings",
        TASKER_WIFI_IP_SETTINGS to "open_wifi_ip_settings",
        TASKER_WIRELESS_SETTINGS to "open_wireless_settings",
        TASKER_APP_SETTINGS to "open_app_settings",
        TASKER_BLUETOOTH_SETTINGS to "open_bluetooth_settings",
        TASKER_MOBILE_DATA_SETTINGS to "open_mobile_network_settings",
        TASKER_DISPLAY_SETTINGS to "open_display_settings",
        TASKER_LOCALE_SETTINGS to "open_locale_settings",
        TASKER_APP_MANAGE_SETTINGS to "open_manage_apps_settings",
        TASKER_MEMORY_CARD_SETTINGS to "open_memory_card_settings",
        TASKER_NETWORK_OPERATOR_SETTINGS to "open_mobile_network_settings",
        TASKER_QUICK_LAUNCH_SETTINGS to "open_quick_launch_settings",
        TASKER_SECURITY_SETTINGS to "open_security_settings",
        TASKER_SEARCH_SETTINGS to "open_search_settings",
        TASKER_SOUND_SETTINGS to "open_sound_settings",
        TASKER_DICTIONARY_SETTINGS to "open_dictionary_settings",
        TASKER_ACCESSIBILITY_SETTINGS to "open_accessibility_settings",
        TASKER_NOTIFICATION_LISTENER_SETTINGS to "open_notification_listener_settings",
        TASKER_PRIVACY_SETTINGS to "open_privacy_settings",
        TASKER_PRINT_SETTINGS to "open_print_settings",
        TASKER_POWER_USAGE_SETTINGS to "open_power_usage_settings",
        TASKER_NOTIFICATION_SETTINGS to "open_system_notification_settings",
        TASKER_NFC_SETTINGS to "open_nfc_settings",
    )
    private val TASKER_ACTION_LABELS = mapOf(
        TASKER_LAUNCH_APP to "Launch App",
        TASKER_GO_HOME to "Go Home",
        TASKER_WAIT to "Wait",
        TASKER_VIBRATE to "Vibrate",
        TASKER_VIBRATE_PATTERN to "Vibrate Pattern",
        TASKER_BROWSE_URL to "Browse URL",
        TASKER_SET_CLIPBOARD to "Set Clipboard",
        TASKER_WIFI_TETHER to "WiFi Tether",
        TASKER_RUN_SHELL to "Run Shell",
        TASKER_POWER_MODE to "Power Mode",
        TASKER_DEVELOPER_SETTINGS to "Developer Settings",
        TASKER_DEVICE_INFO_SETTINGS to "Device Info Settings",
        TASKER_ADD_ACCOUNT_SETTINGS to "Add Account Settings",
        TASKER_ALL_SETTINGS to "All Settings",
        TASKER_AIRPLANE_MODE_SETTINGS to "Airplane Mode Settings",
        TASKER_APN_SETTINGS to "APN Settings",
        TASKER_DATE_SETTINGS to "Date Settings",
        TASKER_INTERNAL_STORAGE_SETTINGS to "Internal Storage Settings",
        TASKER_WIFI_SETTINGS to "Wi-Fi Settings",
        TASKER_LOCATION_SETTINGS to "Location Settings",
        TASKER_INPUT_METHOD_SETTINGS to "Input Method Settings",
        TASKER_SYNC_SETTINGS to "Sync Settings",
        TASKER_WIFI_IP_SETTINGS to "Wi-Fi IP Settings",
        TASKER_WIRELESS_SETTINGS to "Wireless Settings",
        TASKER_APP_SETTINGS to "App Settings",
        TASKER_BLUETOOTH_SETTINGS to "Bluetooth Settings",
        TASKER_QUICK_SETTINGS to "Quick Settings",
        TASKER_MOBILE_DATA_SETTINGS to "Mobile Data Settings",
        TASKER_DISPLAY_SETTINGS to "Display Settings",
        TASKER_LOCALE_SETTINGS to "Locale Settings",
        TASKER_APP_MANAGE_SETTINGS to "App Manage Settings",
        TASKER_MEMORY_CARD_SETTINGS to "Memory Card Settings",
        TASKER_NETWORK_OPERATOR_SETTINGS to "Network Operator Settings",
        TASKER_QUICK_LAUNCH_SETTINGS to "Quick Launch Settings",
        TASKER_SECURITY_SETTINGS to "Security Settings",
        TASKER_SEARCH_SETTINGS to "Search Settings",
        TASKER_SOUND_SETTINGS to "Sound Settings",
        TASKER_DICTIONARY_SETTINGS to "Dictionary Settings",
        TASKER_CUSTOM_SETTING to "Custom Setting",
        TASKER_ACCESSIBILITY_SETTINGS to "Accessibility Settings",
        TASKER_NOTIFICATION_LISTENER_SETTINGS to "Notification Listener Settings",
        TASKER_PRIVACY_SETTINGS to "Privacy Settings",
        TASKER_PRINT_SETTINGS to "Print Settings",
        TASKER_BACK_BUTTON to "Back Button",
        TASKER_SHOW_RECENTS to "Show Recents",
        TASKER_TURN_OFF to "Turn Off",
        TASKER_POWER_USAGE_SETTINGS to "Power Usage Settings",
        TASKER_BLUETOOTH to "Bluetooth",
        TASKER_DO_NOT_DISTURB to "Do Not Disturb",
        TASKER_AIRPLANE_MODE to "Airplane Mode",
        TASKER_NOTIFICATION_SETTINGS to "Notification Settings",
        TASKER_DELETE_FILE to "Delete File",
        TASKER_WRITE_FILE to "Write File",
        TASKER_WIFI to "Wi-Fi",
        TASKER_MOBILE_DATA to "Mobile Data",
        TASKER_NOTIFY to "Notify",
        TASKER_VARIABLE_SET to "Variable Set",
        TASKER_FLASH to "Flash",
        TASKER_VARIABLE_CLEAR to "Variable Clear",
        TASKER_END_CALL to "End Call",
        TASKER_NFC_SETTINGS to "NFC Settings",
    )
}
