package com.nousresearch.hermesagent.device

import android.content.Context
import android.content.Intent
import android.os.Bundle
import org.json.JSONObject
import java.util.UUID

object HermesTaskerConditionBridge {
    const val ACTION_EDIT_CONDITION = "com.twofortyfouram.locale.intent.action.EDIT_CONDITION"
    const val ACTION_QUERY_CONDITION = "com.twofortyfouram.locale.intent.action.QUERY_CONDITION"
    const val EXTRA_BUNDLE = HermesTaskerPluginBridge.EXTRA_BUNDLE
    const val EXTRA_STRING_BLURB = HermesTaskerPluginBridge.EXTRA_STRING_BLURB
    const val EXTRA_VARIABLES = "net.dinglisch.android.tasker.extras.VARIABLES"
    const val KEY_CONDITION_TYPE = "com.nousresearch.hermesagent.tasker.CONDITION_TYPE"
    const val KEY_AUTOMATION_ID = "com.nousresearch.hermesagent.tasker.CONDITION_AUTOMATION_ID"
    const val KEY_VARIABLE_NAME = "com.nousresearch.hermesagent.tasker.CONDITION_VARIABLE_NAME"
    const val KEY_EXPECTED_VALUE = "com.nousresearch.hermesagent.tasker.CONDITION_EXPECTED_VALUE"
    const val KEY_TOKEN = "com.nousresearch.hermesagent.tasker.CONDITION_TOKEN"
    const val KEY_LABEL = "com.nousresearch.hermesagent.tasker.CONDITION_LABEL"

    const val CONDITION_SHIZUKU_AVAILABLE = "shizuku_available"
    const val CONDITION_SHIZUKU_UNAVAILABLE = "shizuku_unavailable"
    const val CONDITION_AUTOMATION_ENABLED = "automation_enabled"
    const val CONDITION_AUTOMATION_DISABLED = "automation_disabled"
    const val CONDITION_AUTOMATION_LAST_SUCCESS = "automation_last_success"
    const val CONDITION_AUTOMATION_LAST_FAILED = "automation_last_failed"
    const val CONDITION_VARIABLE_SET = "variable_set"
    const val CONDITION_VARIABLE_EQUALS = "variable_equals"

    const val RESULT_CONDITION_SATISFIED = 16
    const val RESULT_CONDITION_UNSATISFIED = 17
    const val RESULT_CONDITION_UNKNOWN = 18

    data class ConditionChoice(val id: String, val label: String) {
        override fun toString(): String = label
    }

    data class QueryResult(
        val resultCode: Int,
        val variables: Bundle,
        val error: String = "",
    )

    fun conditionChoices(): List<ConditionChoice> = listOf(
        ConditionChoice(CONDITION_SHIZUKU_AVAILABLE, "Shizuku available"),
        ConditionChoice(CONDITION_SHIZUKU_UNAVAILABLE, "Shizuku unavailable"),
        ConditionChoice(CONDITION_AUTOMATION_ENABLED, "Automation enabled"),
        ConditionChoice(CONDITION_AUTOMATION_DISABLED, "Automation disabled"),
        ConditionChoice(CONDITION_AUTOMATION_LAST_SUCCESS, "Automation last run succeeded"),
        ConditionChoice(CONDITION_AUTOMATION_LAST_FAILED, "Automation last run failed"),
        ConditionChoice(CONDITION_VARIABLE_SET, "Hermes variable is set"),
        ConditionChoice(CONDITION_VARIABLE_EQUALS, "Hermes variable equals"),
    )

    fun bundleFromIntent(intent: Intent): Bundle? = intent.getBundleExtra(EXTRA_BUNDLE)

    fun buildResultIntent(
        context: Context,
        conditionType: String,
        automationId: String = "",
        variableName: String = "",
        expectedValue: String = "",
        label: String = "",
        existingToken: String = "",
    ): Intent {
        val normalizedType = normalizeConditionType(conditionType)
            ?: throw IllegalArgumentException("Choose a supported Hermes condition")
        val normalizedAutomationId = normalizeOptionalAutomationId(automationId)
        val normalizedVariableName = HermesAutomationStore.normalizeVariableName(variableName).orEmpty()
        val trimmedExpectedValue = expectedValue.take(MAX_EXPECTED_VALUE_CHARS)
        validateConditionInput(normalizedType, normalizedAutomationId, normalizedVariableName)
        val token = ensureAuthorizedToken(
            context = context,
            signature = conditionSignature(
                normalizedType,
                normalizedAutomationId,
                normalizedVariableName,
                trimmedExpectedValue,
            ),
            existingToken = existingToken,
        )
        val blurb = blurbFor(
            conditionType = normalizedType,
            automationId = normalizedAutomationId,
            variableName = normalizedVariableName,
            expectedValue = trimmedExpectedValue,
            label = label,
        )
        val bundle = Bundle().apply {
            putString(KEY_CONDITION_TYPE, normalizedType)
            putString(KEY_AUTOMATION_ID, normalizedAutomationId)
            putString(KEY_VARIABLE_NAME, normalizedVariableName)
            putString(KEY_EXPECTED_VALUE, trimmedExpectedValue)
            putString(KEY_TOKEN, token)
            putString(KEY_LABEL, label.take(MAX_LABEL_CHARS))
        }
        return Intent()
            .putExtra(EXTRA_BUNDLE, bundle)
            .putExtra(EXTRA_STRING_BLURB, blurb)
    }

    fun queryCondition(context: Context, bundle: Bundle?): QueryResult {
        if (bundle == null) {
            return unknown("Hermes condition query is missing Locale EXTRA_BUNDLE")
        }
        val conditionType = normalizeConditionType(bundle.getString(KEY_CONDITION_TYPE).orEmpty())
            ?: return unknown("Hermes condition query has an unsupported condition type")
        val automationId = runCatching {
            normalizeOptionalAutomationId(bundle.getString(KEY_AUTOMATION_ID).orEmpty())
        }.getOrElse {
            return unknown("Hermes condition query has an invalid automation ID")
        }
        val variableName = HermesAutomationStore.normalizeVariableName(bundle.getString(KEY_VARIABLE_NAME).orEmpty()).orEmpty()
        val expectedValue = bundle.getString(KEY_EXPECTED_VALUE).orEmpty().take(MAX_EXPECTED_VALUE_CHARS)
        val signature = conditionSignature(conditionType, automationId, variableName, expectedValue)
        val token = bundle.getString(KEY_TOKEN).orEmpty().trim()
        if (!isAuthorizedToken(context, signature, token)) {
            return unknown("Hermes condition token is missing or invalid")
        }
        return runCatching {
            evaluateCondition(context.applicationContext, conditionType, automationId, variableName, expectedValue)
        }.getOrElse { error ->
            unknown("Hermes condition query failed: ${error.message.orEmpty()}")
        }
    }

    fun blurbFor(
        conditionType: String,
        automationId: String,
        variableName: String,
        expectedValue: String,
        label: String,
    ): String {
        val custom = label.trim()
        if (custom.isNotBlank()) {
            return custom.take(MAX_LABEL_CHARS)
        }
        return when (conditionType) {
            CONDITION_SHIZUKU_AVAILABLE -> "Hermes: Shizuku available"
            CONDITION_SHIZUKU_UNAVAILABLE -> "Hermes: Shizuku unavailable"
            CONDITION_AUTOMATION_ENABLED -> "Hermes automation enabled: $automationId"
            CONDITION_AUTOMATION_DISABLED -> "Hermes automation disabled: $automationId"
            CONDITION_AUTOMATION_LAST_SUCCESS -> "Hermes automation succeeded: $automationId"
            CONDITION_AUTOMATION_LAST_FAILED -> "Hermes automation failed: $automationId"
            CONDITION_VARIABLE_SET -> "Hermes variable set: %$variableName"
            CONDITION_VARIABLE_EQUALS -> "Hermes variable %$variableName = $expectedValue"
            else -> "Hermes condition"
        }.take(MAX_LABEL_CHARS)
    }

    private fun evaluateCondition(
        context: Context,
        conditionType: String,
        automationId: String,
        variableName: String,
        expectedValue: String,
    ): QueryResult {
        val store = HermesAutomationStore(context)
        val variables = Bundle().apply {
            putString("%hermes_condition", conditionType)
            putString("%hermes_automation_id", automationId)
            putString("%hermes_variable_name", if (variableName.isBlank()) "" else "%$variableName")
        }
        val satisfied = when (conditionType) {
            CONDITION_SHIZUKU_AVAILABLE,
            CONDITION_SHIZUKU_UNAVAILABLE -> {
                val status = HermesPrivilegedAccessBridge.readStatus(context)
                val available = status.shizukuBinderAlive && status.shizukuPermissionGranted
                variables.putString("%hermes_shizuku_available", available.toString())
                variables.putString("%hermes_shizuku_running", status.shizukuBinderAlive.toString())
                variables.putString("%hermes_shizuku_permission", status.shizukuPermissionGranted.toString())
                variables.putString("%hermes_shizuku_uid", status.shizukuUid?.toString().orEmpty())
                if (conditionType == CONDITION_SHIZUKU_AVAILABLE) available else !available
            }
            CONDITION_AUTOMATION_ENABLED -> store.get(automationId)?.enabled == true
            CONDITION_AUTOMATION_DISABLED -> store.get(automationId)?.let { !it.enabled } == true
            CONDITION_AUTOMATION_LAST_SUCCESS -> store.get(automationId)?.lastSuccess == true
            CONDITION_AUTOMATION_LAST_FAILED -> store.get(automationId)?.lastSuccess == false
            CONDITION_VARIABLE_SET -> store.getVariable(variableName) != null
            CONDITION_VARIABLE_EQUALS -> {
                val value = store.getVariable(variableName)
                variables.putString("%hermes_variable_value", value.orEmpty().take(MAX_VARIABLE_RETURN_CHARS))
                value == expectedValue
            }
            else -> return unknown("Unsupported Hermes condition type")
        }
        variables.putString("%hermes_satisfied", satisfied.toString())
        return QueryResult(
            resultCode = if (satisfied) RESULT_CONDITION_SATISFIED else RESULT_CONDITION_UNSATISFIED,
            variables = variables,
        )
    }

    private fun validateConditionInput(conditionType: String, automationId: String, variableName: String) {
        if (conditionType in AUTOMATION_CONDITIONS && automationId.isBlank()) {
            throw IllegalArgumentException("Automation conditions require a saved Hermes automation ID")
        }
        if (conditionType in VARIABLE_CONDITIONS && variableName.isBlank()) {
            throw IllegalArgumentException("Variable conditions require a Hermes variable name")
        }
    }

    private fun normalizeConditionType(raw: String): String? {
        val normalized = raw.trim().lowercase().replace('-', '_').replace(' ', '_')
        return CONDITION_SYNONYMS[normalized] ?: normalized.takeIf { it in SUPPORTED_CONDITIONS }
    }

    private fun normalizeOptionalAutomationId(raw: String): String {
        val id = raw.trim()
        if (id.isBlank()) {
            return ""
        }
        require(id.length <= MAX_AUTOMATION_ID_CHARS && id.indexOf('\u0000') < 0) {
            "Hermes automation ID is too long or invalid"
        }
        return id
    }

    private fun ensureAuthorizedToken(context: Context, signature: String, existingToken: String = ""): String {
        val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val reusable = existingToken.trim().takeIf { token ->
            token.length in 16..128 && prefs.getString(tokenKey(token), "") == signature
        }
        val token = reusable ?: UUID.randomUUID().toString()
        prefs.edit()
            .putString(tokenKey(token), signature)
            .apply()
        return token
    }

    private fun isAuthorizedToken(context: Context, signature: String, token: String): Boolean {
        if (token.length !in 16..128) {
            return false
        }
        val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(tokenKey(token), "") == signature
    }

    private fun conditionSignature(
        conditionType: String,
        automationId: String,
        variableName: String,
        expectedValue: String,
    ): String {
        return JSONObject()
            .put("condition_type", conditionType)
            .put("automation_id", automationId)
            .put("variable_name", variableName)
            .put("expected_value", expectedValue)
            .toString()
    }

    private fun unknown(message: String): QueryResult {
        return QueryResult(
            resultCode = RESULT_CONDITION_UNKNOWN,
            variables = Bundle().apply {
                putString("%hermes_satisfied", "unknown")
                putString("%hermes_error", message.take(MAX_VARIABLE_RETURN_CHARS))
            },
            error = message,
        )
    }

    private fun tokenKey(token: String): String = "tasker_condition_token:$token"

    private const val PREFS_NAME = "hermes_tasker_condition_tokens"
    private const val MAX_AUTOMATION_ID_CHARS = 128
    private const val MAX_LABEL_CHARS = 80
    private const val MAX_EXPECTED_VALUE_CHARS = 512
    private const val MAX_VARIABLE_RETURN_CHARS = 512

    private val AUTOMATION_CONDITIONS = setOf(
        CONDITION_AUTOMATION_ENABLED,
        CONDITION_AUTOMATION_DISABLED,
        CONDITION_AUTOMATION_LAST_SUCCESS,
        CONDITION_AUTOMATION_LAST_FAILED,
    )
    private val VARIABLE_CONDITIONS = setOf(
        CONDITION_VARIABLE_SET,
        CONDITION_VARIABLE_EQUALS,
    )
    private val SUPPORTED_CONDITIONS = setOf(
        CONDITION_SHIZUKU_AVAILABLE,
        CONDITION_SHIZUKU_UNAVAILABLE,
        CONDITION_AUTOMATION_ENABLED,
        CONDITION_AUTOMATION_DISABLED,
        CONDITION_AUTOMATION_LAST_SUCCESS,
        CONDITION_AUTOMATION_LAST_FAILED,
        CONDITION_VARIABLE_SET,
        CONDITION_VARIABLE_EQUALS,
    )
    private val CONDITION_SYNONYMS = mapOf(
        "shizuku" to CONDITION_SHIZUKU_AVAILABLE,
        "shizuku_ready" to CONDITION_SHIZUKU_AVAILABLE,
        "shizuku_running" to CONDITION_SHIZUKU_AVAILABLE,
        "shizuku_permission_granted" to CONDITION_SHIZUKU_AVAILABLE,
        "shizuku_missing" to CONDITION_SHIZUKU_UNAVAILABLE,
        "shizuku_not_running" to CONDITION_SHIZUKU_UNAVAILABLE,
        "shizuku_permission_missing" to CONDITION_SHIZUKU_UNAVAILABLE,
        "enabled" to CONDITION_AUTOMATION_ENABLED,
        "disabled" to CONDITION_AUTOMATION_DISABLED,
        "last_success" to CONDITION_AUTOMATION_LAST_SUCCESS,
        "last_failed" to CONDITION_AUTOMATION_LAST_FAILED,
        "variable" to CONDITION_VARIABLE_SET,
        "variable_exists" to CONDITION_VARIABLE_SET,
        "variable_equal" to CONDITION_VARIABLE_EQUALS,
        "variable_value" to CONDITION_VARIABLE_EQUALS,
    )
}
