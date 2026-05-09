package com.nousresearch.hermesagent.device

import android.app.Activity
import android.os.Bundle
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import com.nousresearch.hermesagent.R

class HermesTaskerConditionEditActivity : Activity() {
    private data class AutomationChoice(val id: String, val label: String) {
        override fun toString(): String = if (label.isBlank() || label == id) id else "$label ($id)"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val existing = HermesTaskerConditionBridge.bundleFromIntent(intent)
        val existingType = existing?.getString(HermesTaskerConditionBridge.KEY_CONDITION_TYPE).orEmpty()
        val existingAutomationId = existing?.getString(HermesTaskerConditionBridge.KEY_AUTOMATION_ID).orEmpty()
        val existingVariableName = existing?.getString(HermesTaskerConditionBridge.KEY_VARIABLE_NAME).orEmpty()
        val existingExpectedValue = existing?.getString(HermesTaskerConditionBridge.KEY_EXPECTED_VALUE).orEmpty()
        val existingToken = existing?.getString(HermesTaskerConditionBridge.KEY_TOKEN).orEmpty()
        val conditionChoices = HermesTaskerConditionBridge.conditionChoices()
        val automationChoices = HermesAutomationStore(applicationContext)
            .list()
            .sortedWith(compareBy<HermesAutomationRecord> { it.label.lowercase() }.thenBy { it.id })
            .map { AutomationChoice(it.id, it.label) }

        val conditionSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@HermesTaskerConditionEditActivity,
                android.R.layout.simple_spinner_dropdown_item,
                conditionChoices,
            )
            val selectedIndex = conditionChoices.indexOfFirst { it.id == existingType }
            if (selectedIndex >= 0) {
                setSelection(selectedIndex)
            }
        }
        val automationSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@HermesTaskerConditionEditActivity,
                android.R.layout.simple_spinner_dropdown_item,
                automationChoices.ifEmpty { listOf(AutomationChoice("", getString(R.string.hermes_tasker_plugin_no_automations))) },
            )
            val selectedIndex = automationChoices.indexOfFirst { it.id == existingAutomationId }
            if (selectedIndex >= 0) {
                setSelection(selectedIndex)
            }
        }
        val automationIdInput = EditText(this).apply {
            hint = getString(R.string.hermes_tasker_plugin_manual_id_hint)
            setSingleLine(true)
            setText(existingAutomationId)
        }
        val variableNameInput = EditText(this).apply {
            hint = getString(R.string.hermes_tasker_condition_variable_hint)
            setSingleLine(true)
            setText(existingVariableName)
        }
        val expectedValueInput = EditText(this).apply {
            hint = getString(R.string.hermes_tasker_condition_expected_value_hint)
            setSingleLine(true)
            setText(existingExpectedValue)
        }
        val labelInput = EditText(this).apply {
            hint = getString(R.string.hermes_tasker_condition_label_hint)
            setSingleLine(true)
            setText(existing?.getString(HermesTaskerConditionBridge.KEY_LABEL).orEmpty())
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
            addView(TextView(this@HermesTaskerConditionEditActivity).apply {
                text = getString(R.string.hermes_tasker_condition_title)
                textSize = 22f
            })
            addView(TextView(this@HermesTaskerConditionEditActivity).apply {
                text = getString(R.string.hermes_tasker_condition_summary)
                textSize = 15f
                setPadding(0, 12, 0, 20)
            })
            addView(TextView(this@HermesTaskerConditionEditActivity).apply {
                text = getString(R.string.hermes_tasker_condition_type)
            })
            addView(conditionSpinner, fullWidthParams())
            addView(TextView(this@HermesTaskerConditionEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_existing_automation)
                setPadding(0, 20, 0, 0)
            })
            addView(automationSpinner, fullWidthParams())
            addView(TextView(this@HermesTaskerConditionEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_manual_id)
                setPadding(0, 20, 0, 0)
            })
            addView(automationIdInput, fullWidthParams())
            addView(TextView(this@HermesTaskerConditionEditActivity).apply {
                text = getString(R.string.hermes_tasker_condition_variable)
                setPadding(0, 20, 0, 0)
            })
            addView(variableNameInput, fullWidthParams())
            addView(TextView(this@HermesTaskerConditionEditActivity).apply {
                text = getString(R.string.hermes_tasker_condition_expected_value)
                setPadding(0, 20, 0, 0)
            })
            addView(expectedValueInput, fullWidthParams())
            addView(TextView(this@HermesTaskerConditionEditActivity).apply {
                text = getString(R.string.hermes_tasker_condition_blurb_label)
                setPadding(0, 20, 0, 0)
            })
            addView(labelInput, fullWidthParams())
            addView(Button(this@HermesTaskerConditionEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_save)
                setOnClickListener {
                    val condition = conditionSpinner.selectedItem as? HermesTaskerConditionBridge.ConditionChoice
                    val selectedAutomation = automationSpinner.selectedItem as? AutomationChoice
                    val automationId = automationIdInput.text.toString().trim().ifBlank { selectedAutomation?.id.orEmpty() }
                    val result = runCatching {
                        HermesTaskerConditionBridge.buildResultIntent(
                            context = this@HermesTaskerConditionEditActivity,
                            conditionType = condition?.id.orEmpty(),
                            automationId = automationId,
                            variableName = variableNameInput.text.toString(),
                            expectedValue = expectedValueInput.text.toString(),
                            label = labelInput.text.toString(),
                            existingToken = existingToken,
                        )
                    }.getOrElse { error ->
                        Toast.makeText(
                            this@HermesTaskerConditionEditActivity,
                            error.message ?: getString(R.string.hermes_tasker_condition_invalid),
                            Toast.LENGTH_SHORT,
                        ).show()
                        return@setOnClickListener
                    }
                    setResult(RESULT_OK, result)
                    finish()
                }
            }, fullWidthParams())
        }

        setContentView(ScrollView(this).apply { addView(root) })
    }

    private fun fullWidthParams(): LinearLayout.LayoutParams {
        return LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        )
    }
}
