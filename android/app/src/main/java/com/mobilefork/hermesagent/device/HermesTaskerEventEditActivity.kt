package com.mobilefork.hermesagent.device

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
import com.mobilefork.hermesagent.R

class HermesTaskerEventEditActivity : Activity() {
    private data class AutomationChoice(val id: String, val label: String) {
        override fun toString(): String = if (label.isBlank() || label == id) id else "$label ($id)"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val existing = HermesTaskerEventBridge.bundleFromIntent(intent)
        val existingType = existing?.getString(HermesTaskerEventBridge.KEY_EVENT_TYPE).orEmpty()
        val existingAutomationId = existing?.getString(HermesTaskerEventBridge.KEY_AUTOMATION_ID).orEmpty()
        val existingToken = existing?.getString(HermesTaskerEventBridge.KEY_TOKEN).orEmpty()
        val eventChoices = HermesTaskerEventBridge.eventChoices()
        val automationChoices = HermesAutomationStore(applicationContext)
            .list()
            .sortedWith(compareBy<HermesAutomationRecord> { it.label.lowercase() }.thenBy { it.id })
            .map { AutomationChoice(it.id, it.label) }

        val eventSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@HermesTaskerEventEditActivity,
                android.R.layout.simple_spinner_dropdown_item,
                eventChoices,
            )
            val selectedIndex = eventChoices.indexOfFirst { it.id == existingType }
            if (selectedIndex >= 0) {
                setSelection(selectedIndex)
            }
        }
        val automationSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@HermesTaskerEventEditActivity,
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
        val labelInput = EditText(this).apply {
            hint = getString(R.string.hermes_tasker_event_label_hint)
            setSingleLine(true)
            setText(existing?.getString(HermesTaskerEventBridge.KEY_LABEL).orEmpty())
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
            addView(TextView(this@HermesTaskerEventEditActivity).apply {
                text = getString(R.string.hermes_tasker_event_title)
                textSize = 22f
            })
            addView(TextView(this@HermesTaskerEventEditActivity).apply {
                text = getString(R.string.hermes_tasker_event_summary)
                textSize = 15f
                setPadding(0, 12, 0, 20)
            })
            addView(TextView(this@HermesTaskerEventEditActivity).apply {
                text = getString(R.string.hermes_tasker_event_type)
            })
            addView(eventSpinner, fullWidthParams())
            addView(TextView(this@HermesTaskerEventEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_existing_automation)
                setPadding(0, 20, 0, 0)
            })
            addView(automationSpinner, fullWidthParams())
            addView(TextView(this@HermesTaskerEventEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_manual_id)
                setPadding(0, 20, 0, 0)
            })
            addView(automationIdInput, fullWidthParams())
            addView(TextView(this@HermesTaskerEventEditActivity).apply {
                text = getString(R.string.hermes_tasker_event_blurb_label)
                setPadding(0, 20, 0, 0)
            })
            addView(labelInput, fullWidthParams())
            addView(Button(this@HermesTaskerEventEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_save)
                setOnClickListener {
                    val event = eventSpinner.selectedItem as? HermesTaskerEventBridge.EventChoice
                    val selectedAutomation = automationSpinner.selectedItem as? AutomationChoice
                    val automationId = automationIdInput.text.toString().trim().ifBlank { selectedAutomation?.id.orEmpty() }
                    val result = runCatching {
                        HermesTaskerEventBridge.buildResultIntent(
                            context = this@HermesTaskerEventEditActivity,
                            eventType = event?.id.orEmpty(),
                            automationId = automationId,
                            label = labelInput.text.toString(),
                            existingToken = existingToken,
                        )
                    }.getOrElse { error ->
                        Toast.makeText(
                            this@HermesTaskerEventEditActivity,
                            error.message ?: getString(R.string.hermes_tasker_event_invalid),
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
