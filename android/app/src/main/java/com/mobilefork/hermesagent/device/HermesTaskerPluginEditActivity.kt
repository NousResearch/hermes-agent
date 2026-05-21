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

class HermesTaskerPluginEditActivity : Activity() {
    private data class AutomationChoice(val id: String, val label: String) {
        override fun toString(): String = if (label.isBlank() || label == id) id else "$label ($id)"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val existing = HermesTaskerPluginBridge.bundleFromIntent(intent)
        val existingId = existing?.getString(HermesTaskerPluginBridge.KEY_AUTOMATION_ID).orEmpty()
        val existingToken = existing?.getString(HermesTaskerPluginBridge.KEY_TOKEN).orEmpty()
        val choices = HermesAutomationStore(applicationContext)
            .list()
            .sortedWith(compareBy<HermesAutomationRecord> { it.label.lowercase() }.thenBy { it.id })
            .map { AutomationChoice(it.id, it.label) }

        val spinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@HermesTaskerPluginEditActivity,
                android.R.layout.simple_spinner_dropdown_item,
                choices.ifEmpty { listOf(AutomationChoice("", getString(R.string.hermes_tasker_plugin_no_automations))) },
            )
            val selectedIndex = choices.indexOfFirst { it.id == existingId }
            if (selectedIndex >= 0) {
                setSelection(selectedIndex)
            }
        }
        val idInput = EditText(this).apply {
            hint = getString(R.string.hermes_tasker_plugin_manual_id_hint)
            setSingleLine(true)
            setText(existingId)
        }
        val labelInput = EditText(this).apply {
            hint = getString(R.string.hermes_tasker_plugin_label_hint)
            setSingleLine(true)
            setText(existing?.getString(HermesTaskerPluginBridge.KEY_LABEL).orEmpty())
        }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
            addView(TextView(this@HermesTaskerPluginEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_title)
                textSize = 22f
            })
            addView(TextView(this@HermesTaskerPluginEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_summary)
                textSize = 15f
                setPadding(0, 12, 0, 20)
            })
            addView(TextView(this@HermesTaskerPluginEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_existing_automation)
            })
            addView(spinner, fullWidthParams())
            addView(TextView(this@HermesTaskerPluginEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_manual_id)
                setPadding(0, 20, 0, 0)
            })
            addView(idInput, fullWidthParams())
            addView(TextView(this@HermesTaskerPluginEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_blurb_label)
                setPadding(0, 20, 0, 0)
            })
            addView(labelInput, fullWidthParams())
            addView(Button(this@HermesTaskerPluginEditActivity).apply {
                text = getString(R.string.hermes_tasker_plugin_save)
                setOnClickListener {
                    val selected = spinner.selectedItem as? AutomationChoice
                    val automationId = idInput.text.toString().trim().ifBlank { selected?.id.orEmpty() }
                    if (automationId.isBlank()) {
                        Toast.makeText(
                            this@HermesTaskerPluginEditActivity,
                            R.string.hermes_tasker_plugin_missing_id,
                            Toast.LENGTH_SHORT,
                        ).show()
                        return@setOnClickListener
                    }
                    val label = labelInput.text.toString().trim().ifBlank { selected?.label.orEmpty() }
                    val result = HermesTaskerPluginBridge.buildResultIntent(
                        this@HermesTaskerPluginEditActivity,
                        automationId,
                        label,
                        existingToken,
                    )
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
