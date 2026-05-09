package com.nousresearch.hermesagent.device

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Bundle

class HermesTaskerConditionQueryReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != HermesTaskerConditionBridge.ACTION_QUERY_CONDITION) {
            return
        }
        val query = HermesTaskerConditionBridge.queryCondition(
            context = context.applicationContext,
            bundle = HermesTaskerConditionBridge.bundleFromIntent(intent),
        )
        setResultCode(query.resultCode)
        setResultExtras(Bundle().apply {
            putBundle(HermesTaskerConditionBridge.EXTRA_VARIABLES, query.variables)
        })
    }
}
