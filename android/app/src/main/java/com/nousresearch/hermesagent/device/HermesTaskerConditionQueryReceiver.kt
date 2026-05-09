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
        val bundle = HermesTaskerConditionBridge.bundleFromIntent(intent)
        val resultCode: Int
        val variables: Bundle
        if (HermesTaskerEventBridge.isEventBundle(bundle)) {
            val query = HermesTaskerEventBridge.queryEvent(
                context = context.applicationContext,
                hostIntent = intent,
                bundle = bundle,
            )
            resultCode = query.resultCode
            variables = query.variables
        } else {
            val query = HermesTaskerConditionBridge.queryCondition(
                context = context.applicationContext,
                bundle = bundle,
            )
            resultCode = query.resultCode
            variables = query.variables
        }
        setResultCode(resultCode)
        setResultExtras(Bundle().apply {
            putBundle(HermesTaskerConditionBridge.EXTRA_VARIABLES, variables)
        })
    }
}
