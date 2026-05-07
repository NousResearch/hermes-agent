package com.nousresearch.hermesagent.device

import android.service.quicksettings.Tile
import android.service.quicksettings.TileService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class HermesQuickSettingsTileService : TileService() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onStartListening() {
        super.onStartListening()
        HermesQuickSettingsTileBridge.updateTile(applicationContext, qsTile)
    }

    override fun onClick() {
        super.onClick()
        qsTile?.state = Tile.STATE_ACTIVE
        qsTile?.updateTile()
        scope.launch {
            HermesQuickSettingsTileBridge.runConfiguredAutomationJson(applicationContext)
            withContext(Dispatchers.Main) {
                HermesQuickSettingsTileBridge.updateTile(applicationContext, qsTile)
            }
        }
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }
}
