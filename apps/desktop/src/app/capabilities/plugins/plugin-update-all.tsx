import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Loader2, RefreshCw } from '@/lib/icons'
import { notify, notifyError } from '@/store/notifications'
import {
  $pluginAutoUpdate,
  $pluginUpdateAllRunning,
  isPluginUpdateAllRunning,
  loadPluginAutoUpdate,
  setPluginAutoUpdate,
  updateAllAgentPlugins
} from '@/store/plugin-update-all'
import { $gatewayState } from '@/store/session'

import { useSettingsGatewayRequest } from '../../settings/settings-owner-request'
import { CatalogSwitch } from '../catalog/catalog-switch'

/** Plugins tab header: "Update all" and the profile's auto-update switch (`plugins.auto_apply`). */
export function PluginUpdateAllActions({ scope }: { scope: null | string }) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const requestGateway = useSettingsGatewayRequest()
  const gatewayState = useStore($gatewayState)
  const running = isPluginUpdateAllRunning(useStore($pluginUpdateAllRunning), scope, requestGateway.ownerKey)
  const auto = useStore($pluginAutoUpdate)
  const autoForScope = auto && auto.profile === scope && auto.ownerKey === requestGateway.ownerKey ? auto : null

  useEffect(() => {
    if (gatewayState === 'open') {
      // An older backend without the action answers 4017: the switch then stays hidden.
      void loadPluginAutoUpdate(requestGateway, scope).catch(() => undefined)
    }
  }, [requestGateway, scope, gatewayState])

  const updateAll = async () => {
    try {
      const summary = await updateAllAgentPlugins(requestGateway, scope)

      if (!summary.rows.length) {
        notify({ kind: 'success', message: p.updateAllNone })

        return
      }

      const message = p.updateAllDone(summary.updated, summary.review, summary.failed)
      const detail = summary.rows.filter(row => row.status !== 'updated' && row.status !== 'current')

      notify({
        kind: summary.failed ? 'warning' : 'success',
        message,
        detail: detail.map(row => `${row.name}: ${row.detail || row.status}`).join('\n') || undefined
      })
    } catch (err) {
      notifyError(err, p.updateAllFailed)
    }
  }

  return (
    <>
      {autoForScope && (
        <Tip label={p.autoUpdateTip(autoForScope.checkHours)}>
          <span className="inline-flex items-center gap-1.5 text-xs text-(--ui-text-secondary)">
            {p.autoUpdate}
            <CatalogSwitch
              aria-label={p.autoUpdate}
              checked={autoForScope.autoApply}
              onCheckedChange={enable =>
                void setPluginAutoUpdate(requestGateway, scope, enable).catch(err =>
                  notifyError(err, p.autoUpdateFailed)
                )
              }
            />
          </span>
        </Tip>
      )}
      <Tip label={p.updateAllTip}>
        <Button disabled={running} onClick={() => void updateAll()} size="xs" variant="ghost">
          {running ? <Loader2 className="animate-spin" /> : <RefreshCw />}
          {running ? p.updatingAll : p.updateAll}
        </Button>
      </Tip>
    </>
  )
}
