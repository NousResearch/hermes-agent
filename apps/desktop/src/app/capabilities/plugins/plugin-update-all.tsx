import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Loader2, RefreshCw } from '@/lib/icons'
import { checkAgentPlugins } from '@/store/agent-plugins'
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

/** Plugins tab header: "Update all" and the profile's auto-update switch (`plugins.auto_apply`).
 *  `checkForUpdates` adds the on-demand update check (the agent's own Plugins page), gated so the
 *  Capabilities header keeps its classic actions. */
export function PluginUpdateAllActions({
  scope,
  checkForUpdates = false
}: {
  scope: null | string
  checkForUpdates?: boolean
}) {
  const { t } = useI18n()
  const p = t.skills.plugins
  const requestGateway = useSettingsGatewayRequest()
  const gatewayState = useStore($gatewayState)
  const running = isPluginUpdateAllRunning(useStore($pluginUpdateAllRunning), scope, requestGateway.ownerKey)
  const auto = useStore($pluginAutoUpdate)
  const autoForScope = auto && auto.profile === scope && auto.ownerKey === requestGateway.ownerKey ? auto : null
  const [checking, setChecking] = useState(false)

  useEffect(() => {
    if (gatewayState === 'open') {
      // An older backend without the action answers 4017: the switch then stays hidden.
      void loadPluginAutoUpdate(requestGateway, scope).catch(() => undefined)
    }
  }, [requestGateway, scope, gatewayState])

  const check = async () => {
    setChecking(true)

    try {
      const summary = await checkAgentPlugins(requestGateway, scope)

      if (!summary.supported) {
        notify({ kind: 'warning', message: p.checkUnsupported })

        return
      }

      notify({
        kind: 'success',
        message: summary.updatesAvailable ? p.updatesAvailable(summary.updatesAvailable) : p.updateAllNone
      })
    } catch (err) {
      notifyError(err, p.checkFailed)
    } finally {
      setChecking(false)
    }
  }

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
      {checkForUpdates && (
        <Button disabled={checking} onClick={() => void check()} size="xs" variant="ghost">
          {checking ? <Loader2 className="animate-spin" /> : <RefreshCw />}
          {checking ? p.checkingForUpdates : p.checkForUpdates}
        </Button>
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
