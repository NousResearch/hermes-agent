import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorBanner } from '@/components/ui/error-state'
import { useI18n } from '@/i18n'

import type { HudModifierStatus } from '../../../electron/hud-modifier-types'

import { ToggleRow } from './primitives'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

const isHudModifierSetting = (target: string) => target === 'hud-modifier'
const hudModifierElementId = () => 'setting-hud-modifier'

export function HudModifierSettings() {
  const { t } = useI18n()
  const copy = t.settings.hudModifier
  const common = t.settings.screenshot
  const api = window.hermesDesktop?.hudModifier
  const [status, setStatus] = useState<HudModifierStatus | null>(null)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const request = useRef(0)
  const revision = useRef(0)
  useDeepLinkHighlight({ param: 'setting', ready: isHudModifierSetting, elementId: hudModifierElementId })

  const refresh = useCallback(
    async (enabled?: boolean) => {
      if (!api) {
        return
      }

      const id = ++request.current
      const version = revision.current
      setBusy(true)
      setError(null)

      try {
        const next = await (enabled === undefined ? api.getSettings() : api.setEnabled(enabled))

        if (id === request.current && version === revision.current) {
          setStatus(next)
        }
      } catch {
        if (id === request.current) {
          setError(enabled === undefined ? common.loadFailed : common.saveFailed)
        }
      } finally {
        if (id === request.current) {
          setBusy(false)
        }
      }
    },
    [api, common.loadFailed, common.saveFailed]
  )

  // eslint-disable-next-line no-restricted-syntax -- native IPC subscription, not a store mirror.
  useEffect(() => {
    if (!api) {
      return
    }

    const unsubscribe = api.onStatus(next => {
      revision.current += 1
      setStatus(next)
      setError(null)
    })

    void refresh()

    return () => {
      request.current += 1
      unsubscribe()
    }
  }, [api, refresh])

  if (!api) {
    return null
  }

  const openPermissionSettings = async () => {
    try {
      await api.openPermissionSettings()
    } catch {
      setError(common.permissionFailed)
    }
  }

  const notice =
    error ??
    (status?.enabled && status.state === 'input-permission' ? copy.permission : null) ??
    (status?.enabled && status.state === 'unavailable' ? copy.unavailable : null)

  return (
    <div id={hudModifierElementId()}>
      <ToggleRow
        checked={status?.enabled ?? false}
        description={copy.description}
        disabled={!status || busy}
        label={copy.title}
        onChange={enabled => void refresh(enabled)}
      />
      {notice && (
        <div className="space-y-2" role="alert">
          <ErrorBanner>{notice}</ErrorBanner>
          <div className="flex flex-wrap items-center gap-2">
            {status?.state === 'input-permission' && (
              <Button disabled={busy} onClick={() => void openPermissionSettings()} size="sm" variant="secondary">
                {common.openSettings}
              </Button>
            )}
            <Button
              disabled={busy}
              onClick={() => void refresh(error ? undefined : status?.enabled)}
              size="sm"
              variant="secondary"
            >
              {common.retry}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
