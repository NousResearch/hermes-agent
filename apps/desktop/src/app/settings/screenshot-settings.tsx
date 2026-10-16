import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorIcon } from '@/components/ui/error-state'
import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'

import type { ScreenshotStatus } from '../../../electron/command-screenshot-types'

import { ListRow, ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

type SettingsError = 'loadFailed' | 'saveFailed' | 'permissionFailed'

export function ScreenshotSettings() {
  const { t } = useI18n()
  const s = t.settings.screenshot
  const api = window.hermesDesktop?.screenshot
  const [status, setStatus] = useState<ScreenshotStatus | null>(null)
  const [busy, setBusy] = useState(true)
  const [error, setError] = useState<SettingsError | null>(null)
  const requestId = useRef(0)
  const statusVersion = useRef(0)

  const refresh = useCallback(
    async (enabled?: boolean) => {
      if (!api) {
        return
      }

      const id = ++requestId.current
      const version = statusVersion.current
      setBusy(true)
      setError(null)

      try {
        const next = await (enabled === undefined ? api.getSettings() : api.setEnabled(enabled))

        // Native events must not be overwritten by an older IPC snapshot.
        if (id === requestId.current && version === statusVersion.current) {
          setStatus(next)
        }
      } catch {
        if (id === requestId.current && (enabled !== undefined || version === statusVersion.current)) {
          setError(enabled === undefined ? 'loadFailed' : 'saveFailed')
        }
      } finally {
        if (id === requestId.current) {
          setBusy(false)
        }
      }
    },
    [api]
  )

  // eslint-disable-next-line no-restricted-syntax -- IPC request/status generations, not atom mirrors.
  useEffect(() => {
    if (!api) {
      return
    }

    const unsubscribe = api.onStatus(next => {
      statusVersion.current += 1
      setStatus(next)
      setError(null)
    })

    void refresh()

    return () => {
      requestId.current += 1
      unsubscribe()
    }
  }, [api, refresh])

  if (!api) {
    return null
  }

  const permission =
    status?.state === 'input-permission' ? 'input' : status?.state === 'screen-permission' ? 'screen' : null

  const openSettings = async () => {
    if (!permission) {
      return
    }

    const id = ++requestId.current
    setBusy(true)
    setError(null)

    try {
      await api.openPermissionSettings(permission)
    } catch {
      if (id === requestId.current) {
        setError('permissionFailed')
      }
    } finally {
      if (id === requestId.current) {
        setBusy(false)
      }
    }
  }

  const descriptions: Record<ScreenshotStatus['state'], string> = {
    disabled: s.disabled,
    starting: s.starting,
    ready: status?.enabled ? s.ready : s.disabled,
    'input-permission': s.inputPermission,
    'screen-permission': s.screenPermission,
    unavailable: s.unavailable
  }

  const description = busy ? s.checking : error ? s[error] : descriptions[status?.state ?? 'disabled']
  const showStatus = busy || error || status?.enabled || status?.state !== 'disabled'
  const canRetry = error || (status?.state !== 'ready' && status?.state !== 'disabled')
  // Failed reads/writes need reconciliation; permission recovery needs a restart.
  const retryEnabled = status?.enabled && error !== 'loadFailed' && error !== 'saveFailed' ? true : undefined

  return (
    <>
      <ToggleRow
        checked={status?.enabled ?? false}
        description={s.enabledDesc}
        disabled={!status || busy}
        id={settingElementId(SETTING_IDS.keybinds.screenshot)}
        label={s.enabledTitle}
        onChange={enabled => void refresh(enabled)}
      />
      {showStatus && (
        <ListRow
          action={
            busy ? (
              <Loader className="size-5" label={s.checking} />
            ) : (
              <div className="flex flex-wrap items-center gap-2">
                {permission && (
                  <Button onClick={() => void openSettings()} size="sm" variant="secondary">
                    {s.openSettings}
                  </Button>
                )}
                {canRetry && (
                  <Button onClick={() => void refresh(retryEnabled)} size="sm" variant="secondary">
                    {s.retry}
                  </Button>
                )}
              </div>
            )
          }
          description={<span aria-live="polite">{description}</span>}
          title={
            error ? (
              <span className="flex items-center gap-2">
                <ErrorIcon size="1rem" />
                {s.errorTitle}
              </span>
            ) : (
              s.statusTitle
            )
          }
        />
      )}
    </>
  )
}
