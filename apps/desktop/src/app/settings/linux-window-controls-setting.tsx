import { useEffect, useState } from 'react'

import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import type { LinuxWindowControlsMode, LinuxWindowControlsStatus } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AlertTriangle, RefreshCw } from '@/lib/icons'
import { linuxWindowControlsAvailable } from '@/lib/platform'
import { notifyError } from '@/store/notifications'

import { ListRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

export function LinuxWindowControlsSetting() {
  const { t } = useI18n()
  const a = t.settings.appearance
  const bridge = window.hermesDesktop?.windowControls
  const [status, setStatus] = useState<LinuxWindowControlsStatus | null>(null)

  useEffect(() => {
    let current = true

    bridge
      ?.getLinuxMode?.()
      .then(saved => current && setStatus(saved))
      .catch(error => notifyError(error, t.settings.config.failedLoad))

    return () => {
      current = false
    }
  }, [bridge, t.settings.config.failedLoad])

  const change = async (mode: LinuxWindowControlsMode) => {
    const previous = status

    triggerHaptic('selection')
    setStatus({ mode, restartNeeded: status?.restartNeeded ?? false })

    try {
      const saved = await bridge?.setLinuxMode?.(mode)

      if (saved) {
        setStatus(saved)
      }
    } catch (error) {
      setStatus(previous)
      notifyError(error, t.messaging.failedSave(a.windowControlsTitle))
    }
  }

  const restart = () =>
    window.hermesDesktop?.relaunchApp?.().catch(error => notifyError(error, t.notifications.errors.restartHermesFailed))

  if (!linuxWindowControlsAvailable()) {
    return null
  }

  return (
    <ListRow
      action={
        <SegmentedControl
          onChange={next => void change(next)}
          options={[
            { id: 'auto', label: a.windowControlsAuto },
            { id: 'show', label: a.windowControlsShow },
            { id: 'hide', label: a.windowControlsHide }
          ]}
          value={status?.mode ?? 'auto'}
        />
      }
      below={
        status?.restartNeeded && (
          <Alert className="mt-3" variant="warning">
            <AlertTriangle />
            <AlertDescription className="flex flex-wrap items-center justify-between gap-2">
              <span>{a.windowControlsRestart}</span>
              <Button onClick={() => void restart()} size="sm" variant="secondary">
                <RefreshCw />
                {t.messaging.restartNow}
              </Button>
            </AlertDescription>
          </Alert>
        )
      }
      description={a.windowControlsDesc}
      id={settingElementId(SETTING_IDS.appearance.windowControls)}
      title={a.windowControlsTitle}
    />
  )
}
