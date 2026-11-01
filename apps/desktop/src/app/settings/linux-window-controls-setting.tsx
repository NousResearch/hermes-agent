import { useEffect, useState } from 'react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import type { LinuxWindowControlsMode } from '@/global'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { linuxWindowControlsAvailable } from '@/lib/platform'
import { notifyError } from '@/store/notifications'

import { ListRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

export function LinuxWindowControlsSetting() {
  const { t } = useI18n()
  const a = t.settings.appearance
  const bridge = window.hermesDesktop?.windowControls
  const [mode, setMode] = useState<LinuxWindowControlsMode | null>(null)

  useEffect(() => {
    let current = true

    bridge
      ?.getLinuxMode?.()
      .then(saved => current && setMode(saved))
      .catch(error => notifyError(error, t.settings.config.failedLoad))

    return () => {
      current = false
    }
  }, [bridge, t.settings.config.failedLoad])

  const change = async (next: LinuxWindowControlsMode) => {
    const previous = mode

    triggerHaptic('selection')
    setMode(next)

    try {
      await bridge?.setLinuxMode?.(next)
    } catch (error) {
      setMode(previous)
      notifyError(error, t.messaging.failedSave(a.windowControlsTitle))
    }
  }

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
          value={mode ?? 'auto'}
        />
      }
      description={a.windowControlsDesc}
      id={settingElementId(SETTING_IDS.appearance.windowControls)}
      title={a.windowControlsTitle}
    />
  )
}
