import { useStore } from '@nanostores/react'
import { createContext, useContext } from 'react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { $desktopMetricsGate } from '@/store/desktop-metrics'
import { $interfaceMode, INTERFACE_MODES, setInterfaceMode } from '@/store/interface-mode'
import { $portalPreferred, setPortalPreferred } from '@/store/shell-variant'

import { ListRow, ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

interface InterfaceModeFeedbackState {
  showFeedback: boolean
  setShowFeedback: (show: boolean) => void
}

export const InterfaceModeFeedbackContext = createContext<InterfaceModeFeedbackState | null>(null)

export function InterfaceModeSettings() {
  const { t } = useI18n()
  const mode = useStore($interfaceMode)
  const preferred = useStore($portalPreferred)
  const ids = SETTING_IDS.appearance
  const metricsGate = useStore($desktopMetricsGate)
  const { setShowFeedback } = useContext(InterfaceModeFeedbackContext)!

  return (
    <>
      <ToggleRow
        checked={preferred}
        description={t.simplifiedAgents.description}
        id={settingElementId(ids.simplifiedAgents)}
        label={t.simplifiedAgents.title}
        onChange={enabled => {
          setPortalPreferred(enabled)
          setShowFeedback(preferred && !enabled && metricsGate === 'on')
        }}
      />
      <ListRow
        action={
          <SegmentedControl
            onChange={id => {
              triggerHaptic('selection')
              setInterfaceMode(id)
            }}
            options={INTERFACE_MODES.map(id => ({ id, label: t.interfaceMode[id].label }))}
            value={mode}
          />
        }
        description={t.interfaceMode.hint}
        id={settingElementId(ids.interfaceMode)}
        title={t.interfaceMode.title}
      />
    </>
  )
}
