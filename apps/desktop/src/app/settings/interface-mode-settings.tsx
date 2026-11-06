import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { $desktopMetricsGate } from '@/store/desktop-metrics'
import { $interfaceMode, INTERFACE_MODES, setInterfaceMode } from '@/store/interface-mode'
import { $simplifiedAgentMode, setSimplifiedAgentMode } from '@/store/simplified-agent-mode'

import { ListRow, ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { SimplifiedAgentFeedback } from './simplified-agent-feedback'

export function InterfaceModeSettings() {
  const { t } = useI18n()
  const mode = useStore($interfaceMode)
  const simplified = useStore($simplifiedAgentMode)
  const ids = SETTING_IDS.appearance
  const metricsGate = useStore($desktopMetricsGate)
  const [showFeedback, setShowFeedback] = useState(false)

  return (
    <>
      <ToggleRow
        checked={simplified}
        description={t.simplifiedAgents.description}
        id={settingElementId(ids.simplifiedAgents)}
        label={t.simplifiedAgents.title}
        onChange={enabled => {
          setSimplifiedAgentMode(enabled)
          setShowFeedback(simplified && !enabled && metricsGate === 'on')
        }}
      />
      {showFeedback && metricsGate === 'on' && <SimplifiedAgentFeedback />}
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
