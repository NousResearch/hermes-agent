import { useStore } from '@nanostores/react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { $interfaceMode, INTERFACE_MODES, setInterfaceMode } from '@/store/interface-mode'
import { $simplifiedAgentMode, setSimplifiedAgentMode } from '@/store/simplified-agent-mode'

import { ListRow, ToggleRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

export function InterfaceModeSettings() {
  const { t } = useI18n()
  const mode = useStore($interfaceMode)
  const simplified = useStore($simplifiedAgentMode)
  const ids = SETTING_IDS.appearance

  return (
    <>
      <ToggleRow
        checked={simplified}
        description={t.simplifiedAgents.description}
        id={settingElementId(ids.simplifiedAgents)}
        label={t.simplifiedAgents.title}
        onChange={setSimplifiedAgentMode}
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
