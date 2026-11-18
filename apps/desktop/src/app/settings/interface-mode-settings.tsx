import { useStore } from '@nanostores/react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { $interfaceMode, INTERFACE_MODES, setInterfaceMode } from '@/store/interface-mode'

import { ListRow } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

export function InterfaceModeSettings() {
  const { t } = useI18n()
  const mode = useStore($interfaceMode)
  const ids = SETTING_IDS.appearance

  return (
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
  )
}
