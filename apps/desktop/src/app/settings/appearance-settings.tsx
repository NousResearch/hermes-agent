import { useStore } from '@nanostores/react'

import { LanguageSwitcher } from '@/components/language-switcher'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Palette } from '@/lib/icons'
import { $toolViewMode, setToolViewMode } from '@/store/tool-view'
import { useTheme } from '@/themes/context'

import { MODE_OPTIONS } from './constants'
import { ListRow, SectionHeading, SettingsContent } from './primitives'

// Consumer appearance page: 语言 + 颜色模式 + 工具调用显示 only. The theme
// grid, the VS Code theme installer, window translucency and the haptics
// toggle were removed for the consumer IA (the haptics/translucency stores and
// the theme engine itself stay — the ⌘K palette still switches themes).
export function AppearanceSettings() {
  const { t, isSavingLocale } = useI18n()
  const { mode, setMode } = useTheme()
  const toolViewMode = useStore($toolViewMode)
  const a = t.settings.appearance

  const modeOptions = MODE_OPTIONS.map(({ id, icon }) => ({ icon, id, label: t.settings.modeOptions[id].label }))

  const toolOptions = [
    { id: 'product', label: a.product },
    { id: 'technical', label: a.technical }
  ] as const

  return (
    <SettingsContent>
      <div>
        <SectionHeading icon={Palette} title={a.title} />
        <p className="p5-section-intro">{a.intro}</p>

        <div className="p5-card p5-rows mt-3.5">
          <ListRow
            action={<LanguageSwitcher />}
            description={isSavingLocale ? t.language.saving : t.language.description}
            title={t.language.label}
          />

          <ListRow
            action={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('crisp')
                  setMode(id)
                }}
                options={modeOptions}
                value={mode}
              />
            }
            description={a.colorModeDesc}
            title={a.colorMode}
          />

          <ListRow
            action={
              <SegmentedControl
                onChange={id => {
                  triggerHaptic('selection')
                  setToolViewMode(id)
                }}
                options={toolOptions}
                value={toolViewMode}
              />
            }
            description={a.toolViewDesc}
            title={a.toolViewTitle}
          />
        </div>
      </div>
    </SettingsContent>
  )
}
