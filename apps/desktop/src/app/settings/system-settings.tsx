import { useStore } from '@nanostores/react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Cpu } from '@/lib/icons'
import { $hapticsMuted, setHapticsMuted } from '@/store/haptics'
import { $keepAwake, setKeepAwake } from '@/store/keep-awake'
import { $translucency, setTranslucency } from '@/store/translucency'
import { $zoomPercent, setZoomPercent } from '@/store/zoom'

import { ListRow, SectionHeading, SettingsContent } from './primitives'

// UI scale presets as zoom percentages (100 = browser default); the ids double
// as the percent sent to main. A Cmd/Ctrl +/- step between presets highlights
// nothing, and the row description keeps showing the exact current percent.
const UI_SCALE_PRESETS = ['90', '100', '110', '125', '150', '175'] as const

type UiScalePreset = (typeof UI_SCALE_PRESETS)[number]

function ToggleRow(props: { checked: boolean; description: string; label: string; onChange: (on: boolean) => void }) {
  return (
    <ListRow
      action={
        <Switch
          aria-label={props.label}
          checked={props.checked}
          onCheckedChange={on => {
            triggerHaptic('selection')
            props.onChange(on)
          }}
        />
      }
      description={props.description}
      title={props.label}
    />
  )
}

export function SystemSettings() {
  const { t } = useI18n()
  const s = t.settings.system
  const keepAwake = useStore($keepAwake)
  const translucency = useStore($translucency)
  const zoomPercent = useStore($zoomPercent)
  const hapticsMuted = useStore($hapticsMuted)

  const uiScaleOptions = UI_SCALE_PRESETS.map(preset => ({ id: preset, label: `${preset}%` }))
  const matchedScale = UI_SCALE_PRESETS.find(preset => Number(preset) === zoomPercent) ?? ('' as UiScalePreset)

  return (
    <SettingsContent>
      <SectionHeading icon={Cpu} title={s.title} />
      <p className="mb-2 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {s.intro}
      </p>

      <ToggleRow checked={keepAwake} description={s.keepAwakeDesc} label={s.keepAwakeTitle} onChange={setKeepAwake} />

      <ListRow
        action={
          <SegmentedControl
            onChange={id => {
              triggerHaptic('selection')
              setZoomPercent(Number(id))
            }}
            options={uiScaleOptions}
            value={matchedScale}
          />
        }
        description={s.uiScaleDesc(zoomPercent)}
        title={s.uiScaleTitle}
      />

      <ListRow
        action={
          <div className="flex items-center gap-3">
            <input
              aria-label={s.translucencyTitle}
              className="h-1 w-40 cursor-pointer appearance-none rounded-full bg-(--ui-stroke-tertiary)"
              max={100}
              min={0}
              onChange={event => {
                triggerHaptic('selection')
                setTranslucency(Number(event.target.value))
              }}
              step={5}
              style={{ accentColor: 'var(--dt-primary)' }}
              type="range"
              value={translucency}
            />
            <span className="w-9 text-right text-[length:var(--conversation-caption-font-size)] tabular-nums text-(--ui-text-tertiary)">
              {translucency}%
            </span>
          </div>
        }
        description={s.translucencyDesc}
        title={s.translucencyTitle}
      />

      <ToggleRow
        checked={!hapticsMuted}
        description={s.hapticsDesc}
        label={s.hapticsTitle}
        onChange={on => setHapticsMuted(!on)}
      />
    </SettingsContent>
  )
}
