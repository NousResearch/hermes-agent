import { useStore } from '@nanostores/react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { $hudOrientation, HUD_ORIENTATIONS, type HudOrientation, setHudOrientation } from '@/store/hud'

import { CONTROL_TEXT } from './constants'
import { ListRow } from './primitives'

/**
 * HUD mode — device-local layout rows.
 *
 * The orientation choice is pure renderer presentation (the window layout is
 * computed in the HUD's own renderer, see app/hud/hud-shell), so unlike Quick
 * Entry there is no main-process round-trip: the store owns it, persists it to
 * localStorage, and the live HUD picks it up reactively while open.
 */
export function HudSettings() {
  const { t } = useI18n()
  const c = t.settings.config
  const orientation = useStore($hudOrientation)

  const labelFor = (value: HudOrientation) =>
    value === 'auto' ? c.hudOrientationAuto : value === 'composer-top' ? c.hudOrientationTop : c.hudOrientationBottom

  return (
    <ListRow
      action={
        <Select onValueChange={value => setHudOrientation(value as HudOrientation)} value={orientation}>
          <SelectTrigger className={CONTROL_TEXT}>
            <SelectValue />
          </SelectTrigger>

          <SelectContent>
            {HUD_ORIENTATIONS.map(value => (
              <SelectItem key={value} value={value}>
                {labelFor(value)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      }
      description={c.hudOrientationDesc}
      title={c.hudOrientationTitle}
    />
  )
}
