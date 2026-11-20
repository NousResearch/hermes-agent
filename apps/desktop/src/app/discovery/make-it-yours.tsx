import { useStore } from '@nanostores/react'
import { useId } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { recordIaAction } from '@/store/ia-metrics'
import { $portalAccent, pickPortalAccent, PORTAL_ACCENTS } from '@/store/portal-accent'
import { useTheme } from '@/themes/context'

import { useFirstTimeCopy } from './first-time-copy'

const MODES = ['light', 'dark'] as const

/**
 * "Make it yours" (1A 477:4004): light or dark, and one of eight colours, each changing the app as it
 * is picked. Both are native radio groups (arrow keys move and pick, Tab leaves the group). Light /
 * dark is the theme system's own mode: it starts on what the computer shows and a pick sets it for
 * good; the colour is the redesign's accent (store/portal-accent.ts).
 */
export function MakeItYours() {
  const copy = useFirstTimeCopy().splash.theme
  const { renderedMode, setMode } = useTheme()
  const accent = useStore($portalAccent)
  const modeGroup = useId()
  const colorGroup = useId()

  return (
    <div className="discovery-theme-controls">
      <fieldset className="discovery-theme-modes">
        <legend className="sr-only">{copy.mode}</legend>
        {MODES.map(mode => (
          <label className="discovery-theme-mode" key={mode}>
            <input
              checked={renderedMode === mode}
              className="sr-only"
              name={modeGroup}
              onChange={() => {
                triggerHaptic('crisp')
                setMode(mode)
                recordIaAction(`ia.theme_pick.mode_${mode}`)
              }}
              type="radio"
              value={mode}
            />
            <span>{copy[mode]}</span>
          </label>
        ))}
      </fieldset>
      <fieldset className="discovery-theme-colors">
        <legend className="sr-only">{copy.color}</legend>
        {PORTAL_ACCENTS.map(color => (
          <label className="discovery-theme-color" data-accent={color} key={color} title={copy.colors[color]}>
            <input
              aria-label={copy.colors[color]}
              checked={accent === color}
              className="sr-only"
              name={colorGroup}
              onChange={() => {
                triggerHaptic('crisp')
                pickPortalAccent(color)
              }}
              type="radio"
              value={color}
            />
            <span aria-hidden className="discovery-theme-swatch" />
          </label>
        ))}
      </fieldset>
    </div>
  )
}
