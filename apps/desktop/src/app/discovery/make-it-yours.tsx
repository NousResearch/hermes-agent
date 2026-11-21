import { useStore } from '@nanostores/react'
import { useId } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { recordIaAction } from '@/store/ia-metrics'
import { $portalAccent, pickPortalAccent, PORTAL_ACCENTS } from '@/store/portal-accent'
import { useTheme } from '@/themes/context'

import { $paintedArtwork, artworksFor, chooseArtwork } from '../redesign/artwork'

import { useFirstTimeCopy } from './first-time-copy'

const MODES = ['light', 'dark'] as const

/**
 * "Make it yours" (Figma 18695:18059): one of eight colours and light or dark in one card, then
 * "Make it beautiful", the artworks the current mode draws. Each changes the app as it is picked, and
 * each is a native radio group (arrow keys move and pick, Tab leaves the group). Light / dark is the
 * theme system's own mode: it starts on what the computer shows and a pick sets it for good; the
 * colour is the redesign's accent (store/portal-accent.ts); the artwork is the person's choice, which
 * every later launch paints instead of rotating (app/redesign/artwork.ts).
 */
export function MakeItYours() {
  const copy = useFirstTimeCopy().splash.theme
  const { renderedMode, setMode } = useTheme()
  const accent = useStore($portalAccent)
  const painted = useStore($paintedArtwork)
  const group = useId()

  return (
    <>
      <div className="discovery-welcome-card discovery-theme-controls">
        <fieldset className="discovery-theme-colors">
          <legend className="sr-only">{copy.color}</legend>
          {PORTAL_ACCENTS.map(color => (
            <label className="discovery-theme-color" data-accent={color} key={color} title={copy.colors[color]}>
              <input
                aria-label={copy.colors[color]}
                checked={accent === color}
                className="sr-only"
                name={`${group}-color`}
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
        <fieldset className="discovery-theme-modes">
          <legend className="sr-only">{copy.mode}</legend>
          {MODES.map(mode => (
            <label className="discovery-theme-mode" key={mode}>
              <input
                checked={renderedMode === mode}
                className="sr-only"
                name={`${group}-mode`}
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
      </div>
      <fieldset className="discovery-artworks">
        <legend className="discovery-welcome-lead-title">{copy.artwork.title}</legend>
        <div className="discovery-artworks-track">
          {artworksFor(renderedMode).map(artwork => {
            const selected = painted?.[renderedMode] === artwork

            return (
              <label className="discovery-artwork" data-artwork={artwork} key={artwork}>
                <input
                  aria-label={copy.artwork.names[artwork]}
                  checked={selected}
                  className="sr-only"
                  name={`${group}-artwork`}
                  onChange={() => {
                    triggerHaptic('crisp')
                    chooseArtwork(artwork)
                  }}
                  type="radio"
                  value={artwork}
                />
                {selected && (
                  <span aria-hidden className="discovery-artwork-tag">
                    {copy.artwork.selected}
                  </span>
                )}
              </label>
            )
          })}
        </div>
      </fieldset>
    </>
  )
}
