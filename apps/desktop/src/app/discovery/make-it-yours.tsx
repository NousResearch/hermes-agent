import { useStore } from '@nanostores/react'
import { useId } from 'react'

import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { recordIaAction } from '@/store/ia-metrics'
import { $portalAccent, pickPortalAccent, PORTAL_ACCENTS } from '@/store/portal-accent'
import { $portalDensity, PORTAL_DENSITIES, setPortalDensity } from '@/store/portal-density'
import { useTheme } from '@/themes/context'

import { $artworkChoice, $paintedArtwork, artworksFor, chooseArtwork, NO_ARTWORK, setArtworkShown } from '../redesign/artwork'
import { appearanceAdvancedCopy } from '../settings/appearance-advanced-copy'

import { useFirstTimeCopy } from './first-time-copy'

const MODES = ['light', 'dark'] as const

/**
 * "Make it yours" (Figma 18695:18059): eight colours, light or dark, comfortable or compact
 * spacing, and the artworks (including a plain no-background option). Shared across first-time
 * onboarding, returning-user splash, and Settings › Appearance.
 */
export function MakeItYours() {
  const { locale, t } = useI18n()
  const copy = useFirstTimeCopy().splash.theme
  const [spacingTitle] = appearanceAdvancedCopy(locale).density
  const spacingLabels = {
    comfortable: t.settings.appearance.sessionDensityComfortable,
    compact: t.settings.appearance.sessionDensityCompact
  }
  const { renderedMode, setMode } = useTheme()
  const accent = useStore($portalAccent)
  const density = useStore($portalDensity)
  const choice = useStore($artworkChoice)
  const painted = useStore($paintedArtwork)
  const group = useId()
  const noArtworkSelected = choice === NO_ARTWORK

  return (
    <>
      <div className="discovery-welcome-card discovery-theme-controls">
        <div className="discovery-theme-controls-row">
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
        <div className="discovery-theme-density-row">
          <span className="discovery-theme-density-label">{spacingTitle}</span>
          <fieldset className="discovery-theme-modes">
            <legend className="sr-only">{spacingTitle}</legend>
            {PORTAL_DENSITIES.map(value => (
              <label className="discovery-theme-mode" key={value}>
                <input
                  checked={density === value}
                  className="sr-only"
                  name={`${group}-density`}
                  onChange={() => {
                    triggerHaptic('crisp')
                    setPortalDensity(value)
                  }}
                  type="radio"
                  value={value}
                />
                <span>{spacingLabels[value]}</span>
              </label>
            ))}
          </fieldset>
        </div>
      </div>
      <fieldset className="discovery-artworks">
        <legend className="discovery-welcome-lead-title">{copy.artwork.title}</legend>
        <div className="discovery-artworks-track">
          {artworksFor(renderedMode).map(artwork => {
            const selected = !noArtworkSelected && painted?.[renderedMode] === artwork

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
          <label className="discovery-artwork" data-artwork="none">
            <input
              aria-label="No background"
              checked={noArtworkSelected}
              className="sr-only"
              name={`${group}-artwork`}
              onChange={() => {
                triggerHaptic('crisp')
                setArtworkShown(false)
              }}
              type="radio"
              value={NO_ARTWORK}
            />
            {noArtworkSelected && (
              <span aria-hidden className="discovery-artwork-tag">
                {copy.artwork.selected}
              </span>
            )}
            <span aria-hidden className="discovery-artwork-none-label">
              No background
            </span>
          </label>
        </div>
      </fieldset>
    </>
  )
}
