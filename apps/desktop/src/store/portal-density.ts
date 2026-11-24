/**
 * The redesign's spacing: comfortable (the default, as Figma draws it) or compact (shorter rows).
 * A shell choice beside the accent (./portal-accent.ts), saved and painted the same way: a
 * machine-level UI preference (`lib/ui-preferences/keys.ts`) mirrored as one attribute on `<html>`
 * (`data-portal-density`, absent at the default) that only the redesign's tokens read
 * (`app/redesign/tokens/space.css`). Classic has its own setting (./session-list-density.ts).
 */

import { persistentAtom } from '@/lib/persisted'

export const PORTAL_DENSITIES = ['comfortable', 'compact'] as const

export type PortalDensity = (typeof PORTAL_DENSITIES)[number]

export const $portalDensity = persistentAtom<PortalDensity>('hermes.desktop.portalDensity.v1', 'comfortable', {
  decode: raw => (raw === 'compact' ? raw : 'comfortable'),
  encode: value => (value === 'comfortable' ? null : value)
})

export function setPortalDensity(density: PortalDensity): void {
  $portalDensity.set(density)
}

/** main.tsx, before first render: mirror the choice onto `<html>` and follow every change. */
export function installPortalDensityAttribute(root: HTMLElement = document.documentElement): () => void {
  return $portalDensity.subscribe(density => {
    if (density === 'comfortable') {
      delete root.dataset.portalDensity
    } else {
      root.dataset.portalDensity = density
    }
  })
}
