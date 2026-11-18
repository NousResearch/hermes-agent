/**
 * The redesign's accent colour ("Make it yours", first-time splash): which colour its brand gesture
 * wears. The redesign's brand belongs to the shell, not the skin (DESIGN.md, Portal design language),
 * so this is a shell choice beside the theme system's light / dark mode, not a skin retint.
 *
 * Saved where the mode is saved: a machine-level UI preference (`lib/ui-preferences/keys.ts`), which
 * is this browser until a visitor has a machine, then shared by every window of it; it moves to the
 * account with the other appearance preferences. Painted by one attribute on `<html>`
 * (`data-portal-accent`, absent for the default blue) that only the redesign's tokens read
 * (`app/redesign/tokens/raw.css`), so classic never changes colour.
 */

import { persistentAtom } from '@/lib/persisted'

import { recordIaAction } from './ia-metrics'

/** In the order the swatches show. Blue is the redesign's own brand. */
export const PORTAL_ACCENTS = ['blue', 'violet', 'green', 'orange', 'pink'] as const

export type PortalAccent = (typeof PORTAL_ACCENTS)[number]

export const PORTAL_ACCENT_KEY = 'hermes.desktop.portalAccent.v1'

const isAccent = (value: string): value is PortalAccent => (PORTAL_ACCENTS as readonly string[]).includes(value)

export const $portalAccent = persistentAtom<PortalAccent>(PORTAL_ACCENT_KEY, 'blue', {
  decode: raw => (isAccent(raw) ? raw : 'blue'),
  encode: value => (value === 'blue' ? null : value)
})

export function pickPortalAccent(accent: PortalAccent): void {
  if ($portalAccent.get() === accent) {
    return
  }

  $portalAccent.set(accent)
  recordIaAction(`ia.theme_pick.color_${accent}`)
}

/** main.tsx, before first render: mirror the accent onto `<html>` and follow every change. */
export function installPortalAccentAttribute(root: HTMLElement = document.documentElement): () => void {
  return $portalAccent.subscribe(accent => {
    if (accent === 'blue') {
      delete root.dataset.portalAccent
    } else {
      root.dataset.portalAccent = accent
    }
  })
}
