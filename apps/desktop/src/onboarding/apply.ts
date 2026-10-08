/**
 * What a confirmed answer changes in the app (D25): the accent is the default profile's accent pick
 * (D17, the PR2 preference, not `$accentOverride`), the layout is a preset plus its interface mode
 * with the sidebar open.
 */

import { setInterfaceMode } from '@/store/interface-mode'
import { setSidebarOpen } from '@/store/layout'
import { applyDesktopLayoutPreset } from '@/store/pane-focus'
import { accentPref } from '@/themes/accent-pref'

import { LAYOUTS } from './visuals/options'

export const QUESTIONNAIRE_PROFILE = 'default'

/** The accent picker the questionnaire paints through: the theme context's, when it serves the default profile. */
export interface AccentTarget {
  /** The live profile is the default one, so a pick repaints at once. */
  live: boolean
  current: null | string
  setAccent: (id: null | string) => void
  /** Paints without storing, so quitting before Confirm keeps the stored accent. */
  preview: (id: null | string) => void
  clearPreview: () => void
}

/** Paint a swatch before it is confirmed; answers the revert. Only the live default profile can preview. */
export function previewAccent(target: AccentTarget, id: null | string): (() => void) | null {
  if (!target.live) {
    return null
  }

  target.preview(id)

  return target.clearPreview
}

export function commitAccent(target: AccentTarget, id: null | string): void {
  if (target.live) {
    target.setAccent(id)
  } else {
    accentPref.assign(QUESTIONNAIRE_PROFILE, id)
  }
}

export function commitLayout(id: string): void {
  const layout = LAYOUTS.find(candidate => candidate.id === id)

  if (!layout) {
    return
  }

  setInterfaceMode(layout.mode)
  applyDesktopLayoutPreset(layout.id)
  setSidebarOpen(true)
}
