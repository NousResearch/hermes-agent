// The gentle-pi sidebar portrait, fetched once from the main process. Branding
// data is app-wide and read by the sidebar footer, so it lives in an atom like
// every other piece of sidebar state — not in component-local state.

import { atom } from 'nanostores'

import type { SidebarPortraitData } from '@/global'

export const $sidebarPortrait = atom<SidebarPortraitData | null>(null)

let requested = false

/** One-shot load; repeated mounts reuse the in-flight/resolved value. */
export function ensureSidebarPortrait(): void {
  if (requested) {
    return
  }
  requested = true

  void window.hermesDesktop.sidebarPortrait.get().then(portrait => {
    $sidebarPortrait.set(portrait)
  })
}
