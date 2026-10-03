import { Codecs, persistentAtom } from '@/lib/persisted'

/** Desktop-local presentation preference; applies to both sidebar edges. */
export const $sidebarHoverReveal = persistentAtom('hermes.desktop.sidebarHoverReveal', true, Codecs.bool)

export function setSidebarHoverReveal(enabled: boolean) {
  $sidebarHoverReveal.set(enabled)
}
