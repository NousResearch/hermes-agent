import { Codecs, persistentAtom } from '@/lib/persisted'

const STORAGE_KEY = 'hermes.desktop.reducedEffects.v1'

export const $reducedEffects = persistentAtom(STORAGE_KEY, false, Codecs.bool)

export function setReducedEffects(enabled: boolean): void {
  $reducedEffects.set(enabled)
}

if (typeof document !== 'undefined') {
  $reducedEffects.subscribe(enabled => {
    document.documentElement.toggleAttribute('data-reduced-effects', enabled)
  })
}
