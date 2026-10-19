import { type Codec, persistentAtom } from '@/lib/persisted'

/**
 * POWER USAGE — how much CPU the renderer may spend on UI effects.
 *
 *  - `normal` (default) — the shipped look, every animation as designed.
 *  - `low` — CPU-heavy widgets are swapped for cheap equivalents. Today that
 *    means the sidebar's running-session cue: the travelling arc is a
 *    240%-scaled repeating-gradient layer Chromium re-rasters every frame for
 *    as long as any session runs (measured 19.5% of one core for an otherwise
 *    idle window; 2.1% with every animation cancelled, 1080p60), so `low`
 *    removes it and the row's own status dot carries the cue with four discrete
 *    opacity steps over three seconds.
 *
 * Deliberately named for the goal, not the widget: future levels (`minimal`,
 * say) can trim more without renaming the setting or its stored key.
 */
export type PowerUsage = 'low' | 'normal'

const STORAGE_KEY = 'hermes.desktop.powerUsage.v1'

// Absence of the key means `normal`: a user who never touched the picker never
// gains a record, and clearing the key returns them to the shipped behaviour.
const codec: Codec<PowerUsage> = {
  decode: raw => (raw === 'low' ? 'low' : 'normal'),
  encode: value => (value === 'normal' ? null : value)
}

export const $powerUsage = persistentAtom<PowerUsage>(STORAGE_KEY, 'normal', codec)

export const POWER_USAGE_ATTRIBUTE = 'data-power-usage'

/**
 * Mirrors the preference onto `:root` so the stylesheet can trim effects
 * without every widget subscribing to this store. Returns its cleanup.
 */
export function installPowerUsageAttribute(): () => void {
  const root = document.documentElement
  const sync = () => root.setAttribute(POWER_USAGE_ATTRIBUTE, $powerUsage.get())

  sync()

  return $powerUsage.subscribe(sync)
}

export function setPowerUsage(usage: PowerUsage) {
  $powerUsage.set(usage)
}
