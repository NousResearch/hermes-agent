import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.model-pricing.v1'

/** Whether the model picker shows per-model token prices. Off by default: most rows are noise until you're comparing cost. */
export const $showModelPricing = atom(storedBoolean(KEY, false))

$showModelPricing.subscribe(on => persistBoolean(KEY, on))

export function setShowModelPricing(on: boolean) {
  $showModelPricing.set(on)
}
