import { atom } from 'nanostores'

import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.model-pricing.v1'

/** Whether the model picker shows per-model token prices. Off by default: most rows are noise until you're comparing cost. */
export const $showModelPricing = atom(storedBoolean(KEY, false))

$showModelPricing.subscribe(on => persistBoolean(KEY, on))

export function setShowModelPricing(on: boolean) {
  $showModelPricing.set(on)
}

export type ModelPriceUnit = '1k' | 'mtok'

const UNIT_KEY = 'hermes.desktop.model-price-unit.v1'

/** Unit for picker prices: per million tokens (the backend's unit) or per 1K. */
export const $modelPriceUnit = atom<ModelPriceUnit>(storedString(UNIT_KEY) === '1k' ? '1k' : 'mtok')

$modelPriceUnit.subscribe(unit => persistString(UNIT_KEY, unit))

export function setModelPriceUnit(unit: ModelPriceUnit) {
  $modelPriceUnit.set(unit)
}
