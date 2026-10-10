import { describe, expect, it } from 'vitest'

import { $modelPriceUnit, setModelPriceUnit } from './model-pricing'

describe('model price unit', () => {
  it('defaults to per-million and remembers a per-1K choice', () => {
    expect($modelPriceUnit.get()).toBe('mtok')
    setModelPriceUnit('1k')
    expect($modelPriceUnit.get()).toBe('1k')
    expect(window.localStorage.getItem('hermes.desktop.model-price-unit.v1')).toBe('1k')
  })
})
