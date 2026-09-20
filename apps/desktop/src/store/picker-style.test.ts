import { afterEach, expect, it, vi } from 'vitest'

import { PICKER_STYLES, pickerBehavior } from './picker-style'

afterEach(() => {
  localStorage.removeItem('hermes.desktop.pickerStyle')
})
it('keeps effort visible and editable in every style', () => {
  for (const style of PICKER_STYLES) {
    const b = pickerBehavior(style)
    expect(b.separateReasoningPill || b.effortInModelLabel).toBe(true)
    expect(b.separateReasoningPill || b.modelSubmenu !== 'hidden').toBe(true)
  }
})
it('round trips preferences and falls back safely for an unknown stored style', async () => {
  for (const style of PICKER_STYLES) {
    vi.resetModules()
    localStorage.setItem('hermes.desktop.pickerStyle', style)
    const store = await import('./picker-style')
    expect(store.$pickerStyle.get()).toBe(style)
  }

  vi.resetModules()
  localStorage.setItem('hermes.desktop.pickerStyle', 'future-style')
  const store = await import('./picker-style')
  expect(store.$pickerStyle.get()).toBe(store.DEFAULT_PICKER_STYLE)
  store.setPickerStyle('classic')
  expect(localStorage.getItem('hermes.desktop.pickerStyle')).toBe('classic')
})
