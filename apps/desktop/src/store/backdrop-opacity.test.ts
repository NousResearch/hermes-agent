import { beforeEach, describe, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.backdrop-opacity.v1'

async function loadStore() {
  vi.resetModules()

  return import('./backdrop-opacity')
}

describe('backdrop opacity preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('defaults to full opacity', async () => {
    const { $backdropOpacity } = await loadStore()

    expect($backdropOpacity.get()).toBe(100)
    expect(window.localStorage.getItem(KEY)).toBe('100')
  })

  it('loads a persisted opacity', async () => {
    window.localStorage.setItem(KEY, '35')

    const { $backdropOpacity } = await loadStore()

    expect($backdropOpacity.get()).toBe(35)
  })

  it('clamps updated opacity to the slider range', async () => {
    const { $backdropOpacity, setBackdropOpacity } = await loadStore()

    setBackdropOpacity(-8)
    expect($backdropOpacity.get()).toBe(0)
    expect(window.localStorage.getItem(KEY)).toBe('0')

    setBackdropOpacity(142)
    expect($backdropOpacity.get()).toBe(100)
    expect(window.localStorage.getItem(KEY)).toBe('100')
  })
})
