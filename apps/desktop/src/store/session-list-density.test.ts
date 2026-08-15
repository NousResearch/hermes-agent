import { beforeEach, describe, expect, it, vi } from 'vitest'

const loadStore = async () => {
  vi.resetModules()

  return import('./session-list-density')
}

describe('session list density preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('defaults to comfortable and persists changes', async () => {
    const first = await loadStore()

    expect(first.$sessionListDensity.get()).toBe('comfortable')

    first.setSessionListDensity('detailed')

    expect(window.localStorage.getItem('hermes.desktop.sessionListDensity')).toBe('detailed')
    expect((await loadStore()).$sessionListDensity.get()).toBe('detailed')
  })

  it('falls back to comfortable for an unknown stored value', async () => {
    window.localStorage.setItem('hermes.desktop.sessionListDensity', 'tiny')

    expect((await loadStore()).$sessionListDensity.get()).toBe('comfortable')
  })
})
