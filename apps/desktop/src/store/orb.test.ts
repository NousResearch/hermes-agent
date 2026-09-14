import { beforeEach, describe, expect, it } from 'vitest'

import {
  $orbConfigUrl,
  $orbEnabled,
  $orbParams,
  $orbUrlError,
  setOrbConfigUrl,
  setOrbEnabled
} from './orb'

const GOOD_URL = 'https://lersent001.github.io/orb/#effect=orb-glass-liquid&style=aurora&speed=1.5'

beforeEach(() => {
  window.localStorage.clear()
  $orbEnabled.set(false)
  $orbConfigUrl.set('')
})

describe('orb store', () => {
  it('is disabled with the default orb out of the box', () => {
    expect($orbEnabled.get()).toBe(false)
    expect($orbConfigUrl.get()).toBe('')
    expect($orbParams.get().style).toBe('siri')
    expect($orbUrlError.get()).toBeNull()
  })

  it('persists the toggle', () => {
    setOrbEnabled(true)

    expect($orbEnabled.get()).toBe(true)
    expect(window.localStorage.getItem('hermes.desktop.orb.enabled')).toBe('true')
  })

  it('parses a valid custom URL into params', () => {
    setOrbConfigUrl(GOOD_URL)

    expect($orbConfigUrl.get()).toBe(GOOD_URL)
    expect($orbParams.get().style).toBe('aurora')
    expect($orbParams.get().speed).toBeCloseTo(1.5)
    expect($orbUrlError.get()).toBeNull()
  })

  it('falls back to the default orb for an invalid URL', () => {
    setOrbConfigUrl('https://example.com/not-an-orb')

    expect($orbParams.get().style).toBe('siri')
    expect($orbUrlError.get()).toBe('no-params')
  })

  it('clears the custom URL on reset', () => {
    setOrbConfigUrl(GOOD_URL)
    setOrbConfigUrl('  ')

    expect($orbConfigUrl.get()).toBe('')
    expect($orbParams.get().style).toBe('siri')
    expect($orbUrlError.get()).toBeNull()
    expect(window.localStorage.getItem('hermes.desktop.orb.configUrl')).toBeNull()
  })
})
