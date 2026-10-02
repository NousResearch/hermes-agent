import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { storedBoolean } from '@/lib/storage'

import { $openLinksExternally, setOpenLinksExternally } from './open-links-externally'

const KEY = 'hermes.desktop.openLinksExternally.v1'

beforeEach(() => {
  setOpenLinksExternally(false)
})

afterEach(() => {
  setOpenLinksExternally(false)
  window.localStorage.clear()
})

describe('open-links-externally store', () => {
  it('defaults to off (in-app browser is the default)', () => {
    expect($openLinksExternally.get()).toBe(false)
    expect(storedBoolean(KEY, false)).toBe(false)
  })

  it('persists the pref across reads', () => {
    setOpenLinksExternally(true)
    expect($openLinksExternally.get()).toBe(true)
    expect(storedBoolean(KEY, false)).toBe(true)

    setOpenLinksExternally(false)
    expect($openLinksExternally.get()).toBe(false)
    expect(storedBoolean(KEY, true)).toBe(false)
  })
})
