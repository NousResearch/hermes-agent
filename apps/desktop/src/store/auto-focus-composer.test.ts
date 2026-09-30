import { beforeEach, describe, expect, it } from 'vitest'

import { storedBoolean } from '@/lib/storage'

import { $autoFocusComposer, setAutoFocusComposer } from './auto-focus-composer'

const KEY = 'hermes.desktop.autoFocusComposer.v1'

beforeEach(() => {
  setAutoFocusComposer(false)
  window.localStorage.removeItem(KEY)
})

describe('auto-focus composer store', () => {
  it('is off by default and persists flips through localStorage', () => {
    expect($autoFocusComposer.get()).toBe(false)

    setAutoFocusComposer(true)
    expect($autoFocusComposer.get()).toBe(true)
    expect(storedBoolean(KEY, false)).toBe(true)

    setAutoFocusComposer(false)
    expect($autoFocusComposer.get()).toBe(false)
    expect(storedBoolean(KEY, true)).toBe(false)
  })
})
