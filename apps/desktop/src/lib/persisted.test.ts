import { beforeEach, describe, expect, it } from 'vitest'

import { Codecs, persistentAtom } from './persisted'

// Unique key per test run — persistentAtom instances are singletons per call,
// so tests mint fresh keys instead of sharing one.
let counter = 0
const freshKey = () => `test.persisted.${++counter}`

describe('persistentAtom', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('seeds from a stored value', () => {
    const key = freshKey()
    window.localStorage.setItem(key, JSON.stringify({ a: 1 }))

    const $value = persistentAtom<Record<string, number>>(key, {})

    expect($value.get()).toEqual({ a: 1 })
  })

  it('does NOT write at creation — an empty read must never clobber storage', () => {
    // A cold boot can evaluate the bundle against a storage snapshot that has
    // not caught up yet (early hidden/boot load). If creation echoed the
    // fallback back out, that load would overwrite the real record other
    // loads are about to read. Creation must be read-only.
    const key = freshKey()
    persistentAtom<Record<string, number>>(key, {})

    expect(window.localStorage.getItem(key)).toBeNull()
  })

  it('leaves an existing stored value untouched at creation even when it fails to decode', () => {
    const key = freshKey()
    window.localStorage.setItem(key, 'not json')

    const $value = persistentAtom<Record<string, number>>(key, { fallback: 1 })

    expect($value.get()).toEqual({ fallback: 1 })
    // The corrupt value survives until a real write replaces it — creation
    // never writes.
    expect(window.localStorage.getItem(key)).toBe('not json')
  })

  it('persists real changes', () => {
    const key = freshKey()
    const $value = persistentAtom<Record<string, number>>(key, {})

    $value.set({ b: 2 })

    expect(window.localStorage.getItem(key)).toBe(JSON.stringify({ b: 2 }))
  })

  it('removes the key when the codec encodes null', () => {
    const key = freshKey()
    window.localStorage.setItem(key, JSON.stringify(['x']))

    const $value = persistentAtom<string[]>(key, [], Codecs.stringArray)

    $value.set([])

    expect(window.localStorage.getItem(key)).toBeNull()
  })
})
