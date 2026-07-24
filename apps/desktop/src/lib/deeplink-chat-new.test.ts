import { describe, expect, it } from 'vitest'

import {
  clearStickySessionId,
  cwdLooksSane,
  deeplinkStickyStorageKey,
  normalizeStickySlot,
  readStickySessionId,
  setStickyPending,
  takeStickyPending,
  writeStickySessionId
} from './deeplink-chat-new'

function memoryStorage(): Storage {
  const map = new Map<string, string>()
  return {
    get length() {
      return map.size
    },
    clear() {
      map.clear()
    },
    getItem(key: string) {
      return map.has(key) ? map.get(key)! : null
    },
    key(index: number) {
      return [...map.keys()][index] ?? null
    },
    removeItem(key: string) {
      map.delete(key)
    },
    setItem(key: string, value: string) {
      map.set(key, String(value))
    }
  }
}

describe('cwdLooksSane', () => {
  it('accepts absolute unix paths', () => {
    expect(cwdLooksSane('/Users/trevor/HomeDome')).toBe(true)
    expect(cwdLooksSane('/tmp')).toBe(true)
  })

  it('accepts windows drive and unc paths', () => {
    expect(cwdLooksSane('C:\\Users\\trevor\\proj')).toBe(true)
    expect(cwdLooksSane('D:/work/repo')).toBe(true)
    expect(cwdLooksSane('\\\\server\\share\\repo')).toBe(true)
  })

  it('rejects relative and traversal', () => {
    expect(cwdLooksSane('relative/path')).toBe(false)
    expect(cwdLooksSane('../etc')).toBe(false)
    expect(cwdLooksSane('/Users/../etc')).toBe(false)
    expect(cwdLooksSane('/Users/trevor/..')).toBe(false)
    expect(cwdLooksSane('')).toBe(false)
  })
})

describe('sticky slots', () => {
  it('normalizes slot names', () => {
    expect(normalizeStickySlot(' CEO ')).toBe('ceo')
    expect(normalizeStickySlot('My Project!')).toBe('my-project')
    expect(normalizeStickySlot('')).toBe('')
  })

  it('round-trips sticky session ids', () => {
    const store = memoryStorage()
    writeStickySessionId('ceo', '20260724_session', store)
    expect(readStickySessionId('CEO', store)).toBe('20260724_session')
    expect(deeplinkStickyStorageKey('ceo')).toContain('ceo')
    clearStickySessionId('ceo', store)
    expect(readStickySessionId('ceo', store)).toBe(null)
  })

  it('pending sticky is one-shot', () => {
    const store = memoryStorage()
    setStickyPending('work', store)
    expect(takeStickyPending(store)).toBe('work')
    expect(takeStickyPending(store)).toBe(null)
  })
})
