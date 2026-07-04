import { describe, expect, it } from 'vitest'

import { en } from './en'
import { applyLocaleOverrides } from './overrides'
import type { Translations } from './types'

describe('applyLocaleOverrides', () => {
  it('replaces a string leaf with a string override', () => {
    const result = applyLocaleOverrides(en, { common: { save: '保存する' } })
    expect(result.common.save).toBe('保存する')
    // untouched siblings keep the bundled value
    expect(result.common.cancel).toBe(en.common.cancel)
  })

  it('returns the same reference when nothing changes (no-op)', () => {
    expect(applyLocaleOverrides(en, {})).toBe(en)
    expect(applyLocaleOverrides(en, { common: { save: en.common.save } })).toBe(en)
    expect(applyLocaleOverrides(en, undefined)).toBe(en)
    expect(applyLocaleOverrides(en, 'not an object')).toBe(en)
  })

  it('never replaces a function leaf with a string', () => {
    const before = en.keybinds.subtitle
    const result = applyLocaleOverrides(en, { keybinds: { subtitle: 'これは文字列' } })
    expect(typeof result.keybinds.subtitle).toBe('function')
    expect(result.keybinds.subtitle).toBe(before)
  })

  it('ignores keys that do not exist in the base catalog', () => {
    const result = applyLocaleOverrides(en, { common: { madeUpKey: 'x' }, nonsense: { a: 'b' } })
    expect((result.common as unknown as Record<string, unknown>).madeUpKey).toBeUndefined()
    expect((result as unknown as Record<string, unknown>).nonsense).toBeUndefined()
    expect(result).toBe(en)
  })

  it('ignores a type mismatch (object override over a string leaf)', () => {
    const result = applyLocaleOverrides(en, { common: { save: { nested: 'no' } } })
    expect(result.common.save).toBe(en.common.save)
  })

  it('preserves the exact key set and value types of the base', () => {
    const result = applyLocaleOverrides(en, { common: { save: 'S' }, keybinds: { title: 'T' } })
    const sameShape = (a: unknown, b: unknown): boolean => {
      if (typeof a !== typeof b) return false
      if (a && typeof a === 'object' && !Array.isArray(a)) {
        const ak = Object.keys(a as object).sort()
        const bk = Object.keys(b as object).sort()
        if (ak.join() !== bk.join()) return false
        return ak.every(k => sameShape((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]))
      }
      return true
    }
    expect(sameShape(result as Translations, en)).toBe(true)
  })
})
