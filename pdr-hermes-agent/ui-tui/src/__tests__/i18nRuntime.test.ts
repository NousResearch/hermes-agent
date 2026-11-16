import { afterEach, describe, expect, it } from 'vitest'

import { en, EN_SIBLINGS } from '../i18n/en.js'
import { flattenKeys } from '../i18n/keys.js'
import { fetchCatalogPack, normalizeLanguageId, resetTuiLocaleSync, syncTuiLocale } from '../i18n/loader.js'
import { $catalog, $locale, applyLocale, catalogFromPack, formatPositional, resetLocale, t } from '../i18n/runtime.js'

afterEach(() => {
  resetLocale()
  resetTuiLocaleSync()
})

describe('TUI i18n catalog', () => {
  it('siblings own disjoint top-level namespaces', () => {
    const seen = new Map<string, number>()

    EN_SIBLINGS.forEach((sibling, index) => {
      for (const key of Object.keys(sibling)) {
        expect(seen.has(key), `namespace "${key}" appears in siblings ${seen.get(key)} and ${index}`).toBe(false)
        seen.set(key, index)
      }
    })
  })

  it('every leaf is a string or a function', () => {
    const check = (node: unknown, path: string) => {
      if (typeof node === 'string' || typeof node === 'function') {
        return
      }

      expect(node, path).toBeTypeOf('object')

      for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
        check(v, `${path}.${k}`)
      }
    }

    check(en, 'en')
  })

  it('flattenKeys emits sorted dotted leaves', () => {
    const keys = flattenKeys({ b: { y: 'y', x: () => 'x' }, a: 'a' })

    expect(keys).toEqual(['a', 'b.x', 'b.y'])
    expect(flattenKeys(en)).toContain('status.ready')
  })
})

describe('TUI i18n runtime', () => {
  it('resolves active → en → key', () => {
    expect(t('status.ready')).toBe(en.status.ready)
    applyLocale('pl', { lang: 'pl', surface: 'tui', messages: { 'status.ready': 'gotowy' } })
    expect($locale.get()).toBe('pl')
    expect(t('status.ready')).toBe('gotowy')
    expect(t('status.running')).toBe(en.status.running)
    expect(t('nope.missing' as never)).toBe('nope.missing')
  })

  it('wraps a string override of a function leaf into a positional formatter', () => {
    const base = { greet: { hello: (name: string, n: number) => `hi ${name} x${n}` } }
    const merged = catalogFromPack.call(null, {}) // en, untouched
    expect(merged).toBe(en)

    const wrapped = formatPositional('cześć {0} ×{1} {2}', ['Ala', 3])
    expect(wrapped).toBe('cześć Ala ×3 {2}')

    // nestPack semantics through the public seam: a function leaf in en stays callable.
    const fnKeys = flattenKeys(en).filter(k => typeof k.split('.').reduce<any>((c, p) => c?.[p], en) === 'function')

    for (const key of fnKeys.slice(0, 3)) {
      applyLocale('pl', { lang: 'pl', surface: 'tui', messages: { [key]: 'X {0} Y' } })
      const leaf = key.split('.').reduce<any>((c, p) => c?.[p], $catalog.get())
      expect(typeof leaf).toBe('function')
      expect(leaf('a')).toBe('X a Y')
    }

    void base
  })

  it('drops pack keys that en does not know', () => {
    applyLocale('pl', { lang: 'pl', surface: 'tui', messages: { 'status.bogus': 'x', 'status.ready': 'ok' } })
    expect(($catalog.get().status as Record<string, unknown>).bogus).toBeUndefined()
    expect($catalog.get().status.ready).toBe('ok')
  })
})

describe('TUI i18n loader', () => {
  it('normalizes display.language like the backend', () => {
    expect(normalizeLanguageId(' PT_BR ')).toBe('pt-br')
    expect(normalizeLanguageId('')).toBe('en')
    expect(normalizeLanguageId(undefined)).toBe('en')
  })

  it('stays English when the backend lacks i18n.catalog', async () => {
    const request = async () => {
      throw new Error('Method not found: i18n.catalog')
    }

    expect(await fetchCatalogPack(request as never, 'pl')).toBeNull()
    await syncTuiLocale({ request } as never, 'pl')
    expect($locale.get()).toBe('pl')
    expect(t('status.ready')).toBe(en.status.ready)
  })

  it('applies the pack for display.language and skips the RPC for English', async () => {
    const calls: unknown[] = []

    const request = async (method: string, params: Record<string, unknown>) => {
      calls.push([method, params])

      return { lang: 'pl', surface: 'tui', messages: { 'status.ready': 'gotowy' } }
    }

    await syncTuiLocale({ request } as never, 'pl')
    expect(calls).toEqual([['i18n.catalog', { lang: 'pl', surface: 'tui' }]])
    expect(t('status.ready')).toBe('gotowy')

    await syncTuiLocale({ request } as never, 'pl')
    expect(calls).toHaveLength(1)

    await syncTuiLocale({ request } as never, 'en')
    expect(calls).toHaveLength(1)
    expect(t('status.ready')).toBe(en.status.ready)
  })
})
